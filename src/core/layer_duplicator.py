"""
Layer Duplication for HuggingFace Transformers Models

This module provides functionality to duplicate/rearrange transformer layers
while sharing weights (memory efficient), similar to ExLlamaV2's approach.

Key insight: We create SHALLOW COPIES of layers and update their layer_idx.
This way:
- Weights are shared (shallow copy references same tensors)
- Each layer position has unique layer_idx for proper KV cache indexing
"""

import copy
from contextlib import contextmanager
from typing import Optional
import torch
import torch.nn as nn
from src.core.layer_config import (
    baseline_layers,
    expand_multi_block_config as _expand_multi_block_config,
    expand_single_block as _expand_single_block,
    ij_to_layers,
    parse_blocks_string as _parse_blocks_string,
    parse_layer_list_string as _parse_layer_list_string,
)


def _get_text_layer_owner(base_model) -> tuple[object, str]:
    """Return (owner, attr) for decoder layers across causal and conditional models."""
    if hasattr(base_model, "model") and hasattr(base_model.model, "layers"):
        return base_model.model, "layers"

    if (
        hasattr(base_model, "model")
        and hasattr(base_model.model, "language_model")
        and hasattr(base_model.model.language_model, "layers")
    ):
        return base_model.model.language_model, "layers"

    if hasattr(base_model, "language_model") and hasattr(base_model.language_model, "layers"):
        return base_model.language_model, "layers"

    raise AttributeError(
        "Could not locate text decoder layers. "
        "Expected one of: model.layers, model.language_model.layers, language_model.layers."
    )


def _rebind_accelerate_hook(original_module: nn.Module, copied_module: nn.Module) -> None:
    """
    Re-bind Accelerate hooks on copied modules.

    `accelerate` rewrites `forward` with a `functools.partial` bound to the source module.
    After `copy.copy`, that stale binding still points at the original module, which can
    break duplicated-layer cache indexing (e.g. multiple copied layers reusing one cache slot).
    """
    source_hook = getattr(original_module, "_hf_hook", None)
    if source_hook is None:
        return

    try:
        from accelerate.hooks import add_hook_to_module
    except Exception:
        return

    # Drop copied hook state that still references the source module instance.
    if hasattr(copied_module, "_hf_hook"):
        delattr(copied_module, "_hf_hook")
    if hasattr(copied_module, "_old_forward"):
        delattr(copied_module, "_old_forward")

    # Restore class forward, then attach a fresh hook instance bound to copied_module.
    copied_module.forward = type(copied_module).forward.__get__(copied_module, type(copied_module))
    add_hook_to_module(copied_module, copy.copy(source_hook), append=False)


def _shallow_copy_layer(layer, new_layer_idx: int):
    """
    Create a shallow copy of a decoder layer with updated layer_idx.

    The shallow copy shares all weight tensors with the original,
    but has its own layer_idx for proper cache indexing.

    Key insight: copy.copy() on nn.Module shares the _modules dict,
    so we need to explicitly create new dicts for _modules, _parameters, etc.
    """
    # Shallow copy the layer
    new_layer = copy.copy(layer)

    # CRITICAL: Create independent _modules dict so assignments don't affect original
    new_layer._modules = dict(layer._modules)

    # Update layer_idx on the attention module
    # This is critical for proper KV cache indexing
    if hasattr(layer, 'self_attn'):
        # Create shallow copy of attention with its own _modules dict
        new_attn = copy.copy(layer.self_attn)
        new_attn._modules = dict(layer.self_attn._modules)
        new_attn.layer_idx = new_layer_idx
        _rebind_accelerate_hook(layer.self_attn, new_attn)
        new_layer._modules['self_attn'] = new_attn

    # Qwen3.5 linear-attention layers use `linear_attn` instead of `self_attn`.
    if hasattr(layer, "linear_attn"):
        new_linear_attn = copy.copy(layer.linear_attn)
        new_linear_attn._modules = dict(layer.linear_attn._modules)
        new_linear_attn.layer_idx = new_layer_idx
        _rebind_accelerate_hook(layer.linear_attn, new_linear_attn)
        new_layer._modules["linear_attn"] = new_linear_attn

    _rebind_accelerate_hook(layer, new_layer)
    return new_layer


class LayerDuplicatedModel(nn.Module):
    """
    Wrapper that enables layer duplication with weight sharing.

    Creates shallow copies of layers with updated layer_idx values
    to ensure proper KV cache indexing while sharing weights.
    """

    def __init__(self, base_model, layer_indices: list[int]):
        """
        Args:
            base_model: HuggingFace causal LM model (e.g., Qwen2ForCausalLM)
            layer_indices: List of layer indices to use in forward pass.
                          Can repeat indices to duplicate layers.
                          e.g., [0,1,2,3,3,4,5] duplicates layer 3
        """
        super().__init__()
        self.base_model = base_model
        self.layer_indices = layer_indices
        self.config = base_model.config

        # Store reference to original layers (as a plain list, not ModuleList)
        self._layers_owner, self._layers_attr = _get_text_layer_owner(base_model)
        self._original_layers = list(getattr(self._layers_owner, self._layers_attr))
        self._original_num_layers = len(self._original_layers)
        self._original_module_list = getattr(self._layers_owner, self._layers_attr)

        # Validate indices
        for idx in layer_indices:
            if idx < 0 or idx >= self._original_num_layers:
                raise ValueError(
                    f"Layer index {idx} out of range [0, {self._original_num_layers})"
                )

        # New configuration
        self._new_num_layers = len(layer_indices)

        # Build the new layer sequence with shallow copies
        # Each copy has its own layer_idx for proper cache indexing
        new_layer_list = []
        for new_pos, orig_idx in enumerate(layer_indices):
            layer_copy = _shallow_copy_layer(self._original_layers[orig_idx], new_pos)
            new_layer_list.append(layer_copy)

        # Gemma-style architectures (e.g. Gemma 4) carry a learned per-layer
        # scalar buffer applied at the end of each decoder layer:
        # `hidden_states *= layer_scalar`. Each layer is calibrated for its
        # training position. Naively duplicating a layer applies its scalar
        # twice, which over-attenuates the residual stream (Gemma 4 31B's mid-
        # stack scalars run ~0.43-0.55, so two applications give ~0.19, far
        # off the calibrated gain) and the model degrades into garbage.
        # Fix: when a layer appears N times in the new sequence, set each
        # copy's scalar to s^(1/N) so the cumulative multiplicative gain
        # equals one baseline application. No-op on architectures without
        # a layer_scalar buffer (Llama, Qwen, etc.).
        from collections import Counter
        occurrence_counts = Counter(layer_indices)
        for new_pos, orig_idx in enumerate(layer_indices):
            n = occurrence_counts[orig_idx]
            if n <= 1:
                continue
            layer_copy = new_layer_list[new_pos]
            buf = layer_copy._buffers.get("layer_scalar")
            if buf is None:
                continue
            layer_copy._buffers = dict(layer_copy._buffers)
            new_scalar = buf.float().pow(1.0 / n).to(buf.dtype).contiguous()
            layer_copy._buffers["layer_scalar"] = new_scalar

        self._new_layers = nn.ModuleList(new_layer_list)

    @property
    def device(self):
        return next(self.base_model.parameters()).device

    def get_layer_sequence(self) -> list[nn.Module]:
        """Returns the sequence of layers based on layer_indices."""
        return [self._original_layers[i] for i in self.layer_indices]

    @contextmanager
    def _apply_layer_config(self):
        """Context manager to apply and restore layer configuration."""
        # Store originals
        orig_layers = getattr(self._layers_owner, self._layers_attr)
        orig_num_layers = getattr(self.base_model.config, "num_hidden_layers", None)
        orig_layer_types = getattr(self.base_model.config, 'layer_types', None)
        orig_text_cfg = getattr(self.base_model.config, "text_config", None)
        orig_text_num_layers = (
            getattr(orig_text_cfg, "num_hidden_layers", None)
            if orig_text_cfg is not None
            else None
        )
        orig_text_layer_types = (
            getattr(orig_text_cfg, "layer_types", None)
            if orig_text_cfg is not None
            else None
        )

        try:
            # Apply new configuration BEFORE any forward pass
            setattr(self._layers_owner, self._layers_attr, self._new_layers)
            if orig_num_layers is not None:
                self.base_model.config.num_hidden_layers = self._new_num_layers

            # Keep model-specific layer metadata aligned with duplicated layer order.
            # Some architectures (e.g. Qwen3.5) require exact layer type patterns.
            if orig_layer_types is not None:
                if len(orig_layer_types) == self._original_num_layers:
                    self.base_model.config.layer_types = [
                        orig_layer_types[i] for i in self.layer_indices
                    ]
                else:
                    self.base_model.config.layer_types = list(orig_layer_types)

            if orig_text_cfg is not None and orig_text_num_layers is not None:
                orig_text_cfg.num_hidden_layers = self._new_num_layers

            if orig_text_cfg is not None and orig_text_layer_types is not None:
                if len(orig_text_layer_types) == self._original_num_layers:
                    orig_text_cfg.layer_types = [
                        orig_text_layer_types[i] for i in self.layer_indices
                    ]
                else:
                    orig_text_cfg.layer_types = list(orig_text_layer_types)

            yield
        finally:
            # Restore originals
            setattr(self._layers_owner, self._layers_attr, orig_layers)
            if orig_num_layers is not None:
                self.base_model.config.num_hidden_layers = orig_num_layers
            if orig_layer_types is not None:
                self.base_model.config.layer_types = orig_layer_types
            if orig_text_cfg is not None and orig_text_num_layers is not None:
                orig_text_cfg.num_hidden_layers = orig_text_num_layers
            if orig_text_cfg is not None and orig_text_layer_types is not None:
                orig_text_cfg.layer_types = orig_text_layer_types

    def generate(self, *args, **kwargs):
        """Delegate generation to the modified model."""
        # Force fresh cache by ensuring use_cache creates new cache
        # Don't pass existing past_key_values
        if 'past_key_values' in kwargs:
            kwargs.pop('past_key_values')

        with self._apply_layer_config():
            return self.base_model.generate(*args, **kwargs)

    def forward(self, *args, **kwargs):
        """Forward pass using custom layer sequence."""
        # Force fresh cache if needed
        if 'past_key_values' in kwargs and kwargs['past_key_values'] is not None:
            # Check if cache size matches our layer count
            cache = kwargs['past_key_values']
            if hasattr(cache, 'key_cache'):
                cache_layers = len(cache.key_cache)
            elif isinstance(cache, tuple):
                cache_layers = len(cache)
            else:
                cache_layers = 0

            if cache_layers != self._new_num_layers:
                # Cache size mismatch - need fresh cache
                kwargs['past_key_values'] = None

        with self._apply_layer_config():
            return self.base_model(*args, **kwargs)


def build_model_with_layers(model, layer_indices: list[int]) -> LayerDuplicatedModel:
    """
    Create a model with a custom layer configuration.

    This is the main entry point for layer duplication experiments.

    Args:
        model: HuggingFace causal LM model (e.g., Qwen2ForCausalLM)
        layer_indices: List of layer indices to include in forward pass.
                      Can repeat indices to duplicate layers.
                      Examples:
                        - [0,1,2,...,35] = original 36-layer model
                        - [0,1,2,3,3,4,5,...,35] = duplicate layer 3
                        - list(range(20)) + list(range(17, 36)) = duplicate 17-19

    Returns:
        LayerDuplicatedModel wrapper with the specified layer configuration
    """
    return LayerDuplicatedModel(model, layer_indices)


def generate_layer_dict(num_layers: int) -> dict[tuple[int, int], list[int]]:
    """
    Generate a dictionary of layer configurations for systematic testing.

    This mirrors the generateLayerDict function from the ExLlamaV2 codebase.

    Args:
        num_layers: Number of layers in the original model

    Returns:
        Dictionary mapping (repeat_start, repeat_end) to layer index lists.
        Key (0, 0) represents the baseline (unmodified) model.
        Key (i, j) means: layers 0..j-1 followed by layers i..num_layers-1
                         (effectively duplicating layers i..j-1)
    """
    layers_dict: dict[tuple[int, int], list[int]] = {(0, 0): baseline_layers(num_layers)}
    for j in range(num_layers + 1):
        for i in range(j):
            layers_dict[(i, j)] = ij_to_layers(num_layers, i, j)
    return layers_dict


def expand_single_block(num_layers: int, block: tuple[int, int]) -> list[int]:
    """
    Expand a single (i, j) block specification into a layer index list.

    Args:
        num_layers: Original model layer count
        block: (start, end) tuple where layers start..end-1 are duplicated

    Returns:
        Layer index list

    Example for 8-layer model:
        block=(3, 6) → [0,1,2,3,4,5,3,4,5,6,7]
        (layers 3,4,5 are duplicated)
    """
    return _expand_single_block(num_layers, block)


def expand_multi_block_config(num_layers: int, blocks: tuple[tuple[int, int], ...]) -> list[int]:
    """
    Expand a multi-block specification into a layer index list.

    Each block (i, j) duplicates layers i..j-1 by inserting them after layer j-1.
    Blocks are applied sequentially - each block modifies the result of the previous.

    Args:
        num_layers: Original model layer count
        blocks: Tuple of (start, end) pairs, e.g., ((3,6), (4,6))

    Returns:
        Layer index list

    Example for 8-layer model:
        blocks=((3,6),) → [0,1,2,3,4,5,3,4,5,6,7]
        blocks=((3,6),(4,6)) → [0,1,2,3,4,5,3,4,5,4,5,6,7]

    How sequential application works:
        1. Start with baseline [0,1,2,3,4,5,6,7]
        2. Apply (3,6): insert 3,4,5 after position 5 → [0,1,2,3,4,5,3,4,5,6,7]
        3. Apply (4,6): insert 4,5 after position where 5 appears → [0,1,2,3,4,5,3,4,5,4,5,6,7]
    """
    return _expand_multi_block_config(num_layers, blocks)


def parse_blocks_string(blocks_str: str) -> tuple[tuple[int, int], ...]:
    """
    Parse a blocks string like "3,6;4,6" into ((3,6), (4,6)).

    Args:
        blocks_str: Semicolon-separated pairs, e.g., "3,6;4,6" or "3,6"

    Returns:
        Tuple of (start, end) pairs
    """
    return _parse_blocks_string(blocks_str)


def parse_layer_list_string(layers_str: str) -> list[int]:
    """
    Parse a layer list string like "0,1,2,3,2,3,4,5" into [0,1,2,3,2,3,4,5].

    Args:
        layers_str: Comma-separated layer indices

    Returns:
        List of layer indices
    """
    return _parse_layer_list_string(layers_str)


def get_memory_usage():
    """Get current GPU memory usage in MB."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024 / 1024
    return 0


def print_model_info(model, layer_indices: Optional[list[int]] = None):
    """Print information about the model configuration."""
    if hasattr(model, 'config'):
        config = model.config
        print(f"Model type: {config.model_type}")
        print(f"Original layers: {config.num_hidden_layers}")

    if layer_indices is not None:
        print(f"Custom layer sequence length: {len(layer_indices)}")

        # Find duplicated layers
        from collections import Counter
        counts = Counter(layer_indices)
        duplicates = {k: v for k, v in counts.items() if v > 1}
        if duplicates:
            print(f"Duplicated layers: {duplicates}")
        else:
            print("No layer duplication")

    print(f"GPU memory used: {get_memory_usage():.1f} MB")
