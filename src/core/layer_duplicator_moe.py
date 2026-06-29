"""
Layer Duplication for HuggingFace Transformers MoE Models

This module provides functionality to duplicate/rearrange transformer layers
while sharing weights (memory efficient) for Mixture of Experts (MoE) models.

Key insight: We create SHALLOW COPIES of layers (including MoE components)
and update their layer_idx. This way:
- Weights are shared (shallow copy references same tensors)
- Expert weights remain shared across duplicated layers
- Each layer position has unique layer_idx for proper KV cache indexing
- Router/gate networks maintain independent _modules dicts
"""

import copy
import os
from contextlib import contextmanager
from typing import Optional
import torch
import torch.nn as nn

# Re-export multi-block functions from dense module (shared logic)
from src.core.layer_duplicator import (
    _get_text_layer_owner,
    _rebind_accelerate_hook,
    expand_multi_block_config,
    expand_single_block,
    parse_blocks_string,
    parse_layer_list_string,
)

# Deep-copying MoE MLP internals for every duplicated layer is very CPU-expensive
# on large expert models. Default to the lightweight path (share MLP modules)
# and keep a runtime escape hatch for debugging.
_MOE_DEEP_COPY_MLP = os.environ.get("LEVELGEN_MOE_DEEP_COPY_MLP", "").strip().lower() in {
    "1",
    "true",
    "yes",
}

__all__ = [
    "LayerDuplicatedModelMoE",
    "build_model_with_layers_moe",
    "generate_layer_dict_strategic",
    "generate_layer_dict",
    "expand_single_block",
    "expand_multi_block_config",
    "parse_blocks_string",
    "parse_layer_list_string",
    "get_memory_usage",
    "print_model_info",
]


def _shallow_copy_layer_moe(layer, new_layer_idx: int):
    """
    Create a shallow copy of a MoE decoder layer with updated layer_idx.

    The shallow copy shares all weight tensors with the original,
    but has its own layer_idx for proper cache indexing.

    Handles MoE-specific components:
    - layer.mlp.gate (router network)
    - layer.mlp.experts (ModuleList of 128 experts)

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

    if hasattr(layer, "linear_attn"):
        new_linear_attn = copy.copy(layer.linear_attn)
        new_linear_attn._modules = dict(layer.linear_attn._modules)
        new_linear_attn.layer_idx = new_layer_idx
        _rebind_accelerate_hook(layer.linear_attn, new_linear_attn)
        new_layer._modules["linear_attn"] = new_linear_attn

    # Optional deep-copy path for MoE MLP internals.
    # Default behavior intentionally shares MLP/router/expert modules because
    # copying these nested module trees dominates CPU time on large MoE models.
    if _MOE_DEEP_COPY_MLP and hasattr(layer, 'mlp'):
        # Check if it's a MoE block (has 'gate' and 'experts')
        if hasattr(layer.mlp, 'gate') and hasattr(layer.mlp, 'experts'):
            # Shallow copy the MoE block
            new_mlp = copy.copy(layer.mlp)
            new_mlp._modules = dict(layer.mlp._modules)
            _rebind_accelerate_hook(layer.mlp, new_mlp)

            # Shallow copy the gate (router) to get independent _modules dict
            new_gate = copy.copy(layer.mlp.gate)
            new_gate._modules = dict(layer.mlp.gate._modules)
            _rebind_accelerate_hook(layer.mlp.gate, new_gate)
            new_mlp._modules['gate'] = new_gate

            # experts ModuleList is shared (no copying needed)
            # The experts weights remain shared across all duplicated layers

            # Handle shared_expert if it exists (Qwen3 style)
            if hasattr(layer.mlp, 'shared_expert') and layer.mlp.shared_expert is not None:
                new_shared_expert = copy.copy(layer.mlp.shared_expert)
                new_shared_expert._modules = dict(layer.mlp.shared_expert._modules)
                _rebind_accelerate_hook(layer.mlp.shared_expert, new_shared_expert)
                new_mlp._modules['shared_expert'] = new_shared_expert

            if hasattr(layer.mlp, 'shared_expert_gate') and layer.mlp.shared_expert_gate is not None:
                new_shared_expert_gate = copy.copy(layer.mlp.shared_expert_gate)
                new_shared_expert_gate._modules = dict(layer.mlp.shared_expert_gate._modules)
                _rebind_accelerate_hook(layer.mlp.shared_expert_gate, new_shared_expert_gate)
                new_mlp._modules['shared_expert_gate'] = new_shared_expert_gate

            # Handle shared_experts if it exists (GLM-4 style - plural)
            if hasattr(layer.mlp, 'shared_experts') and layer.mlp.shared_experts is not None:
                new_shared_experts = copy.copy(layer.mlp.shared_experts)
                new_shared_experts._modules = dict(layer.mlp.shared_experts._modules)
                _rebind_accelerate_hook(layer.mlp.shared_experts, new_shared_experts)
                new_mlp._modules['shared_experts'] = new_shared_experts

            new_layer._modules['mlp'] = new_mlp
        else:
            # Dense MLP layer (for models with mlp_only_layers)
            new_mlp = copy.copy(layer.mlp)
            new_mlp._modules = dict(layer.mlp._modules)
            _rebind_accelerate_hook(layer.mlp, new_mlp)
            new_layer._modules['mlp'] = new_mlp

    _rebind_accelerate_hook(layer, new_layer)
    return new_layer


class LayerDuplicatedModelMoE(nn.Module):
    """
    Wrapper that enables layer duplication with weight sharing for MoE models.

    Creates shallow copies of layers with updated layer_idx values
    to ensure proper KV cache indexing while sharing weights (including
    expert weights).
    """

    def __init__(self, base_model, layer_indices: list[int]):
        """
        Args:
            base_model: HuggingFace MoE causal LM model (e.g., Qwen3MoeForCausalLM)
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
            layer_copy = _shallow_copy_layer_moe(self._original_layers[orig_idx], new_pos)
            new_layer_list.append(layer_copy)

        # Gemma-style architectures carry a learned per-layer scalar buffer
        # applied at the end of each decoder layer. Naive duplication applies
        # the scalar N times, over-attenuating the residual stream. Set each
        # copy's scalar to s^(1/N) so cumulative gain matches one baseline
        # application. No-op when the buffer is absent. See the dense
        # duplicator for the full rationale.
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


def build_model_with_layers_moe(model, layer_indices: list[int]) -> LayerDuplicatedModelMoE:
    """
    Create a MoE model with a custom layer configuration.

    This is the main entry point for MoE layer duplication experiments.

    Args:
        model: HuggingFace MoE causal LM model (e.g., Qwen3MoeForCausalLM)
        layer_indices: List of layer indices to include in forward pass.
                      Can repeat indices to duplicate layers.
                      Examples:
                        - [0,1,2,...,47] = original 48-layer model
                        - [0,1,2,3,3,4,5,...,47] = duplicate layer 3
                        - list(range(20)) + list(range(17, 48)) = duplicate 17-19

    Returns:
        LayerDuplicatedModelMoE wrapper with the specified layer configuration
    """
    return LayerDuplicatedModelMoE(model, layer_indices)


def generate_layer_dict_strategic(num_layers: int) -> dict[tuple[int, int], list[int]]:
    """
    Generate a strategic subset of layer configurations for MoE testing.

    This creates ~92 carefully selected configs to test key hypotheses:
    - Early, middle, and late layer duplications
    - Small and large block duplications
    - Strategic spacing patterns

    Args:
        num_layers: Number of layers in the original model (e.g., 48 for Qwen3-30B)

    Returns:
        Dictionary mapping (repeat_start, repeat_end) to layer index lists.
        Key (0, 0) represents the baseline (unmodified) model.
        Key (i, j) means: layers 0..j-1 followed by layers i..num_layers-1
                         (effectively duplicating layers i..j-1)
    """
    layers_dict = {}
    layers_dict[(0, 0)] = list(range(num_layers))  # Baseline

    # Early layer duplication (first 25% of layers)
    # For 48 layers: 0-11
    early_end = num_layers // 4
    for i in range(early_end):
        j = i + 1
        layers_dict[(i, j)] = list(range(0, j)) + list(range(i, num_layers))

    # Middle layer duplication (37.5-62.5% of layers)
    # For 48 layers: 18-29
    middle_start = num_layers * 3 // 8
    middle_end = num_layers * 5 // 8
    for i in range(middle_start, middle_end):
        j = i + 1
        layers_dict[(i, j)] = list(range(0, j)) + list(range(i, num_layers))

    # Late layer duplication (last 25% of layers)
    # For 48 layers: 36-47
    late_start = num_layers * 3 // 4
    for i in range(late_start, num_layers):
        j = min(i + 1, num_layers)
        if j <= num_layers:
            layers_dict[(i, j)] = list(range(0, j)) + list(range(i, num_layers))

    # 3-layer blocks at different depths
    # For 48 layers: (0,3), (5,8), (10,13), (15,18), (20,23), (25,28), (30,33), (35,38), (40,43), (45,48)
    for start in range(0, num_layers, 5):
        end = min(start + 3, num_layers)
        layers_dict[(start, end)] = list(range(0, end)) + list(range(start, num_layers))

    # 5-layer blocks
    # For 48 layers: (10,15), (20,25), (30,35), (40,45), (42,47)
    for start in [num_layers // 5, num_layers * 2 // 5, num_layers * 3 // 5, num_layers * 5 // 6, num_layers * 7 // 8]:
        end = min(start + 5, num_layers)
        layers_dict[(start, end)] = list(range(0, end)) + list(range(start, num_layers))

    # Large duplications (8-layer blocks)
    # For 48 layers: (0,8), (8,16), (16,24), (24,32), (32,40), (40,48)
    for start in range(0, num_layers, 8):
        end = min(start + 8, num_layers)
        if start < num_layers:
            layers_dict[(start, end)] = list(range(0, end)) + list(range(start, num_layers))

    # 12-layer blocks (quarter-model blocks)
    # For 48 layers: (0,12), (12,24), (24,36), (36,48)
    for start in range(0, num_layers, 12):
        end = min(start + 12, num_layers)
        if start < num_layers:
            layers_dict[(start, end)] = list(range(0, end)) + list(range(start, num_layers))

    # Fibonacci-style spacing
    # For 48 layers: (0,1), (1,2), (2,4), (4,8), (8,16), (16,24), (24,32), (32,40), (40,44), (44,48)
    fib_points = [0, 1, 2, 4, 8, 16, 24, 32, 40, 44, num_layers]
    for i in range(len(fib_points) - 1):
        start = fib_points[i]
        end = min(fib_points[i + 1], num_layers)
        if start < end:
            layers_dict[(start, end)] = list(range(0, end)) + list(range(start, num_layers))

    return layers_dict


def generate_layer_dict(num_layers: int) -> dict[tuple[int, int], list[int]]:
    """
    Generate a dictionary of ALL possible layer configurations for systematic testing.

    This mirrors the generateLayerDict function from the ExLlamaV2 codebase.
    For a 48-layer model, this generates 1177 configurations.

    Args:
        num_layers: Number of layers in the original model

    Returns:
        Dictionary mapping (repeat_start, repeat_end) to layer index lists.
        Key (0, 0) represents the baseline (unmodified) model.
        Key (i, j) means: layers 0..j-1 followed by layers i..num_layers-1
                         (effectively duplicating layers i..j-1)
    """
    layers_dict = {}
    layers_dict[(0, 0)] = list(range(num_layers))  # baseline

    for j in range(num_layers + 1):
        for i in range(j):
            # layers 0..j-1, then i..num_layers-1
            # This duplicates layers i through j-1
            layer_list = list(range(0, j)) + list(range(i, num_layers))
            layers_dict[(i, j)] = layer_list

    return layers_dict


def get_memory_usage():
    """Get current GPU memory usage in MB."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024 / 1024
    return 0


def print_model_info(model, layer_indices: Optional[list[int]] = None):
    """Print information about the MoE model configuration."""
    if hasattr(model, 'config'):
        config = model.config
        print(f"Model type: {config.model_type}")
        print(f"Original layers: {config.num_hidden_layers}")

        # MoE-specific info
        if hasattr(config, 'num_experts'):
            print(f"Experts per layer: {config.num_experts}")
        if hasattr(config, 'num_experts_per_tok'):
            print(f"Experts activated per token: {config.num_experts_per_tok}")

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
