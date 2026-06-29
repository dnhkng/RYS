#!/usr/bin/env python
"""
Smoke test: layer duplicator end-to-end.

Loads a model in bf16, runs a baseline generation, applies the duplicator
with a small mid-stack block, generates again, and prints both outputs.

The real failure signal is *garbage tokens* in the duplicated output
(repeated CJK / replacement chars / token loops) — that indicates a
broken residual stream or per-layer-state mismatch. Coherent natural
language (possibly slightly different from baseline) means the duplicator
is functioning correctly on this architecture.

For architectures with per-layer learned residual scalars (e.g. Gemma 4),
the duplicator splits each scalar across N occurrences (s^(1/N)) to
preserve baseline cumulative gain. This test exercises that path.

Usage:
    CUDA_VISIBLE_DEVICES=0 python scripts/smoke_test_layer_duplicator.py \\
        --model-path /path/to/model

For multi-GPU sharding (large models):
    CUDA_VISIBLE_DEVICES=0,1 python scripts/smoke_test_layer_duplicator.py \\
        --model-path /path/to/model --device-map auto
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core.layer_duplicator import build_model_with_layers
from src.core.layer_config import expand_single_block, is_baseline_layers
from src.workers.model_utils import (
    get_text_num_layers,
    load_model_and_tokenizer,
    parse_device_map_arg,
    parse_max_memory_json,
)


def parse_block(raw: str, num_layers: int) -> tuple[int, int]:
    parts = [p.strip() for p in raw.split(",")]
    if len(parts) != 2:
        raise ValueError(f"--block expects 'i,j', got {raw!r}")
    i, j = int(parts[0]), int(parts[1])
    if not (0 <= i < j <= num_layers):
        raise ValueError(f"--block {raw!r} invalid for {num_layers}-layer model")
    return i, j


def default_block(num_layers: int) -> tuple[int, int]:
    mid = num_layers // 2
    return max(0, mid - 2), min(num_layers, mid + 2)


def render_prompt(tokenizer, prompt: str, use_chat_template: bool) -> str:
    """Apply the chat template if available. Instruct-tuned models emit
    degenerate repetition on raw text without their turn framing."""
    if not use_chat_template or getattr(tokenizer, "chat_template", None) is None:
        return prompt
    messages = [{"role": "user", "content": prompt}]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def generate_once(model_like, tokenizer, rendered: str, max_new_tokens: int) -> tuple[str, float]:
    inputs = tokenizer(rendered, return_tensors="pt", add_special_tokens=False)
    target_device = next(model_like.parameters()).device if hasattr(model_like, "parameters") else "cuda:0"
    inputs = {k: v.to(target_device) for k, v in inputs.items()}
    t0 = time.perf_counter()
    with torch.inference_mode():
        out = model_like.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    dt = time.perf_counter() - t0
    decoded = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return decoded, dt


def main() -> int:
    parser = argparse.ArgumentParser(description="Layer duplicator smoke test")
    parser.add_argument("--model-path", required=True, help="HF model directory")
    parser.add_argument("--device-map", default="cuda:0",
                        help="HF device_map; use 'auto' for multi-GPU sharding")
    parser.add_argument("--max-memory-json", default=None,
                        help="Optional max_memory JSON, e.g. '{\"cuda:0\":\"22GiB\",\"cuda:1\":\"22GiB\"}'")
    parser.add_argument("--attention-impl", default="eager",
                        choices=["eager", "flash_attention_2", "sdpa"])
    parser.add_argument("--trust-remote-code", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--block", default=None,
                        help="Mid-stack block 'i,j' to duplicate; default = ~4 layers around the middle")
    parser.add_argument("--prompt", default="What is the capital of France?")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--use-chat-template", action=argparse.BooleanOptionalAction, default=True,
                        help="Render --prompt through the tokenizer's chat template")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA not available; this test needs a GPU.", file=sys.stderr)
        return 1
    print(f"Visible GPUs: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"  cuda:{i}  {torch.cuda.get_device_name(i)}")

    resolved_device_map = parse_device_map_arg(args.device_map)
    resolved_max_memory = parse_max_memory_json(args.max_memory_json)
    attn_impl = args.attention_impl if args.attention_impl != "eager" else None

    print(f"\nLoading {args.model_path} (bf16, device_map={resolved_device_map})...")
    t_load = time.perf_counter()
    tokenizer, model, meta = load_model_and_tokenizer(
        model_path=args.model_path,
        trust_remote_code=args.trust_remote_code,
        local_files_only=args.local_files_only,
        torch_dtype=torch.bfloat16,
        device_map=resolved_device_map,
        attn_implementation=attn_impl,
        max_memory=resolved_max_memory,
    )
    print(f"Loaded in {time.perf_counter() - t_load:.1f}s")
    print(f"  loader={meta['loader']}  arch={meta['architectures']}  text_stack={meta['text_stack']}")
    print(f"  num_layers={meta['num_layers']}")
    print(f"  hf_device_map={meta.get('hf_device_map')}")

    num_layers = get_text_num_layers(model)
    block = parse_block(args.block, num_layers) if args.block else default_block(num_layers)
    layer_indices = expand_single_block(num_layers, block)
    assert not is_baseline_layers(layer_indices, num_layers), "block must duplicate at least one layer"
    print(f"\nDuplicating block {block} -> {len(layer_indices)} layers "
          f"(+{len(layer_indices) - num_layers} extra)")

    rendered = render_prompt(tokenizer, args.prompt, args.use_chat_template)
    chat_used = rendered != args.prompt
    print(f"Chat template applied: {chat_used}")

    print("\n--- Baseline generation ---")
    baseline_out, baseline_dt = generate_once(model, tokenizer, rendered, args.max_new_tokens)
    print(f"({baseline_dt:.2f}s) prompt={args.prompt!r}")
    print(f"  -> {baseline_out!r}")

    print("\n--- Building duplicated model ---")
    wrapped = build_model_with_layers(model, layer_indices)

    print("\n--- Duplicated generation ---")
    dup_out, dup_dt = generate_once(wrapped, tokenizer, rendered, args.max_new_tokens)
    print(f"({dup_dt:.2f}s) prompt={args.prompt!r}")
    print(f"  -> {dup_out!r}")

    print("\n--- Result ---")
    same = baseline_out.strip() == dup_out.strip()
    print(f"Outputs identical: {same}")
    print("Eyeball the duplicated output: garbage tokens (CJK/replacement chars, "
          "repeated loops) indicate a broken duplicator path; coherent natural "
          "language means the duplicator is healthy on this architecture.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
