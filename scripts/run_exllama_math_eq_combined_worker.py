#!/usr/bin/env python
"""
ExLlamaV3 combined worker:
- loads the EXL3 model once
- for each queue config, runs all math + all EQ prompts in one mixed generation pass
- writes combined, math-only, and eq-only result pickles
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import sys
import time
from pathlib import Path


def add_exllamav3_to_path() -> None:
    env_path = os.environ.get("EXLLAMAV3_PATH")
    if env_path and os.path.isdir(env_path):
        sys.path.append(env_path)
        return
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "exllamav3"))
    if os.path.isdir(repo_root):
        sys.path.append(repo_root)


def add_repo_to_path() -> None:
    env_path = os.environ.get("RYS_PATH") or os.environ.get("RYS_REPRO_PATH") or os.environ.get("LEVELGEN_PATH")
    if env_path and os.path.isdir(env_path):
        sys.path.insert(0, env_path)
        return
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if os.path.isdir(os.path.join(repo_root, "src")):
        sys.path.insert(0, repo_root)


def _strip_forced_think(prompt: str) -> str:
    if prompt.endswith("<think>\n"):
        return prompt[:-len("<think>\n")]
    if prompt.endswith("<think>"):
        return prompt[:-len("<think>")]
    return prompt


def _append_think_seed(prompt: str, think_seed_mode: str, think_seed_text: str) -> str:
    if think_seed_mode == "off":
        return prompt
    if think_seed_mode == "closed_direct":
        return f"{prompt}<think>{think_seed_text}</think>\n"
    raise ValueError(f"Unknown think seed mode: {think_seed_mode}")


def apply_chat_template(
    hf_tokenizer,
    messages,
    think_seed_mode: str = "off",
    think_seed_text: str = "I can answer this directly.",
) -> str:
    try:
        prompt = hf_tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        return _append_think_seed(_strip_forced_think(prompt), think_seed_mode, think_seed_text)
    except TypeError:
        prompt = hf_tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        return _append_think_seed(_strip_forced_think(prompt), think_seed_mode, think_seed_text)


MATH_SYSTEM_PROMPT = (
    "You are a highly intelligent AI. You have extraordinary intuition and can "
    "easily make accurate estimations. For the following questions, you will "
    "always provide an answer, even if you are not certain."
)
DEFAULT_THINK_SEED_TEXT = "I can answer this now, and will do so succinctly."


def generate_math_messages(question: str, use_no_think_prefix: bool = True) -> list[dict]:
    user_text = f"/no_think {question}" if use_no_think_prefix else question
    return [
        {"role": "system", "content": MATH_SYSTEM_PROMPT},
        {"role": "user", "content": user_text},
    ]


def add_no_think_prefix(messages: list[dict]) -> list[dict]:
    if not messages:
        return messages
    updated = [dict(m) for m in messages]
    last = updated[-1]
    if last.get("role") == "user":
        content = str(last.get("content", ""))
        last["content"] = f"/no_think {content}"
    return updated


def round_up(value: int, multiple: int) -> int:
    if multiple <= 0:
        return value
    return ((value + multiple - 1) // multiple) * multiple


def parse_float_list(value: str | None) -> list[float] | None:
    if value is None:
        return None
    items = [v.strip() for v in value.split(",") if v.strip()]
    if not items:
        return None
    return [float(v) for v in items]


def estimate_max_prompt_tokens(hf_tokenizer, prompts: list[str]) -> int:
    max_tokens = 0
    for prompt in prompts:
        tokenized = hf_tokenizer(prompt, return_tensors="pt")
        max_tokens = max(max_tokens, tokenized["input_ids"].shape[1])
    return max_tokens


def build_layer_map(num_blocks: int, block: tuple[int, int]) -> list[int] | None:
    start, end = block
    if start == 0 and end == 0:
        return None
    if start < 0 or end < 0 or start >= end or end > num_blocks:
        raise ValueError(f"Invalid block {block} for {num_blocks} blocks")
    return list(range(0, end)) + list(range(start, num_blocks))


def load_exllama_model(
    model_dir: str,
    max_chunk_size: int,
    max_output_size: int,
    device: str | None = None,
    reserve_per_device: list[float] | None = None,
    use_per_device: list[float] | None = None,
):
    from exllamav3 import Config, Model, Tokenizer

    config = Config.from_directory(model_dir)
    model = Model.from_config(config)
    load_kwargs = dict(
        progressbar=True,
        max_chunk_size=max_chunk_size,
        max_output_size=max_output_size,
    )
    if device:
        load_kwargs["device"] = device
    else:
        if reserve_per_device is not None:
            load_kwargs["reserve_per_device"] = reserve_per_device
        if use_per_device is not None:
            load_kwargs["use_per_device"] = use_per_device
    model.load(**load_kwargs)
    tokenizer = Tokenizer.from_config(config)
    return config, model, tokenizer


def build_cache_and_generator(model, tokenizer, layer_map, cache_size, max_chunk_size, max_batch_size):
    from exllamav3 import Cache, Generator

    model.layer_map = layer_map
    cache = Cache(model, max_num_tokens=cache_size, layer_map=layer_map)
    for module in model.get_cache_layers():
        if getattr(module, "num_kv_heads", 1) == 0:
            continue
        for cl in module.cache_layers:
            cl.alloc(module.device)
    generator = Generator(
        model=model,
        cache=cache,
        tokenizer=tokenizer,
        max_chunk_size=max_chunk_size,
        max_batch_size=max_batch_size,
    )
    return cache, generator


def _save_pickle_result(path: Path, config_key: tuple[int, ...], value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        with path.open("wb") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            try:
                import pickle

                pickle.dump({}, f)
            finally:
                fcntl.flock(f, fcntl.LOCK_UN)

    with path.open("r+b") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            import pickle

            f.seek(0)
            data = pickle.load(f)
            data[config_key] = value
            f.seek(0)
            f.truncate()
            pickle.dump(data, f)
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def run_combined_single_pass(
    generator,
    prompts: list[str],
    items: list[dict],
    max_new_tokens: int,
) -> tuple[dict, dict]:
    from exllamav3.generator.sampler import GreedySampler
    from src.workers.math_worker import calculate_score, extract_integers, strip_thinking
    from src.workers.eq_worker import extract_emotion_scores, calculate_eq_score

    sampler = GreedySampler()
    outputs = generator.generate(
        prompt=prompts,
        max_new_tokens=max_new_tokens,
        completion_only=True,
        encode_special_tokens=True,
        sampler=sampler,
    )

    math_scores: list[float] = []
    math_responses: list[dict] = []
    eq_scores: list[float] = []
    eq_responses: list[dict] = []

    for output, item in zip(outputs, items):
        task = item["task"]
        raw_text = output
        stripped_text = strip_thinking(raw_text)

        if task == "math":
            answer = item["answer"]
            qid = item["qid"]
            stripped_integers = extract_integers(stripped_text)
            integers = stripped_integers
            has_valid_final_answer = len(stripped_integers) > 0
            fallback_used = False
            if len(integers) == 0:
                integers = extract_integers(raw_text)
                fallback_used = len(integers) > 0
            if len(integers) == 0:
                question_score = 0.0
            else:
                question_score = max(calculate_score(answer, i) for i in integers)
            math_scores.append(question_score)
            math_responses.append(
                {
                    "qid": qid,
                    "raw_output": raw_text,
                    "stripped_output": stripped_text,
                    "extracted": integers,
                    "has_valid_final_answer": has_valid_final_answer,
                    "fallback_used": fallback_used,
                    "reference": answer,
                    "score": question_score,
                }
            )
            continue

        reference = item["reference"]
        qid = item["qid"]
        predicted, confidence = extract_emotion_scores(stripped_text)
        question_score = calculate_eq_score(predicted, reference, confidence)
        eq_scores.append(question_score)
        eq_responses.append(
            {
                "qid": qid,
                "raw_output": raw_text,
                "extracted": predicted,
                "confidence": confidence,
                "reference": reference,
                "score": question_score,
            }
        )

    math_valid_count = sum(1 for r in math_responses if r["has_valid_final_answer"])
    math_fallback_count = sum(1 for r in math_responses if r["fallback_used"])

    math_result = {
        "score": (sum(math_scores) / len(math_scores)) if math_scores else 0.0,
        "valid_final_answer_count": math_valid_count,
        "valid_final_answer_rate": (math_valid_count / len(math_responses)) if math_responses else 0.0,
        "fallback_used_count": math_fallback_count,
        "fallback_used_rate": (math_fallback_count / len(math_responses)) if math_responses else 0.0,
        "responses": math_responses,
    }
    eq_result = {
        "score": (sum(eq_scores) / len(eq_scores)) if eq_scores else 0.0,
        "responses": eq_responses,
    }
    return math_result, eq_result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ExLlamaV3 combined math+EQ worker using one mixed generation pass per config."
    )
    parser.add_argument(
        "--queue-file",
        required=True,
        help="Path to shared queue JSON (canonical layers entries or legacy key entries).",
    )
    parser.add_argument("--combined-results-file", required=True, help="Path to combined results pickle.")
    parser.add_argument("--math-results-file", required=True, help="Path to math-only results pickle.")
    parser.add_argument("--eq-results-file", required=True, help="Path to eq-only results pickle.")
    parser.add_argument("--model-dir", required=True, help="Path to EXL3 model directory.")
    parser.add_argument("--math-dataset-path", default="datasets/math_16.json")
    parser.add_argument("--eq-dataset-path", default="datasets/eq_16.json")
    parser.add_argument("--math-max-new", type=int, default=64)
    parser.add_argument("--eq-max-new", type=int, default=64)
    parser.add_argument("--cache-size", type=int, default=0, help="Override cache size in tokens (0=auto).")
    parser.add_argument("--auto-cache", action="store_true", help="Auto-size cache from mixed prompt lengths.")
    parser.add_argument("--cache-page", type=int, default=256)
    parser.add_argument("--max-chunk-size", type=int, default=2048)
    parser.add_argument(
        "--max-output-size",
        type=int,
        default=0,
        help="ExLlama load-time output-size hint; 0=auto (use max(math_max_new, eq_max_new)).",
    )
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--reserve-per-device", type=str, default=None)
    parser.add_argument("--use-per-device", type=str, default=None)
    parser.add_argument("--worker-id", type=str, default=None)
    parser.add_argument("--disable-no-think-prefix", action="store_true")
    parser.add_argument("--disable-eq-no-think-prefix", action="store_true")
    parser.add_argument("--think-seed-mode", choices=["off", "closed_direct"], default="closed_direct")
    parser.add_argument("--eq-think-seed-mode", choices=["off", "closed_direct"], default="closed_direct")
    parser.add_argument("--think-seed-text", default=DEFAULT_THINK_SEED_TEXT)
    args = parser.parse_args()

    add_exllamav3_to_path()
    add_repo_to_path()

    import torch
    from transformers import AutoTokenizer
    from src.core.layer_config import layer_spec_string, parse_queue_entry_layers
    from src.workers.eq_worker import generate_eq_messages, select_eq_reference
    from src.workers.shared_queue import SharedWorkQueue

    if args.worker_id is None:
        cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "all")
        args.worker_id = f"EXL3-COMB-{cuda_visible}"

    print("=" * 80)
    print(f"ExLlamaV3 Combined Worker [{args.worker_id}]")
    print("=" * 80)
    print(f"Queue file:            {args.queue_file}")
    print(f"Combined results file: {args.combined_results_file}")
    print(f"Math results file:     {args.math_results_file}")
    print(f"EQ results file:       {args.eq_results_file}")
    print(f"Model:                 {args.model_dir}")
    print(f"Math dataset:          {args.math_dataset_path}")
    print(f"EQ dataset:            {args.eq_dataset_path}")
    print(f"Math max_new:          {args.math_max_new}")
    print(f"EQ max_new:            {args.eq_max_new}")
    print(f"Math no_think prefix:  {not args.disable_no_think_prefix}")
    print(f"EQ no_think prefix:    {not args.disable_eq_no_think_prefix}")
    print(f"Math think seed mode:  {args.think_seed_mode}")
    print(f"EQ think seed mode:    {args.eq_think_seed_mode}")

    with open(args.math_dataset_path, "r") as f:
        math_dataset = json.load(f)
    with open(args.eq_dataset_path, "r") as f:
        eq_dataset = json.load(f)

    hf_tokenizer = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)
    use_no_think_prefix = not args.disable_no_think_prefix
    use_eq_no_think_prefix = not args.disable_eq_no_think_prefix

    prompts: list[str] = []
    items: list[dict] = []

    for qid, sample in math_dataset.items():
        messages = generate_math_messages(sample["question"], use_no_think_prefix)
        prompt = apply_chat_template(
            hf_tokenizer,
            messages,
            think_seed_mode=args.think_seed_mode,
            think_seed_text=args.think_seed_text,
        )
        prompts.append(prompt)
        items.append({"task": "math", "qid": qid, "answer": sample["answer"]})

    for qid, sample in eq_dataset.items():
        messages = generate_eq_messages(sample["prompt"])
        if use_eq_no_think_prefix:
            messages = add_no_think_prefix(messages)
        prompt = apply_chat_template(
            hf_tokenizer,
            messages,
            think_seed_mode=args.eq_think_seed_mode,
            think_seed_text=args.think_seed_text,
        )
        prompts.append(prompt)
        items.append(
            {
                "task": "eq",
                "qid": qid,
                "reference": select_eq_reference(sample),
            }
        )

    if args.auto_cache or args.cache_size <= 0:
        max_prompt = estimate_max_prompt_tokens(hf_tokenizer, prompts)
        max_new = max(args.math_max_new, args.eq_max_new)
        seq_len = round_up(max_prompt + max_new, args.cache_page)
        args.cache_size = seq_len * len(prompts)
        print(
            f"[auto-cache] max_prompt={max_prompt} seq_len={seq_len} "
            f"prompts={len(prompts)} cache_size={args.cache_size}"
        )

    reserve_per_device = parse_float_list(args.reserve_per_device)
    use_per_device = parse_float_list(args.use_per_device)

    max_new_tokens = max(args.math_max_new, args.eq_max_new)
    resolved_max_output_size = args.max_output_size if args.max_output_size > 0 else max_new_tokens
    print(f"Load max_output_size:  {resolved_max_output_size}")
    print("Loading model weights once (reusing across configs)...")
    config, model, exllama_tokenizer = load_exllama_model(
        args.model_dir,
        max_chunk_size=args.max_chunk_size,
        max_output_size=resolved_max_output_size,
        device=args.device,
        reserve_per_device=reserve_per_device,
        use_per_device=use_per_device,
    )
    num_blocks = config.num_hidden_layers

    queue = SharedWorkQueue(args.queue_file, args.combined_results_file)

    while True:
        entry = queue.get_next_config()
        if entry is None:
            print("Queue empty. Exiting.")
            break

        try:
            parsed_entry = parse_queue_entry_layers(num_blocks, entry)
        except Exception as exc:
            print(f"[{args.worker_id}] Invalid queue entry {entry!r}: {exc}")
            continue
        config_key = parsed_entry["layer_key"]
        layer_indices = parsed_entry["layers"]
        config_spec = parsed_entry["spec"]
        remaining, completed = queue.get_queue_status()
        print(
            f"\n[{args.worker_id}] Running config {config_spec} ({layer_spec_string(layer_indices)}) "
            f"(remaining={remaining}, completed={completed})"
        )

        layer_map = None if layer_indices == list(range(num_blocks)) else list(layer_indices)
        max_batch_size = max(1, args.cache_size // max(1, args.cache_page))
        max_batch_size = max(max_batch_size, len(prompts))

        t0 = time.time()
        cache, generator = build_cache_and_generator(
            model,
            exllama_tokenizer,
            layer_map=layer_map,
            cache_size=args.cache_size,
            max_chunk_size=args.max_chunk_size,
            max_batch_size=max_batch_size,
        )
        math_result, eq_result = run_combined_single_pass(
            generator=generator,
            prompts=prompts,
            items=items,
            max_new_tokens=max_new_tokens,
        )
        elapsed = time.time() - t0

        combined_result = {
            "config_key": config_key,
            "config_layers": list(layer_indices),
            "config_spec": config_spec,
            "elapsed": elapsed,
            "mode": "single_pass_all",
            "num_prompts": len(prompts),
            "math_score": math_result["score"],
            "eq_score": eq_result["score"],
            "combined_score": 0.5 * (math_result["score"] + eq_result["score"]),
            "math_valid_final_answer_count": math_result["valid_final_answer_count"],
            "math_valid_final_answer_rate": math_result["valid_final_answer_rate"],
            "math_fallback_used_count": math_result["fallback_used_count"],
            "math_fallback_used_rate": math_result["fallback_used_rate"],
        }

        queue.save_result(config_key, combined_result)
        _save_pickle_result(Path(args.math_results_file), config_key, math_result)
        _save_pickle_result(Path(args.eq_results_file), config_key, eq_result)

        print(
            f"[{args.worker_id}] math={math_result['score']:.4f} "
            f"eq={eq_result['score']:.4f} combined={combined_result['combined_score']:.4f} "
            f"valid={math_result['valid_final_answer_count']}/{len(math_result['responses'])} "
            f"elapsed={elapsed:.1f}s"
        )

        for layer in cache.layers.values():
            layer.free()
        cache.detach_from_model(model)
        del generator, cache
        torch.cuda.empty_cache()

    model.unload()
    del exllama_tokenizer, model


if __name__ == "__main__":
    main()
