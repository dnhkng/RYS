#!/usr/bin/env python
"""ExLlamaV3 EQ-Bench worker for layer duplication sweeps."""
import argparse
import json
import os
import re
import sys
import time
from typing import Optional


def add_exllamav3_to_path():
    env_path = os.environ.get("EXLLAMAV3_PATH")
    if env_path and os.path.isdir(env_path):
        sys.path.append(env_path)
        return
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "exllamav3"))
    if os.path.isdir(repo_root):
        sys.path.append(repo_root)


def add_repo_to_path():
    env_path = os.environ.get("RYS_PATH") or os.environ.get("RYS_REPRO_PATH") or os.environ.get("LEVELGEN_PATH")
    if env_path and os.path.isdir(env_path):
        sys.path.insert(0, env_path)
        return
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if os.path.isdir(os.path.join(repo_root, "src")):
        sys.path.insert(0, repo_root)


def apply_chat_template(hf_tokenizer, messages):
    def _strip_forced_think(prompt: str) -> str:
        if prompt.endswith("<think>\n"):
            return prompt[:-len("<think>\n")]
        if prompt.endswith("<think>"):
            return prompt[:-len("<think>")]
        return prompt

    try:
        prompt = hf_tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        return _strip_forced_think(prompt)
    except TypeError:
        prompt = hf_tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        return _strip_forced_think(prompt)


# EQ-Bench scoring constants
# Using first-pass scores only (gut feeling, no deliberate reasoning)
EMOTION_KEYS = ['emotion1_score', 'emotion2_score', 'emotion3_score', 'emotion4_score']


def strip_thinking(text: str) -> str:
    """Remove <think>...</think> blocks from model output."""
    result = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    result = re.sub(r'<think>.*$', '', result, flags=re.DOTALL)
    return result.strip()


def generate_eq_messages(prompt: str) -> list[dict]:
    """Generate chat messages for EQ-Bench question."""
    return [{"role": "user", "content": prompt}]


def extract_scores_from_section(text: str) -> Optional[list[float]]:
    """Extract 4 scores from a section."""
    score_pattern = r'(?:\d\.\s*)?[A-Za-z]+:\s*(\d+(?:\.\d+)?)'
    matches = re.findall(score_pattern, text)

    valid_scores = []
    for m in matches[:4]:
        try:
            val = float(m)
            if 0 <= val <= 10:
                valid_scores.append(val)
        except ValueError:
            continue

    if len(valid_scores) >= 3:
        while len(valid_scores) < 4:
            valid_scores.append(5.0)
        return valid_scores
    return None


def extract_emotion_scores(text: str) -> tuple[Optional[dict], float]:
    """Extract first-pass emotion scores from model output (gut feeling only)."""
    default_scores = {k: 5.0 for k in EMOTION_KEYS}

    # Look for "First pass scores:" section
    first_pass_match = re.search(r'First pass scores:', text, re.IGNORECASE)
    if first_pass_match:
        after_first = text[first_pass_match.end():]
        # Stop at end of answer or any other section
        end_match = re.search(r'\[End of answer\]|Critique:|Revised scores:', after_first, re.IGNORECASE)
        if end_match:
            after_first = after_first[:end_match.start()]
        first_pass_scores = extract_scores_from_section(after_first)
        if first_pass_scores:
            return {f'emotion{i+1}_score': first_pass_scores[i] for i in range(4)}, 1.0

    # Fallback: extract all valid numbers in 0-10 range
    all_numbers = re.findall(r'\b(\d+(?:\.\d+)?)\b', text)
    valid_in_range = []
    for n in all_numbers:
        try:
            val = float(n)
            if 0 <= val <= 10:
                valid_in_range.append(val)
        except ValueError:
            continue

    if len(valid_in_range) >= 4:
        return {f'emotion{i+1}_score': valid_in_range[i] for i in range(4)}, 0.5

    if len(valid_in_range) >= 1:
        scores = default_scores.copy()
        for i, val in enumerate(valid_in_range[:4]):
            scores[f'emotion{i+1}_score'] = val
        return scores, len(valid_in_range) / 8.0

    return default_scores, 0.0


def calculate_eq_score(predicted: dict, reference: dict, confidence: float = 1.0) -> float:
    """Calculate EQ-Bench score comparing predicted vs reference emotion scores."""
    if predicted is None:
        return 0.5

    pred_scores = [predicted.get(k, 5.0) for k in EMOTION_KEYS]
    ref_scores = [reference.get(k, 5.0) for k in EMOTION_KEYS]

    total_diff = sum(abs(p - r) for p, r in zip(pred_scores, ref_scores))
    max_possible_diff = 10 * 4

    raw_score = 1.0 - (total_diff / max_possible_diff)
    raw_score = max(0.0, raw_score)

    weighted_score = confidence * raw_score + (1 - confidence) * 0.5
    return weighted_score


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


def estimate_max_prompt_tokens(hf_tokenizer, dataset) -> int:
    max_tokens = 0
    for _, sample in dataset.items():
        messages = generate_eq_messages(sample["prompt"])
        prompt = apply_chat_template(hf_tokenizer, messages)
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
    model_dir,
    max_chunk_size,
    max_output_size,
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
    model.load(
        **load_kwargs,
    )
    tokenizer = Tokenizer.from_config(config)
    return config, model, tokenizer


def build_cache_and_generator(model, tokenizer, layer_map, cache_size, max_chunk_size, max_batch_size):
    from exllamav3 import Cache, Generator

    if layer_map is not None:
        model.layer_map = layer_map
    else:
        model.layer_map = None
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


def run_eq(generator, hf_tokenizer, prompts, references, qids, batch_size, max_new_tokens):
    from exllamav3.generator.sampler import GreedySampler

    scores = []
    responses = []
    sampler = GreedySampler()

    for start in range(0, len(prompts), batch_size):
        batch_prompts = prompts[start:start + batch_size]
        batch_references = references[start:start + batch_size]
        batch_qids = qids[start:start + batch_size]
        outputs = generator.generate(
            prompt=batch_prompts,
            max_new_tokens=max_new_tokens,
            completion_only=True,
            encode_special_tokens=True,
            sampler=sampler,
        )
        for output, reference, qid in zip(outputs, batch_references, batch_qids):
            raw_text = output
            stripped_text = strip_thinking(raw_text)
            predicted, confidence = extract_emotion_scores(stripped_text)
            question_score = calculate_eq_score(predicted, reference, confidence)
            scores.append(question_score)
            responses.append({
                "qid": qid,
                "raw_output": raw_text,
                "extracted": predicted,
                "confidence": confidence,
                "reference": reference,
                "score": question_score,
            })

    avg_score = sum(scores) / len(scores) if scores else 0
    return {"score": avg_score, "responses": responses}


def main():
    parser = argparse.ArgumentParser(description="ExLlamaV3 EQ-Bench worker for layer duplication sweeps.")
    parser.add_argument(
        "--queue-file",
        type=str,
        required=True,
        help="Path to shared queue file (canonical layers entries or legacy key entries).",
    )
    parser.add_argument("--results-file", type=str, required=True, help="Path to shared results file")
    parser.add_argument("--model-dir", type=str, required=True, help="Path to EXL3 model directory")
    parser.add_argument("--dataset-path", type=str, default="datasets/eq_16.json",
                        help="Path to EQ-Bench dataset")
    parser.add_argument("--batch-size", type=int, default=5, help="Batch size for inference")
    parser.add_argument("--max-new", type=int, default=384, help="Max new tokens")
    parser.add_argument("--cache-size", type=int, default=16384, help="Cache size in tokens")
    parser.add_argument("--auto-cache", action="store_true", help="Auto-size cache based on prompt length")
    parser.add_argument("--cache-page", type=int, default=256, help="Round seq len to this multiple")
    parser.add_argument("--max-chunk-size", type=int, default=2048, help="Max chunk size for load")
    parser.add_argument("--max-output-size", type=int, default=512, help="Max output size hint for load")
    parser.add_argument("--device", type=str, default=None, help="Single device to load model on (e.g. cuda:0).")
    parser.add_argument(
        "--reserve-per-device",
        type=str,
        default=None,
        help="Comma-separated GB to reserve per device for autosplit (e.g. 8,8).",
    )
    parser.add_argument(
        "--use-per-device",
        type=str,
        default=None,
        help="Comma-separated GB target use per device for autosplit (e.g. 70,70).",
    )
    parser.add_argument("--worker-id", type=str, default=None, help="Worker ID for logging")
    args = parser.parse_args()

    add_exllamav3_to_path()
    add_repo_to_path()

    import torch
    from transformers import AutoTokenizer
    from src.core.layer_config import layer_spec_string, parse_queue_entry_layers
    from src.workers.eq_worker import select_eq_reference
    from src.workers.shared_queue import SharedWorkQueue

    if args.worker_id is None:
        cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "all")
        args.worker_id = f"EXL3-{cuda_visible}"

    print("=" * 80)
    print(f"ExLlamaV3 EQ-Bench Worker [{args.worker_id}]")
    print("=" * 80)
    print(f"Queue file:   {args.queue_file}")
    print(f"Results file: {args.results_file}")
    print(f"Model:        {args.model_dir}")
    print(f"Dataset:      {args.dataset_path}")
    print(f"Batch size:   {args.batch_size}")

    with open(args.dataset_path, "r") as f:
        dataset = json.load(f)
    hf_tokenizer = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)
    reserve_per_device = parse_float_list(args.reserve_per_device)
    use_per_device = parse_float_list(args.use_per_device)

    if args.auto_cache:
        max_prompt = estimate_max_prompt_tokens(hf_tokenizer, dataset)
        seq_len = round_up(max_prompt + args.max_new, args.cache_page)
        args.cache_size = seq_len * args.batch_size
        print(f"[auto-cache] seq_len={seq_len} cache_size={args.cache_size}")

    prompts = []
    references = []
    qids = []
    for qid, sample in dataset.items():
        messages = generate_eq_messages(sample["prompt"])
        prompts.append(apply_chat_template(hf_tokenizer, messages))
        references.append(select_eq_reference(sample))
        qids.append(qid)

    print("Loading model weights once (reusing across configs)...")
    config, model, exllama_tokenizer = load_exllama_model(
        args.model_dir,
        max_chunk_size=args.max_chunk_size,
        max_output_size=args.max_output_size,
        device=args.device,
        reserve_per_device=reserve_per_device,
        use_per_device=use_per_device,
    )
    num_blocks = config.num_hidden_layers

    queue = SharedWorkQueue(args.queue_file, args.results_file)

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
        print(f"\n[{args.worker_id}] Running config {config_spec} ({layer_spec_string(layer_indices)}) "
              f"(remaining={remaining}, completed={completed})")

        layer_map = None if layer_indices == list(range(num_blocks)) else list(layer_indices)
        max_batch_size = max(1, args.cache_size // args.cache_page)

        t0 = time.time()
        cache, generator = build_cache_and_generator(
            model,
            exllama_tokenizer,
            layer_map=layer_map,
            cache_size=args.cache_size,
            max_chunk_size=args.max_chunk_size,
            max_batch_size=max_batch_size,
        )

        result = run_eq(generator, hf_tokenizer, prompts, references, qids, args.batch_size, args.max_new)
        elapsed = time.time() - t0
        result["elapsed"] = elapsed
        result["config_key"] = config_key
        result["config_layers"] = list(layer_indices)
        result["config_spec"] = config_spec

        queue.save_result(config_key, result)
        print(f"[{args.worker_id}] Score={result['score']:.4f} elapsed={elapsed:.1f}s")

        for layer in cache.layers.values():
            layer.free()
        cache.detach_from_model(model)
        del generator, cache
        torch.cuda.empty_cache()

    model.unload()
    del exllama_tokenizer, model


if __name__ == "__main__":
    main()
