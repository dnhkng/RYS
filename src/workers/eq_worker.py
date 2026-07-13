#!/usr/bin/env python
"""
EQ-Bench Worker for Layer Duplication Experiments

Worker that pulls configs from a shared work queue and evaluates them
on EQ-Bench (emotional intelligence benchmark).

Supports both dense and MoE (Mixture of Experts) models.

Usage:
    # Launch on GPU 0
    CUDA_VISIBLE_DEVICES=0 python -m src.workers.eq_worker --model-path /path/to/model &

    # Launch on GPU 1
    CUDA_VISIBLE_DEVICES=1 python -m src.workers.eq_worker --model-path /path/to/model &
"""

import argparse
import json
import os
import pickle
import re
import sys
import time
from pathlib import Path
from typing import Optional

import torch
from tqdm.auto import tqdm

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.core.layer_duplicator_moe import (
    build_model_with_layers_moe,
    expand_multi_block_config,
)
from src.core.layer_duplicator import build_model_with_layers
from src.core.layer_config import (
    is_baseline_layers,
    layer_spec_string,
    parse_blocks_string,
    parse_layer_list_string,
    parse_queue_entry_layers,
)
from src.workers.batch_control import adaptive_batch_execute
from src.workers.model_utils import (
    is_moe_model,
    load_model_and_tokenizer,
    parse_device_map_arg,
    parse_max_memory_json,
    strip_thinking,
)
from src.workers.shared_queue import SharedWorkQueue, format_eta


# EQ-Bench scoring constants
REVISE_COEFF = 0.5  # Weight for revised scores
EMOTION_KEYS = ['emotion1_score', 'emotion2_score', 'emotion3_score', 'emotion4_score']
PADDING_MODE_MASKED = "masked"
PADDING_MODE_INPROMPT_SPACE = "inprompt_space"


def generate_eq_messages(prompt: str, *, use_no_think_prefix: bool = True) -> list[dict]:
    """Generate chat messages for EQ-Bench question.

    Note: The EQ-Bench prompts already contain detailed format instructions
    including "First pass scores:" and "Revised scores:" format.
    We just pass the prompt directly without adding extra system prompts.
    """
    prompt_text = f"/no_think {prompt}" if use_no_think_prefix else prompt
    return [{"role": "user", "content": prompt_text}]


def select_eq_reference(sample: dict) -> dict:
    """Select labels on the independent 0-10 scale requested by the prompt."""
    return sample.get("reference_answer_fullscale", sample.get("reference_answer", {}))


def extract_scores_from_section(text: str) -> Optional[list[float]]:
    """
    Extract 4 scores from a section like "First pass scores:" or "Revised scores:".

    Handles formats like:
    - "Emotion: 7"
    - "1. Emotion: 7"
    - garbled text with numbers
    """
    # Match patterns like "EmotionName: 7" or "1. EmotionName: 7"
    score_pattern = r'(?:\d\.\s*)?[A-Za-z]+:\s*(\d+(?:\.\d+)?)'
    matches = re.findall(score_pattern, text)

    valid_scores = []
    for m in matches[:4]:  # Take first 4
        try:
            val = float(m)
            if 0 <= val <= 10:
                valid_scores.append(val)
        except ValueError:
            continue

    if len(valid_scores) >= 3:  # Need at least 3 scores
        # Pad with 5.0 (neutral) if needed
        while len(valid_scores) < 4:
            valid_scores.append(5.0)
        return valid_scores

    return None


def extract_emotion_scores(text: str) -> tuple[Optional[dict], float]:
    """
    Robustly extract emotion scores from model output.

    Handles EQ-Bench format with "First pass scores:" and "Revised scores:".
    Returns (scores_dict, confidence) where confidence is 0.0-1.0.

    Extraction strategies:
    1. Look for "Revised scores:" section (preferred - more thoughtful)
    2. Look for "First pass scores:" section
    3. Any 4 valid numbers in 0-10 range
    4. Partial extraction with neutral fallback
    """
    default_scores = {
        'emotion1_score': 5.0,
        'emotion2_score': 5.0,
        'emotion3_score': 5.0,
        'emotion4_score': 5.0,
    }

    first_pass_scores = None
    revised_scores = None

    # Try to find "Revised scores:" section
    revised_match = re.search(r'Revised scores:', text, re.IGNORECASE)
    if revised_match:
        after_revised = text[revised_match.end():]
        # Stop at [End of answer] if present
        end_match = re.search(r'\[End of answer\]', after_revised, re.IGNORECASE)
        if end_match:
            after_revised = after_revised[:end_match.start()]
        revised_scores = extract_scores_from_section(after_revised)

    # Try to find "First pass scores:" section
    first_pass_match = re.search(r'First pass scores:', text, re.IGNORECASE)
    if first_pass_match:
        after_first = text[first_pass_match.end():]
        # Stop at "Critique:" if present
        critique_match = re.search(r'Critique:', after_first, re.IGNORECASE)
        if critique_match:
            after_first = after_first[:critique_match.start()]
        first_pass_scores = extract_scores_from_section(after_first)

    # Combine scores using REVISE_COEFF weighting if both available
    if revised_scores and first_pass_scores:
        combined = []
        for fp, rv in zip(first_pass_scores, revised_scores):
            # Weighted average: (1-REVISE_COEFF)*first + REVISE_COEFF*revised
            combined.append((1 - REVISE_COEFF) * fp + REVISE_COEFF * rv)
        return {
            'emotion1_score': combined[0],
            'emotion2_score': combined[1],
            'emotion3_score': combined[2],
            'emotion4_score': combined[3],
        }, 1.0  # High confidence - found both sections

    # Use revised scores if available
    if revised_scores:
        return {
            'emotion1_score': revised_scores[0],
            'emotion2_score': revised_scores[1],
            'emotion3_score': revised_scores[2],
            'emotion4_score': revised_scores[3],
        }, 0.9  # Good confidence

    # Use first pass scores if available
    if first_pass_scores:
        return {
            'emotion1_score': first_pass_scores[0],
            'emotion2_score': first_pass_scores[1],
            'emotion3_score': first_pass_scores[2],
            'emotion4_score': first_pass_scores[3],
        }, 0.8  # Decent confidence

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
        return {
            'emotion1_score': valid_in_range[0],
            'emotion2_score': valid_in_range[1],
            'emotion3_score': valid_in_range[2],
            'emotion4_score': valid_in_range[3],
        }, 0.5  # Lower confidence - found numbers but not in expected format

    # Partial extraction
    if len(valid_in_range) >= 1:
        scores = default_scores.copy()
        for i, val in enumerate(valid_in_range[:4]):
            scores[f'emotion{i+1}_score'] = val
        return scores, len(valid_in_range) / 8.0  # Very low confidence

    # Complete failure
    return default_scores, 0.0


def calculate_eq_score(predicted: dict, reference: dict, confidence: float = 1.0) -> float:
    """
    Calculate EQ-Bench score comparing predicted vs reference emotion scores.

    Uses difference-based scoring with confidence weighting.
    Low confidence pulls score toward 0.5 (neutral/random baseline).

    Args:
        predicted: dict with emotion scores
        reference: dict with reference scores
        confidence: 0.0 to 1.0 extraction confidence
    """
    if predicted is None:
        return 0.5  # Return neutral baseline for failed extraction

    # Get scores
    pred_scores = [predicted.get(k, 5.0) for k in EMOTION_KEYS]
    ref_scores = [reference.get(k, 5.0) for k in EMOTION_KEYS]

    # Calculate absolute difference score (lower is better)
    total_diff = sum(abs(p - r) for p, r in zip(pred_scores, ref_scores))
    max_possible_diff = 10 * 4  # 4 emotions, max diff of 10 each

    # Convert to 0-1 score (higher is better)
    raw_score = 1.0 - (total_diff / max_possible_diff)
    raw_score = max(0.0, raw_score)

    # Apply confidence weighting: low confidence pulls toward 0.5
    weighted_score = confidence * raw_score + (1 - confidence) * 0.5

    return weighted_score


def pretokenize_eq_dataset(dataset: dict, tokenizer, device, *, use_no_think_prefix: bool = True) -> dict:
    """Pre-tokenize all EQ-Bench questions."""
    print("Pre-tokenizing EQ dataset...")
    tokenized = {}

    for qid, sample in tqdm(dataset.items(), desc="Tokenizing"):
        messages = generate_eq_messages(sample["prompt"], use_no_think_prefix=use_no_think_prefix)
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False  # Disable thinking mode for faster inference
        )
        inputs = tokenizer(prompt, return_tensors="pt")
        tokenized[qid] = {
            'input_ids': inputs['input_ids'].to(device),
            'attention_mask': inputs['attention_mask'].to(device),
            'reference': select_eq_reference(sample),
        }

    print(f"Pre-tokenized {len(tokenized)} questions")
    return tokenized


def run_eq_test(
    model,
    tokenized_dataset: dict,
    tokenizer,
    batch_size: int = 1,
    max_new_tokens: int = 384,
    save_responses: bool = True,
    padding_mode: str = PADDING_MODE_MASKED,
    prompt_pad_id: int | None = None,
):
    """
    Run EQ-Bench test on the model.

    EQ-Bench prompts ask for: First pass scores, Critique, Revised scores.
    384 tokens is enough to capture both score sections even with a critique.

    Returns:
        If save_responses=False: float (average score)
        If save_responses=True: dict with 'score' and 'responses'
    """
    scores = []
    responses = [] if save_responses else None
    questions = list(tokenized_dataset.items())
    default_pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    for i in range(0, len(questions), batch_size):
        batch_items = questions[i:i+batch_size]

        if batch_size == 1:
            qid, cached = batch_items[0]
            with torch.no_grad():
                outputs = model.generate(
                    input_ids=cached['input_ids'],
                    attention_mask=cached['attention_mask'],
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=default_pad_id,
                )

            input_length = cached['input_ids'].shape[1]
            generated_ids = outputs[0][input_length:]
            raw_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

            # Strip thinking blocks before parsing
            stripped_text = strip_thinking(raw_text)
            predicted, confidence = extract_emotion_scores(stripped_text)
            question_score = calculate_eq_score(predicted, cached['reference'], confidence)
            scores.append(question_score)

            if save_responses:
                responses.append({
                    'qid': qid,
                    'raw_output': raw_text,
                    'extracted': predicted,
                    'confidence': confidence,
                    'reference': cached['reference'],
                    'score': question_score
                })

        else:
            # Batched processing
            batch_input_ids = []
            batch_attention_masks = []
            batch_references = []
            batch_input_lengths = []
            batch_qids = []

            max_len = max(item[1]['input_ids'].shape[1] for item in batch_items)
            pad_id = int(
                prompt_pad_id
                if (padding_mode == PADDING_MODE_INPROMPT_SPACE and prompt_pad_id is not None)
                else default_pad_id
            )

            for qid, cached in batch_items:
                input_ids = cached['input_ids']
                attention_mask = cached['attention_mask']

                pad_length = max_len - input_ids.shape[1]
                if pad_length > 0:
                    input_ids = torch.cat([
                        torch.full(
                            (1, pad_length),
                            pad_id,
                            device=input_ids.device,
                            dtype=input_ids.dtype,
                        ),
                        input_ids
                    ], dim=1)
                    if padding_mode == PADDING_MODE_INPROMPT_SPACE:
                        left_mask = torch.ones(
                            (1, pad_length),
                            device=attention_mask.device,
                            dtype=attention_mask.dtype,
                        )
                    else:
                        left_mask = torch.zeros(
                            (1, pad_length),
                            device=attention_mask.device,
                            dtype=attention_mask.dtype,
                        )
                    attention_mask = torch.cat([
                        left_mask,
                        attention_mask
                    ], dim=1)

                batch_input_ids.append(input_ids)
                batch_attention_masks.append(attention_mask)
                batch_references.append(cached['reference'])
                batch_input_lengths.append(max_len)
                batch_qids.append(qid)

            batch_input_ids = torch.cat(batch_input_ids, dim=0)
            batch_attention_masks = torch.cat(batch_attention_masks, dim=0)

            with torch.no_grad():
                outputs = model.generate(
                    input_ids=batch_input_ids,
                    attention_mask=batch_attention_masks,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=pad_id,
                )

            for j, (output, input_length, reference, qid) in enumerate(
                zip(outputs, batch_input_lengths, batch_references, batch_qids)
            ):
                generated_ids = output[input_length:]
                raw_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

                # Strip thinking blocks before parsing
                stripped_text = strip_thinking(raw_text)
                predicted, confidence = extract_emotion_scores(stripped_text)
                question_score = calculate_eq_score(predicted, reference, confidence)
                scores.append(question_score)

                if save_responses:
                    responses.append({
                        'qid': qid,
                        'raw_output': raw_text,
                        'extracted': predicted,
                        'confidence': confidence,
                        'reference': reference,
                        'score': question_score
                    })

    avg_score = sum(scores) / len(scores) if scores else 0.0

    if save_responses:
        return {'score': avg_score, 'responses': responses}
    return avg_score


def run_eq_preflight(
    model,
    tokenized_dataset: dict,
    tokenizer,
    *,
    samples: int,
    batch_size: int,
    max_new_tokens: int,
    padding_mode: str,
    prompt_pad_id: int,
    min_nonzero_conf_rate: float,
) -> dict[str, float]:
    """Quick baseline probe to catch parser-collapse runs before long sweeps."""
    probe_items = list(tokenized_dataset.items())[:samples]
    if not probe_items:
        raise RuntimeError("Preflight requested but tokenized EQ dataset is empty.")

    probe_dataset = dict(probe_items)
    probe_batch = max(1, min(batch_size, len(probe_dataset)))
    probe_result = run_eq_test(
        model,
        probe_dataset,
        tokenizer,
        batch_size=probe_batch,
        max_new_tokens=max_new_tokens,
        save_responses=True,
        padding_mode=padding_mode,
        prompt_pad_id=prompt_pad_id,
    )

    responses = probe_result.get("responses", [])
    nonzero_conf = sum(1 for row in responses if float(row.get("confidence", 0.0)) > 0.0)
    nonzero_rate = nonzero_conf / len(responses) if responses else 0.0

    if nonzero_conf == 0:
        raise RuntimeError(
            "EQ preflight failed: zero non-zero extraction confidence across probe samples. "
            "This usually indicates an incompatible checkpoint loader/runtime."
        )
    if nonzero_rate < min_nonzero_conf_rate:
        raise RuntimeError(
            "EQ preflight failed: non-zero confidence extraction rate "
            f"{nonzero_rate:.2%} is below required threshold {min_nonzero_conf_rate:.2%}."
        )

    return {
        "samples": float(len(responses)),
        "score": float(probe_result.get("score", 0.0)),
        "nonzero_conf_rate": float(nonzero_rate),
    }


def main():
    parser = argparse.ArgumentParser(description="EQ benchmark worker for shared relayer queues")
    parser.add_argument("--queue-file", type=str, default="eq_work_queue.json",
                        help="Path to shared queue file (canonical layers entries or legacy key entries)")
    parser.add_argument("--results-file", type=str, default="results/eq_results.pkl",
                        help="Path to shared results file")
    parser.add_argument("--model-path", type=str, required=True,
                        help="Path to Hugging Face model directory")
    parser.add_argument("--dataset-path", type=str, default="datasets/eq_16.json",
                        help="Path to EQ benchmark dataset")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Batch size for inference (smaller than math due to longer outputs)")
    parser.add_argument("--max-new", type=int, default=384,
                        help="max_new_tokens for EQ generation")
    parser.add_argument("--padding-mode", type=str, choices=[PADDING_MODE_MASKED, PADDING_MODE_INPROMPT_SPACE],
                        default=PADDING_MODE_MASKED,
                        help="Padding behavior for batched generation")
    parser.add_argument("--prompt-pad-id", type=int, default=None,
                        help="Token id used for inprompt-space left padding")
    parser.add_argument("--use-no-think-prefix", action=argparse.BooleanOptionalAction, default=True,
                        help="Prefix EQ prompts with /no_think")
    parser.add_argument("--adaptive-batch-retry", action=argparse.BooleanOptionalAction, default=True,
                        help="Auto-retry with smaller batch size on OOM/context failures")
    parser.add_argument("--min-batch-size", type=int, default=1,
                        help="Smallest batch size allowed during adaptive retries")
    parser.add_argument("--max-retries-per-phase", type=int, default=8,
                        help="Maximum adaptive retries for EQ inference")
    parser.add_argument("--attention-impl", type=str, default="eager",
                        choices=["eager", "flash_attention_2", "sdpa"],
                        help="Attention implementation")
    parser.add_argument("--trust-remote-code", action=argparse.BooleanOptionalAction, default=True,
                        help="Whether to enable HuggingFace trust_remote_code when loading model/tokenizer")
    parser.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=False,
                        help="Load model/tokenizer from local files only (no Hub fetch)")
    parser.add_argument("--device-map", type=str, default="cuda:0",
                        help="HF device_map value: e.g. 'cuda:0', 'auto', or JSON object string")
    parser.add_argument("--max-memory-json", type=str, default=None,
                        help="Optional max_memory JSON, e.g. '{\"cuda:0\":\"80GiB\",\"cuda:1\":\"80GiB\"}'")
    parser.add_argument("--cpu-offload", action=argparse.BooleanOptionalAction, default=False,
                        help="Enable HF state-dict offload during loading (useful with device_map auto)")
    parser.add_argument("--offload-folder", type=str, default=None,
                        help="Optional folder for HF offload files when --cpu-offload is enabled")
    parser.add_argument("--worker-id", type=str, default=None,
                        help="Worker ID for logging")
    parser.add_argument("--skip-preflight", action="store_true",
                        help="Skip baseline generation sanity check before queue/custom run")
    parser.add_argument("--preflight-samples", type=int, default=8,
                        help="Number of questions to sample for EQ preflight")
    parser.add_argument("--preflight-max-new", type=int, default=256,
                        help="max_new_tokens for EQ preflight probe")
    parser.add_argument("--preflight-min-nonzero-conf-rate", type=float, default=0.5,
                        help="Minimum non-zero confidence extraction rate in preflight (0..1)")

    # Direct layer specification (single-config mode, bypasses queue)
    parser.add_argument("--layer-list", type=str, default=None,
                        help="Direct layer list, e.g., '0,1,2,3,2,3,4,5'. Bypasses queue.")
    parser.add_argument("--blocks", type=str, default=None,
                        help="Multi-block spec, e.g., '3,6;4,6'. Bypasses queue.")
    parser.add_argument("--config-file", type=str, default=None,
                        help="File with configs (one per line). Lines starting with 'layers:' are layer lists, "
                             "otherwise treated as block specs. Bypasses queue.")

    args = parser.parse_args()

    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1")
    if args.max_new < 1:
        raise ValueError("--max-new must be >= 1")
    if args.min_batch_size < 1:
        raise ValueError("--min-batch-size must be >= 1")
    if args.max_retries_per_phase < 0:
        raise ValueError("--max-retries-per-phase must be >= 0")
    if args.preflight_samples < 1:
        raise ValueError("--preflight-samples must be >= 1")
    if args.preflight_max_new < 1:
        raise ValueError("--preflight-max-new must be >= 1")
    if not (0.0 <= args.preflight_min_nonzero_conf_rate <= 1.0):
        raise ValueError("--preflight-min-nonzero-conf-rate must be in [0, 1]")

    try:
        resolved_device_map = parse_device_map_arg(args.device_map)
    except Exception as exc:
        raise ValueError(f"Invalid --device-map value: {exc}") from exc
    try:
        resolved_max_memory = parse_max_memory_json(args.max_memory_json)
    except Exception as exc:
        raise ValueError(f"Invalid --max-memory-json value: {exc}") from exc

    # Auto-detect worker ID
    if args.worker_id is None:
        cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', '0')
        args.worker_id = f"GPU{cuda_visible}"

    print("=" * 80)
    print(f"EQ-Bench Worker [{args.worker_id}]")
    print("=" * 80)
    print(f"Queue file: {args.queue_file}")
    print(f"Results file: {args.results_file}")
    print(f"Model: {args.model_path}")
    print(f"Dataset: {args.dataset_path}")
    print(f"Batch size: {args.batch_size}")
    print(f"EQ max_new: {args.max_new}")
    print(f"Padding mode: {args.padding_mode}")
    print(f"Adaptive retry: {args.adaptive_batch_retry} (min={args.min_batch_size}, max_retries={args.max_retries_per_phase})")
    print(
        "Preflight: "
        + (
            "disabled"
            if args.skip_preflight
            else (
                "enabled "
                f"(samples={args.preflight_samples}, max_new={args.preflight_max_new}, "
                f"min_nonzero_conf_rate={args.preflight_min_nonzero_conf_rate:.2f})"
            )
        )
    )
    print(f"Trust remote: {args.trust_remote_code}")
    print(f"Local files only: {args.local_files_only}")
    print(f"Device map: {resolved_device_map}")
    if resolved_max_memory is not None:
        print(f"Max memory: {resolved_max_memory}")
    print(f"CPU offload: {args.cpu_offload} (folder={args.offload_folder})")

    custom_mode = bool(args.layer_list or args.blocks or args.config_file)

    # Initialize work queue
    queue = SharedWorkQueue(args.queue_file, args.results_file)

    if not custom_mode:
        remaining, completed = queue.get_queue_status()
        print(f"\nQueue status: {remaining} remaining, {completed} completed")
        if remaining == 0:
            print("Queue is empty, nothing to do!")
            return

    # Load dataset
    print(f"\nLoading EQ dataset from {args.dataset_path}")
    with open(args.dataset_path, "r") as f:
        dataset = json.load(f)
    print(f"Loaded {len(dataset)} EQ questions")

    # Load model
    print(f"\nLoading model from {args.model_path}")
    attn_impl = args.attention_impl if args.attention_impl != "eager" else None
    tokenizer, model, load_meta = load_model_and_tokenizer(
        model_path=args.model_path,
        trust_remote_code=args.trust_remote_code,
        local_files_only=args.local_files_only,
        torch_dtype=torch.bfloat16,
        device_map=resolved_device_map,
        attn_implementation=attn_impl,
        max_memory=resolved_max_memory,
        cpu_offload=args.cpu_offload,
        offload_folder=args.offload_folder,
    )
    print(f"Loader: {load_meta['loader']} | architectures={load_meta['architectures']} | text_stack={load_meta['text_stack']}")
    print(f"Resolved HF device map: {load_meta.get('hf_device_map')}")

    # Detect model type (MoE vs dense)
    model_is_moe = is_moe_model(model)
    model_type = "MoE" if model_is_moe else "Dense"
    print(f"Model type: {model_type}")

    # Select appropriate layer duplication function
    if model_is_moe:
        build_duplicated_model = build_model_with_layers_moe
    else:
        build_duplicated_model = build_model_with_layers

    num_layers = int(load_meta["num_layers"])
    print(f"Model has {num_layers} text layers")

    # Pre-tokenize dataset
    tokenized_dataset = pretokenize_eq_dataset(
        dataset,
        tokenizer,
        model.device,
        use_no_think_prefix=args.use_no_think_prefix,
    )
    if args.prompt_pad_id is not None:
        prompt_pad_id = int(args.prompt_pad_id)
    else:
        try:
            space_ids = tokenizer(" ", add_special_tokens=False)["input_ids"]
            prompt_pad_id = int(space_ids[-1]) if space_ids else int(tokenizer.pad_token_id or tokenizer.eos_token_id)
        except Exception:
            prompt_pad_id = int(tokenizer.pad_token_id or tokenizer.eos_token_id)

    if not args.skip_preflight:
        preflight = run_eq_preflight(
            model,
            tokenized_dataset,
            tokenizer,
            samples=args.preflight_samples,
            batch_size=args.batch_size,
            max_new_tokens=args.preflight_max_new,
            padding_mode=args.padding_mode,
            prompt_pad_id=prompt_pad_id,
            min_nonzero_conf_rate=args.preflight_min_nonzero_conf_rate,
        )
        print(
            "Preflight passed: "
            f"samples={int(preflight['samples'])}, "
            f"score={preflight['score']:.4f}, "
            f"nonzero_conf_rate={preflight['nonzero_conf_rate']:.2%}"
        )

    def run_eq_with_retry(run_model):
        """Run EQ eval with optional adaptive batch fallback."""
        execution = adaptive_batch_execute(
            lambda batch: run_eq_test(
                run_model,
                tokenized_dataset,
                tokenizer,
                batch_size=batch,
                max_new_tokens=args.max_new,
                save_responses=True,
                padding_mode=args.padding_mode,
                prompt_pad_id=prompt_pad_id,
            ),
            initial_batch_size=args.batch_size,
            min_batch_size=args.min_batch_size,
            max_retries=args.max_retries_per_phase,
            enabled=args.adaptive_batch_retry,
            phase_name="eq",
            on_retry=lambda msg: print(f"[{args.worker_id}] {msg}"),
        )
        return execution.result, execution.batch_size, execution.retries

    # Single-config mode: direct layer list or blocks specification
    if custom_mode:
        print(f"\n{'='*80}")
        print("Custom config mode (bypassing queue)")
        print(f"{'='*80}")

        # Build list of configs to process
        configs_to_run = []

        if args.config_file:
            # Read configs from file
            print(f"Loading configs from {args.config_file}")
            with open(args.config_file, 'r') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue  # Skip empty lines and comments

                    if line.startswith('layers:'):
                        # Explicit layer list: "layers:0,1,2,3,2,3,4,5"
                        layer_str = line[7:].strip()
                        layer_indices = parse_layer_list_string(layer_str)
                        config_name = f"layers:{layer_str}"
                    else:
                        # Block spec: "3,6" or "3,6;4,6"
                        blocks = parse_blocks_string(line)
                        layer_indices = expand_multi_block_config(num_layers, blocks)
                        config_name = f"blocks:{line}"

                    configs_to_run.append((config_name, layer_indices))
            print(f"Loaded {len(configs_to_run)} configs from file")

        elif args.layer_list:
            layer_indices = parse_layer_list_string(args.layer_list)
            configs_to_run.append((f"layer_list:{args.layer_list}", layer_indices))

        else:  # args.blocks
            blocks = parse_blocks_string(args.blocks)
            layer_indices = expand_multi_block_config(num_layers, blocks)
            configs_to_run.append((f"blocks:{args.blocks}", layer_indices))

        # Validate all configs
        for config_name, layer_indices in configs_to_run:
            for idx in layer_indices:
                if idx < 0 or idx >= num_layers:
                    print(f"ERROR: Config '{config_name}' has layer index {idx} out of range [0, {num_layers})")
                    return

        # Prepare results storage
        results_path = None
        all_results = {}
        if args.results_file:
            results_path = Path(args.results_file)
            results_path.parent.mkdir(parents=True, exist_ok=True)
            if results_path.exists():
                with open(results_path, 'rb') as f:
                    all_results = pickle.load(f)

        # Process each config
        total_start = time.time()
        for i, (config_name, layer_indices) in enumerate(configs_to_run):
            print(f"\n[{i+1}/{len(configs_to_run)}] Config: {config_name}")
            print(f"Layer indices ({len(layer_indices)} layers): {layer_indices[:20]}{'...' if len(layer_indices) > 20 else ''}")
            key = tuple(layer_indices)
            if key in all_results:
                existing = all_results[key]
                existing_score = existing.get("score") if isinstance(existing, dict) else None
                if isinstance(existing_score, (int, float)):
                    print(f"Skipping (already in results): score={existing_score:.4f}")
                else:
                    print("Skipping (already in results).")
                continue

            config_start = time.time()

            if layer_indices == list(range(num_layers)):
                print("Running baseline (original model)...")
                result, effective_batch, retries = run_eq_with_retry(model)
            else:
                print("Building duplicated model...")
                dup_model = build_duplicated_model(model, layer_indices)
                result, effective_batch, retries = run_eq_with_retry(dup_model)
                del dup_model
                torch.cuda.empty_cache()

            elapsed = time.time() - config_start
            score = result['score']
            print(
                f"Result: score={score:.4f} ({elapsed:.1f}s, "
                f"batch={effective_batch}, retries={retries})"
            )

            # Save result incrementally
            all_results[key] = result
            if results_path:
                with open(results_path, 'wb') as f:
                    pickle.dump(all_results, f)

        total_elapsed = time.time() - total_start
        print(f"\n{'='*80}")
        print(f"Completed {len(configs_to_run)} configs in {total_elapsed:.1f}s")
        if results_path:
            print(f"Results saved to {results_path}")
        print(f"{'='*80}")

        return

    print("Queue mode: canonical layer-list configs (legacy (i,j) keys auto-converted).")

    # Worker loop
    print(f"\n{'='*80}")
    print("Starting EQ worker loop...")
    print(f"{'='*80}\n")

    configs_processed = 0
    start_time = time.time()

    while True:
        entry = queue.get_next_config()

        if entry is None:
            print(f"\n[{args.worker_id}] Queue empty, worker finished!")
            break

        try:
            parsed_entry = parse_queue_entry_layers(num_layers, entry)
        except Exception as exc:
            print(f"[{args.worker_id}] WARNING: Invalid queue entry {entry!r}: {exc}")
            continue
        config_idx = parsed_entry["idx"]
        layer_indices = parsed_entry["layers"]
        key = parsed_entry["layer_key"]
        config_spec = parsed_entry["spec"]

        config_start_time = time.time()

        if is_baseline_layers(layer_indices, num_layers):
            result, effective_batch, retries = run_eq_with_retry(model)
        else:
            dup_model = build_duplicated_model(model, layer_indices)
            result, effective_batch, retries = run_eq_with_retry(dup_model)
            del dup_model
            torch.cuda.empty_cache()

        config_time = time.time() - config_start_time
        configs_processed += 1

        # Extract score for logging
        score = result['score']

        # Save result (with responses)
        queue.save_result(key, result)

        remaining, completed = queue.get_queue_status()
        elapsed = time.time() - start_time
        rate = configs_processed / elapsed if elapsed > 0 else 0

        if remaining > 0 and rate > 0:
            eta_str = format_eta(remaining / rate)
        else:
            eta_str = "N/A"

        print(f"[{args.worker_id}] Config {config_idx} {config_spec} ({layer_spec_string(layer_indices)}): score={score:.4f} "
              f"({config_time:.1f}s, batch={effective_batch}, retries={retries}) | "
              f"Queue: {remaining} left | Rate: {rate:.2f}/s | ETA: {eta_str}")

    total_time = time.time() - start_time
    print(f"\n{'='*80}")
    print(f"Worker [{args.worker_id}] Summary")
    print(f"{'='*80}")
    print(f"Configs processed: {configs_processed}")
    print(f"Total time: {format_eta(total_time)}")
    if total_time > 0:
        print(f"Average rate: {configs_processed/total_time:.2f} configs/s")


if __name__ == "__main__":
    main()
