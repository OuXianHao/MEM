import argparse
import json
from typing import Iterable, List, Tuple

import pandas as pd
from transformers import AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser(description="Build incremental SFT parquet from MEM1 trajectories")
    parser.add_argument("--input_jsonl", action="append", required=True,
                        help="Input trajectory jsonl path. Can be passed multiple times or comma-separated.")
    parser.add_argument("--output_parquet", required=True)
    parser.add_argument("--base_model", required=True)
    parser.add_argument("--max_length", type=int, default=8196)
    return parser.parse_args()


def expand_input_paths(paths: List[str]) -> List[str]:
    out = []
    for p in paths:
        out.extend([x.strip() for x in p.split(",") if x.strip()])
    return out


def _split_by_information_spans(text: str) -> List[Tuple[str, bool]]:
    """Split text into segments (segment_text, is_information)."""
    if not text:
        return []

    open_tag = "<information>"
    close_tag = "</information>"

    parts: List[Tuple[str, bool]] = []
    i = 0
    n = len(text)
    while i < n:
        start = text.find(open_tag, i)
        if start == -1:
            parts.append((text[i:], False))
            break

        if start > i:
            parts.append((text[i:start], False))

        end = text.find(close_tag, start + len(open_tag))
        if end == -1:
            # conservative fallback for broken tags: mask to end
            parts.append((text[start:], True))
            break

        end += len(close_tag)
        parts.append((text[start:end], True))
        i = end

    return [(seg, is_info) for seg, is_info in parts if seg]


def _tokenize_segments(tokenizer, segments: Iterable[Tuple[str, bool]], normal_mask_val: int) -> Tuple[List[int], List[int], int]:
    ids: List[int] = []
    lm: List[int] = []
    info_tokens = 0

    for seg, is_info in segments:
        if not seg:
            continue
        seg_ids = tokenizer(seg, add_special_tokens=False)["input_ids"]
        ids.extend(seg_ids)
        if is_info:
            lm.extend([0] * len(seg_ids))
            info_tokens += len(seg_ids)
        else:
            lm.extend([normal_mask_val] * len(seg_ids))

    return ids, lm, info_tokens


def build_incremental_samples(item: dict) -> List[Tuple[str, str]]:
    q = (item.get("q") or "").strip()
    num_rounds = int(item.get("num_rounds", 0))
    samples = []

    history_blocks: List[str] = []
    for j in range(num_rounds):
        t_j = (item.get(f"t{j}") or "").strip()
        r_j = (item.get(f"r{j}") or "").strip()
        i_j = (item.get(f"i{j}") or "").strip()

        prompt_j = q
        if history_blocks:
            prompt_j = "\n".join([q] + history_blocks).strip()

        response_j = "\n".join([x for x in [t_j, r_j] if x]).strip()

        if response_j:
            samples.append((prompt_j, response_j))

        cur_block = "\n".join([x for x in [t_j, r_j, i_j] if x]).strip()
        if cur_block:
            history_blocks.append(cur_block)

    return samples


def encode_one(tokenizer, prompt: str, response: str, max_length: int, pad_token_id: int):
    prompt_segments = _split_by_information_spans(prompt)
    response_segments = _split_by_information_spans(response)

    prompt_ids, prompt_lm, prompt_info = _tokenize_segments(tokenizer, prompt_segments, normal_mask_val=0)
    response_ids, response_lm, response_info = _tokenize_segments(tokenizer, response_segments, normal_mask_val=1)

    input_ids = prompt_ids + response_ids
    loss_mask = prompt_lm + response_lm

    response_truncated = False
    if len(input_ids) > max_length:
        overflow = len(input_ids) - max_length
        if len(prompt_ids) >= overflow:
            prompt_ids = prompt_ids[overflow:]
            prompt_lm = prompt_lm[overflow:]
            input_ids = prompt_ids + response_ids
            loss_mask = prompt_lm + response_lm
        else:
            # prompt fully removed; still too long => truncate response from left (keep tail)
            input_ids = (prompt_ids + response_ids)[-max_length:]
            loss_mask = (prompt_lm + response_lm)[-max_length:]
            response_truncated = True

    if len(input_ids) < max_length:
        pad_len = max_length - len(input_ids)
        input_ids = input_ids + [pad_token_id] * pad_len
        loss_mask = loss_mask + [0] * pad_len

    attention_mask = [1 if x != pad_token_id else 0 for x in input_ids]
    position_ids = []
    pos = 0
    for m in attention_mask:
        if m:
            position_ids.append(pos)
            pos += 1
        else:
            position_ids.append(0)

    assert len(input_ids) == len(attention_mask) == len(position_ids) == len(loss_mask)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
        "loss_mask": loss_mask,
        "response_truncated": response_truncated,
        "response_len": len(response_ids),
        "prompt_len": len(prompt_ids),
        "info_masked_tokens": prompt_info + response_info,
    }


def main():
    args = parse_args()
    input_paths = expand_input_paths(args.input_jsonl)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    pad_token_id = tokenizer.pad_token_id

    rows = []
    total_trajectories = 0
    total_samples = 0
    total_len = 0
    response_truncated_count = 0
    total_info_masked_tokens = 0
    total_nonpad_tokens = 0

    for path in input_paths:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                total_trajectories += 1
                for prompt, response in build_incremental_samples(item):
                    encoded = encode_one(tokenizer, prompt, response, args.max_length, pad_token_id)
                    rows.append({
                        "input_ids": encoded["input_ids"],
                        "attention_mask": encoded["attention_mask"],
                        "position_ids": encoded["position_ids"],
                        "loss_mask": encoded["loss_mask"],
                    })
                    total_samples += 1
                    nonpad_len = int(sum(encoded["attention_mask"]))
                    total_len += nonpad_len
                    total_nonpad_tokens += nonpad_len
                    total_info_masked_tokens += int(encoded["info_masked_tokens"])
                    response_truncated_count += int(encoded["response_truncated"])

    if not rows:
        raise ValueError("No samples generated from input trajectories.")

    df = pd.DataFrame(rows)
    df.to_parquet(args.output_parquet, index=False)

    avg_len = total_len / total_samples
    info_mask_ratio = (total_info_masked_tokens / max(total_nonpad_tokens, 1))

    print(f"trajectories={total_trajectories}")
    print(f"samples={total_samples}")
    print(f"avg_nonpad_len={avg_len:.2f}")
    print(f"response_truncated_count={response_truncated_count}")
    print(f"response_truncated_ratio={response_truncated_count / total_samples:.4f}")
    print(f"info_masked_tokens={total_info_masked_tokens}")
    print(f"info_mask_hit_ratio={info_mask_ratio:.4f}")

    required_cols = ["input_ids", "attention_mask", "position_ids", "loss_mask"]
    for c in required_cols:
        if c not in df.columns:
            raise ValueError(f"Missing column {c}")

    bad = 0
    for _, row in df.head(min(1000, len(df))).iterrows():
        l0 = len(row["input_ids"])
        if not (l0 == len(row["attention_mask"]) == len(row["position_ids"]) == len(row["loss_mask"])):
            bad += 1
    if bad > 0:
        raise ValueError(f"Found {bad} malformed samples with inconsistent lengths.")

    print("sanity_check=passed")


if __name__ == "__main__":
    main()
