"""
Filter multilingual RxR dataset to English-only subset.

Usage:
    python3 scripts/dataset_converters/filter_rxr_english.py

    # Custom paths:
    python3 scripts/dataset_converters/filter_rxr_english.py --rxr_root data/vln_ce/raw_data/rxr/RxR_VLNCE_v0 --splits val_unseen
"""

import argparse
import gzip
import json
import os

DEFAULT_RXR_ROOT = "data/vln_ce/raw_data/rxr/RxR_VLNCE_v0"


def peek_language_field(episodes):
    """Auto-detect the language field location in episode structure."""
    if not episodes:
        return None, None
    ep = episodes[0]
    # Case 1: instruction.language
    if "instruction" in ep and isinstance(ep["instruction"], dict):
        if "language" in ep["instruction"]:
            return "instruction.language", ep["instruction"]["language"]
    # Case 2: top-level language
    if "language" in ep:
        return "language", ep["language"]
    # Case 3: instruction_language
    if "instruction_language" in ep:
        return "instruction_language", ep["instruction_language"]
    # Print full first episode for debugging
    print(f"[DEBUG] Cannot find language field. First episode keys: {list(ep.keys())}")
    print(f"[DEBUG] First episode (truncated): {json.dumps(ep, ensure_ascii=False)[:1000]}")
    return None, None


def get_language(ep, field):
    """Extract language tag from episode using detected field path."""
    if field == "instruction.language":
        return ep.get("instruction", {}).get("language", "")
    elif field == "language":
        return ep.get("language", "")
    elif field == "instruction_language":
        return ep.get("instruction_language", "")
    return ""


def filter_split(rxr_root, split):
    src_path = os.path.join(rxr_root, split, f"{split}_guide.json.gz")
    dst_path = os.path.join(rxr_root, split, f"{split}_guide_en.json.gz")

    if not os.path.exists(src_path):
        print(f"[SKIP] {src_path} not found")
        return

    with gzip.open(src_path, "rt", encoding="utf-8") as f:
        data = json.load(f)

    episodes = data.get("episodes", [])
    total = len(episodes)

    lang_field, sample_lang = peek_language_field(episodes)
    if lang_field is None:
        print(f"[{split}] Cannot detect language field, skipping")
        return

    # Show language distribution
    lang_counts = {}
    for ep in episodes:
        lang = get_language(ep, lang_field)
        lang_counts[lang] = lang_counts.get(lang, 0) + 1
    print(f"[{split}] Language distribution: {lang_counts}")

    en_episodes = [ep for ep in episodes if get_language(ep, lang_field).startswith("en")]
    print(f"[{split}] {total} episodes -> {len(en_episodes)} English episodes")

    # Clean instruction field: only keep keys accepted by habitat InstructionData
    allowed_instruction_keys = {"instruction_text", "instruction_tokens"}
    for ep in en_episodes:
        if "instruction" in ep and isinstance(ep["instruction"], dict):
            ep["instruction"] = {
                k: v for k, v in ep["instruction"].items() if k in allowed_instruction_keys
            }
            # Ensure instruction_tokens exists (required by dataset loader)
            if "instruction_tokens" not in ep["instruction"]:
                ep["instruction"]["instruction_tokens"] = []

    if not en_episodes:
        print(f"[{split}] No English episodes found, skipping save")
        return

    data["episodes"] = en_episodes

    # Add instruction_vocab if missing (required by habitat R2RVLN-v1 dataset loader)
    if "instruction_vocab" not in data:
        print(f"[{split}] Adding dummy instruction_vocab (not used by InternVLA-N1)")
        data["instruction_vocab"] = {
            "word_list": ["<pad>", "<unk>", "<stop>"],
            "word2idx_dict": {"<pad>": 0, "<unk>": 1, "<stop>": 2},
            "itos": ["<pad>", "<unk>", "<stop>"],
            "stoi": {"<pad>": 0, "<unk>": 1, "<stop>": 2},
            "num_vocab": 3,
            "UNK_INDEX": 1,
            "PAD_INDEX": 0,
        }

    with gzip.open(dst_path, "wt", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
    print(f"[{split}] Saved to {dst_path}")


def main():
    parser = argparse.ArgumentParser(description="Filter RxR dataset to English-only")
    parser.add_argument("--rxr_root", type=str, default=DEFAULT_RXR_ROOT, help="Root dir of RxR VLNCE data")
    parser.add_argument("--splits", nargs="+", default=["val_unseen"], help="Splits to process")
    args = parser.parse_args()

    for split in args.splits:
        filter_split(args.rxr_root, split)


if __name__ == "__main__":
    main()
