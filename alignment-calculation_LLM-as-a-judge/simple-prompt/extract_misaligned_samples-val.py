# some processing script, no need to run

#!/usr/bin/env python3
"""
extract_misaligned_detailed.py

For each misaligned entry in alignment_checks_gpt-4o.json, grab:
  - index, model, question, true_label, final_ans, aligns, why
  - image_url from the VSR dataset
  - true_response & predicted_response from the corresponding polished JSON

Outputs: misaligned_detailed.json
"""

import json
from pathlib import Path
from datasets import load_dataset

# # ───────────────── Configuration ─────────────────────────────────────────────
# ALIGN_PATH   = Path("alignment_checks_gpt-4o.json")
# BASE_PATH    = Path("../responses/base/generated_responses_polished.json")
# GRPO_PATH    = Path("../responses/grpo/generated_responses_polished.json")
# OUT_PATH     = Path("misaligned_samples_detailed.json")
# DATASET_NAME = "cambridgeltl/vsr_zeroshot"
# # ──────────────────────────────────────────────────────────────────────────────

# ───────────────── Configuration ─────────────────────────────────────────────
ALIGN_PATH   = Path("alignment_checks_val_o4-mini.json")
BASE_PATH    = Path("../responses/base/generated_responses_val.json")
GRPO_PATH    = Path("../responses/grpo/generated_responses_val.json")
OUT_PATH     = Path("detailed_val_split.json")
DATASET_NAME = "cambridgeltl/vsr_zeroshot"
# ──────────────────────────────────────────────────────────────────────────────

def main():
    # 1) Load alignment results
    alignment = json.loads(ALIGN_PATH.read_text())

    # 2) Load polished responses
    base_polished = json.loads(BASE_PATH.read_text())
    grpo_polished = json.loads(GRPO_PATH.read_text())

    # 3) Load dataset to fetch image URLs
    vsr = load_dataset(DATASET_NAME, split="validation")

    detailed = []
    for entry in alignment:
        if entry.get("aligns", True):
            continue

        idx    = entry["index"]
        model  = entry["model"]

        # pick the right polished list
        if model == "base":
            resp_obj = base_polished[idx]
        elif model == "grpo":
            resp_obj = grpo_polished[idx]
        else:
            raise ValueError(f"Unknown model: {model}")

        # fetch image_url
        img_url = vsr[idx]["image_link"]

        # build a merged record
        merged = {
            **entry,
            "image_url":         img_url,
            "true_response":     resp_obj.get("true_response", "").strip(),
            "predicted_response":resp_obj.get("predicted_response", "").strip(),
        }
        detailed.append(merged)

    # 4) Save
    OUT_PATH.write_text(json.dumps(detailed, indent=2))
    print(f"Wrote {len(detailed)} detailed misaligned samples → {OUT_PATH}")

if __name__ == "__main__":
    main()
