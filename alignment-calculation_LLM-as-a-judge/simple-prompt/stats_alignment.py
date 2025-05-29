
# import json
# from collections import Counter
# from pathlib import Path

# # Path to your alignment results
# ALIGN_PATH = Path("alignment_checks_gpt-4o.json")

# # Load the data
# with ALIGN_PATH.open() as f:
#     data = json.load(f)

# # Tally totals and mis‐alignments
# total       = Counter()
# misaligned  = Counter()

# for entry in data:
#     model = entry["model"]
#     total[model] += 1
#     if not entry.get("aligns", False):
#         misaligned[model] += 1
#         print(f"Misaligned → model={model}, index={entry['index']}")

# # Helper for formatting percentages
# def pct(part, whole):
#     return f"{part/whole:.2%}" if whole else "n/a"

# # Print summary table
# print()
# print("Model     | Total | Misaligned |  % misaligned  |  Accuracy")
# print("----------+-------+------------+----------------+-----------")
# for model in sorted(total):
#     t = total[model]
#     m = misaligned[model]
#     aligned = t - m
#     acc = aligned / t if t else 0.0

#     print(
#         f"{model:8} | "
#         f"{t:5d} | "
#         f"{m:10d} | "
#         f"{pct(m, t):>14} | "
#         f"{pct(aligned, t):>8}"
#     )


#!/usr/bin/env python3
"""
stats_summary.py

Combine prediction‐accuracy (vs. true_response) and
alignment‐accuracy (vs. chain‐of‐thought checks) for base & grpo.
"""

import json
import re
from pathlib import Path
from collections import Counter

# # ── Config - val ───────────────────────────────────────────────────────────────────
# BASE_PATH   = Path("../responses/base/generated_responses_polished.json")
# GRPO_PATH   = Path("../responses/grpo/generated_responses_polished.json")
# ALIGN_PATH  = Path("alignment_checks_gpt-4o.json")
# # ──────────────────────────────────────────────────────────────────────────────

# ── Config - test ───────────────────────────────────────────────────────────────────
SPLIT       = "TRAIN"
BASE_PATH   = Path(f"../responses/base/generated_responses_train.json")
GRPO_PATH   = Path(f"../responses/grpo/generated_responses_train.json")
ALIGN_PATH  = Path(f"alignment_checks_{SPLIT}_o4-mini.json")
# ─────────────────────────────────────────────────────────────────────────────────────


def extract_final_answer(pred_text: str) -> str:
    """Grab the 'Answer: True|False' line (or fallback to last non-empty line)."""
    m = re.search(r"Answer:\s*(\w+)", pred_text, flags=re.IGNORECASE)
    if m:
        return m.group(1).strip()
    lines = [l.strip() for l in pred_text.splitlines() if l.strip()]
    if not lines:
        return ""
    last = lines[-1]
    if last.lower().startswith("answer"):
        return last.split(":", 1)[1].strip()
    return last


def normalize(label: str) -> str:
    """Lowercase & strip for safe comparison."""
    return label.strip().lower()


# ── 1. Compute prediction accuracy for each model ────────────────────────────
def compute_accuracy(json_path: Path):
    data = json.loads(json_path.read_text())
    total = len(data)
    correct = 0
    for idx, entry in enumerate(data):
        true_lbl = normalize(entry["true_response"])
        pred_lbl = normalize(extract_final_answer(entry["predicted_response"]))
        if pred_lbl == true_lbl:
            correct += 1
        else:
            # you can uncomment the next line to debug mismatches
            # print(f"[ACC MISMATCH] {json_path.stem} idx={idx}: true={true_lbl}, pred={pred_lbl}")
            pass
    return correct, total

base_correct, base_total = compute_accuracy(BASE_PATH)
grpo_correct, grpo_total = compute_accuracy(GRPO_PATH)

# ── 2. Compute mis‐alignment (chain‐of‐thought vs. final answer) ─────────────
align_data = json.loads(ALIGN_PATH.read_text())
misaligned = Counter()
align_total = Counter()

for entry in align_data:
    m = entry["model"]
    align_total[m] += 1
    if not entry.get("aligns", False):
        misaligned[m] += 1

# ── 3. Print combined summary ────────────────────────────────────────────────
def pct(part, whole):
    return f"{part/whole:.2%}" if whole else "n/a"

print()
print("Model |  Accuracy (corr/total) |   Accuracy%   | Misaligned (count) | Misaligned%")
print("------+------------------------+---------------+--------------------+------------")

for model, (corr, tot) in [("base", (base_correct, base_total)),
                           ("grpo", (grpo_correct, grpo_total))]:
    acc_pct   = pct(corr, tot)
    mis_cnt   = misaligned[model]
    mis_pct   = pct(mis_cnt, align_total[model])
    print(
        f"{model:5} | "
        f"{corr:3d}/{tot:3d}{' '*6} | "
        f"{acc_pct:>11}   | "
        f"{mis_cnt:4d}{' '*12} | "
        f"{mis_pct:>8}"
    )
print()

# PROMPT 1

# TEST SPLIT─────────────────────────────────────────────────────────────────────────────────────

# --------------------------------------------------------------------------------------------------------------
# Model |  Accuracy (corr/total) |   Accuracy%   | Misaligned (count) | Misaligned% o4-mini
# ------+------------------------+---------------+--------------------+------------
# base  | 253/350                |      72.29%   |   71             |   20.29%
# grpo  | 292/350                |      83.43%   |   72             |   20.57%

# model |  Accuracy (corr/total) |   Accuracy%   | Misaligned (count) | Misaligned% o4-mini run2
# ------+------------------------+---------------+--------------------+------------
# base  | 253/350       |      72.29%   |   64             |   18.29%
# grpo  | 292/350       |      83.43%   |   77             |   22.00%


# Model |  Accuracy (corr/total) |   Accuracy%   | Misaligned (count) | Misaligned% o4-mini run 3
# ------+------------------------+---------------+--------------------+------------
# base  | 253/350       |      72.29%   |   69             |   19.71%
# grpo  | 292/350       |      83.43%   |   71             |   20.29%

# --------------------------------------------------------------------------------------------------------------


# Model |  Accuracy (corr/total) |   Accuracy%   | Misaligned (count) | Misaligned% gpt-4o 
# ------+------------------------+---------------+--------------------+------------
# base  | 253/350       |      72.29%   |  111             |   31.71%
# grpo  | 292/350       |      83.43%   |  132             |   37.71%

# VAL SPLIT ─────────────────────────────────────────────────────────────────────────────────────


# Model |  Accuracy (corr/total) |   Accuracy%   | Misaligned (count) | Misaligned% o4-mini
# ------+------------------------+---------------+--------------------+------------
# base  | 245/340               |      72.06%       |   81             |   23.82%
# grpo  | 293/340                |      86.18%      |   93             |   27.35%

# Model |  Accuracy (corr/total) |   Accuracy%   | Misaligned (count) | Misaligned% gpt-4o
# ------+------------------------+---------------+--------------------+------------
# base  | 253/350       |      72.29%            |  103               |   30.29%
# grpo  | 292/350       |      83.43%            |  137               |   40.29%

# TRAIN SPLIT ─────────────────────────────────────────────────────────────────────────────────────

# Model |  Accuracy (corr/total) |   Accuracy%   | Misaligned (count) | Misaligned% o4-mini
# ------+------------------------+---------------+--------------------+------------
# base  | 246/350               |      70.29%    |  132               |   37.71%
# grpo  | 293/350               |      83.71%    |  156               |   44.57%

# Model |  Accuracy (corr/total) |   Accuracy%   | Misaligned (count) | Misaligned% gpt-4o
# ------+------------------------+---------------+--------------------+------------
# base  | 246/350       |      70.29%   |  124             |   35.43%
# grpo  | 293/350       |      83.71%   |  166             |   47.43%


# PROMPT 2
