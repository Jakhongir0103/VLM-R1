# for the report we flipped the misalignment with alignment

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

# ── Config - test ───────────────────────────────────────────────────────────────────
SPLIT       = "VALIDATION"
BASE_PATH   = Path(f"../responses/base/generated_responses_{SPLIT.lower()}.json")
GRPO_PATH   = Path(f"../responses/grpo/generated_responses_{SPLIT.lower()}.json")
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

# ── 1. Compute prediction accuracy for each model ─────────────────────────────
def compute_accuracy(json_path: Path):
    data = json.loads(json_path.read_text())
    total = len(data)
    correct = 0
    for idx, entry in enumerate(data):
        true_lbl = normalize(entry["true_response"])
        pred_lbl = normalize(extract_final_answer(entry["predicted_response"]))
        if pred_lbl == true_lbl:
            correct += 1
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

# ── 2.5. Extract and count misalignment types by model ───────────────────────
# Build a nested counter: model -> type -> count
misalignment_types_by_model = {}
for entry in align_data:
    if not entry.get("aligns", False):
        m = entry["model"]
        t = entry.get("type") or "null"
        misalignment_types_by_model.setdefault(m, Counter())[t] += 1

# Print breakdown of misalignment types per model
print("Model | Misalignment Type   | Count")
print("------+---------------------+-------")
for m, counter in misalignment_types_by_model.items():
    for t, cnt in sorted(counter.items()):
        print(f"{m:5} | {t:19} | {cnt:5d}")
print()

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

# TEST

# Model | Misalignment Type   | Count
# ------+---------------------+-------
# grpo  | contradiction       |    48
# grpo  | irrelevant          |     4
# grpo  | missing_detail      |     8
# base  | contradiction       |    26
# base  | irrelevant          |     3
# base  | missing_detail      |    10
# base  | other               |     2


# Model |  Accuracy (corr/total) |   Accuracy%   | Misaligned (count) | Misaligned%
# ------+------------------------+---------------+--------------------+------------
# base  | 253/350       |      72.29%   |   41             |   11.71%
# grpo  | 292/350       |      83.43%   |   60             |   17.14%

# TRAIN


# Model | Misalignment Type   | Count
# ------+---------------------+-------
# grpo  | contradiction       |    67
# grpo  | irrelevant          |     6
# grpo  | missing_detail      |    18
# base  | contradiction       |    24
# base  | irrelevant          |     4
# base  | missing_detail      |    18
# base  | other               |     4


# Model |  Accuracy (corr/total) |   Accuracy%   | Misaligned (count) | Misaligned%
# ------+------------------------+---------------+--------------------+------------
# base  | 246/350       |      70.29%   |   50             |   14.29%
# grpo  | 293/350       |      83.71%   |   91             |   26.00%


# VAL

# Model | Misalignment Type   | Count
# ------+---------------------+-------
# base  | contradiction       |    24
# base  | irrelevant          |     7
# base  | missing_detail      |    11
# base  | other               |     3
# grpo  | contradiction       |    60
# grpo  | irrelevant          |     6
# grpo  | missing_detail      |     8
# grpo  | other               |     2


# Model |  Accuracy (corr/total) |   Accuracy%   | Misaligned (count) | Misaligned%
# ------+------------------------+---------------+--------------------+------------
# base  | 245/340       |      72.06%   |   45             |   13.24%
# grpo  | 293/340       |      86.18%   |   76             |   22.35%