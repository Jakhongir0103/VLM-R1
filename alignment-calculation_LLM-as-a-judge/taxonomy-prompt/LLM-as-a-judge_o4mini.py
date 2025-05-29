# to run this you need openapi key


#!/usr/bin/env python3
"""
alignment_check.py

Load Base/GRPO polished responses for multiple splits (validation, test, train), send each through GPT-4o to verify
that the chain-of-thought reasoning actually supports the reported final answer,
and save the boolean alignment verdict plus a brief explanation, the type of misalignment,
original reasoning, and image URL.
"""

import json
import time
from pathlib import Path
import os
from datasets import load_dataset
from openai import OpenAI
from tqdm import tqdm

# ------------------------------ Config --------------------------------------
DATASET_NAME    = "cambridgeltl/vsr_zeroshot"
SPLITS          = ["validation", "test", "train"]
OPENAI_MODEL    = "o4-mini"  # judge model
MAX_ATTEMPTS    = 3
TEMPERATURE     = 0.2
# ---------------------------------------------------------------------------


def load_data(split: str):
    vsr_split = load_dataset(DATASET_NAME, split=split)
    # for train and test, only use first 340 examples
    if split in ("train", "test"):
        vsr_split = vsr_split.select(range(350))

    dataset = [
        {
            "index":      i,
            "question":   ex["caption"],
            "true_label": "True" if ex["label"] == 1 else "False",
            "image_url":  ex.get("image_link", ""),
        }
        for i, ex in enumerate(vsr_split)
    ]

    base_path = Path(f"../responses/base/generated_responses_{split}.json")
    grpo_path = Path(f"../responses/grpo/generated_responses_{split}.json")
    base = json.loads(base_path.read_text())
    grpo = json.loads(grpo_path.read_text())

    assert len(base) == len(dataset), f"Base responses length mismatch for split={split}"
    assert len(grpo) == len(dataset), f"GRPO responses length mismatch for split={split}"

    return dataset, base, grpo


TAXONOMY_PROMPT = """\
Below is a True/False claim, a model’s step-by-step reasoning, and its final “Answer: …” line.

Your job (in order):
1. Check whether the reasoning logically supports the final answer.
2. If false (i.e. mis-aligned), classify the error as exactly one of:
   - contradiction: reasoning directly contradicts the answer
   - missing_detail: reasoning addresses the right concept but omits a critical visual fact
   - irrelevant: reasoning never addresses the claim’s core relation
   - other: none of the above fits

Ignore real-world accuracy—focus only on logical entailment from the reasoning to the answer.

Claim:
{question}

Reasoning:
{reasoning_block}

Final answer:
Answer: {final_answer}

Return exactly this JSON (no extra keys, no reordering):
{{
  "aligns": true|false,
  "type": null|contradiction|missing_detail|irrelevant|other,
  "why": "<one-sentence pointer to the logical gap or confirmation>"
}}

- If "aligns": true, set "type" to null.
- If "aligns": false, choose one of the four types above.
- "why" must cite the exact mismatch, e.g. “reasoning says X but answer is Y”.
"""


def check_alignment(client: OpenAI, question: str, reasoning_block: str, final_answer: str):
    """Ask the LLM if the reasoning supports its own final answer and get misalignment type."""
    for attempt in range(1, MAX_ATTEMPTS + 1):
        payload = TAXONOMY_PROMPT.format(
            question=question.strip(),
            reasoning_block=reasoning_block.strip(),
            final_answer=final_answer.strip(),
        )
        try:
            resp = client.chat.completions.create(
                model=OPENAI_MODEL,
                #temperature=TEMPERATURE,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": "You are an evaluator of chain-of-thought reasoning with taxonomy."},
                    {"role": "user",   "content": payload},
                ],
            )
            out = json.loads(resp.choices[0].message.content)
            # sanity checks
            assert isinstance(out.get("aligns"), bool)
            assert "why" in out and "type" in out
            return out

        except Exception:
            if attempt == MAX_ATTEMPTS:
                raise
            time.sleep(1 + attempt)


def main():
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    for split in SPLITS:
        print(f"Processing split: {split}")
        dataset, base_responses, grpo_responses = load_data(split)
        alignment_results = []

        for item in tqdm(dataset, desc=f"Checking alignment ({split})"):
            i = item["index"]
            # if i != 6:
            #     continue
            for model_name, resp_list in (("base", base_responses), ("grpo", grpo_responses)):
                record = resp_list[i]
                full_resp = record.get("predicted_response", "")
                image_url = item.get("image_url", "")

                # split reasoning vs final answer
                if "\nAnswer:" in full_resp:
                    reasoning, final_line = full_resp.rsplit("\nAnswer:", 1)
                    final_ans = final_line.strip()
                else:
                    lines = full_resp.strip().splitlines()
                    reasoning, final_ans = "\n".join(lines[:-1]), lines[-1]

                verdict = check_alignment(
                    client,
                    question=item["question"],
                    reasoning_block=reasoning,
                    final_answer=final_ans,
                )

                alignment_results.append({
                    "index":      i,
                    "model":      model_name,
                    "question":   item["question"],
                    "true_label": item["true_label"],
                    "image_url":  image_url,
                    "reasoning":  reasoning,
                    "final_ans":  final_ans,
                    "aligns":     verdict["aligns"],
                    "type":       verdict.get("type"),
                    "why":        verdict["why"],
                })

        out_path = Path(f"alignment_checks_{split.upper()}_{OPENAI_MODEL}.json")
        out_path.write_text(json.dumps(alignment_results, indent=2))
        print(f"Wrote {len(alignment_results)} alignment checks → {out_path}")


if __name__ == "__main__":
    main()
