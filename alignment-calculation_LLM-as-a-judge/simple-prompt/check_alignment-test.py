
#!/usr/bin/env python3
"""
alignment_check.py

Load Base/GRPO polished responses, send each through GPT-4o to verify
that the chain-of-thought reasoning actually supports the reported final answer,
and save the boolean alignment verdict plus a brief explanation.
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
BASE_PATH       = Path("../responses/base/generated_responses_validation.json")
GRPO_PATH       = Path("../responses/grpo/generated_responses_validation.json")
OPENAI_MODEL    = "o4-mini"  # judge model
OUT_PATH        = Path(f"alignment_checks_VAL_{OPENAI_MODEL}_1.json")

MAX_ATTEMPTS    = 3
TEMPERATURE     = 0.2
# ---------------------------------------------------------------------------


def load_data():
    vsr_validation = load_dataset(DATASET_NAME, split="validation")
    #vsr_validation = vsr_validation.select(range(350))
    dataset = [
        {
            "index":      i,
            "question":   ex["caption"],
            "true_label": "True" if ex["label"] == 1 else "False",
        }
        for i, ex in enumerate(vsr_validation)
    ]

    base = json.loads(BASE_PATH.read_text())
    grpo = json.loads(GRPO_PATH.read_text())

    assert len(base) == len(dataset), "Base responses length mismatch"
    assert len(grpo) == len(dataset), "GRPO responses length mismatch"

    return dataset, base, grpo


# ALIGN_PROMPT = """\
# Below is a model’s step‐by‐step reasoning followed by its final “Answer: …” line.
# Your job is to check whether the reasoning logically supports the final answer.

# Reasoning:
# {reasoning_block}

# Final answer:
# Answer: {final_answer}

# Return JSON exactly in this format:
# {{"aligns": true|false, "why": "<brief explanation>"}}\
# """

ALIGN_PROMPT = """\
Below is one multiple‐choice question, a model’s step-by-step reasoning, and its final “Answer: …” line.  
**Your job:** Ignore whether the reasoning is factually correct. Only decide if the chain-of-thought **logically entails** the final answer.  

Question:
{question}

Reasoning:
{reasoning_block}

Final answer:
Answer: {final_answer}

Return JSON **exactly** in this format (no extra keys):
{"aligns": true|false, "why": "<brief, 1-sentence explanation>"}  

- “aligns: true” means every step of the reasoning supports the answer.  
- “aligns: false” means there’s at least one logical gap or contradiction.  
- Your “why” should point to the specific gap (e.g. “reasoning says X but answer is Y”).\
"""

def check_alignment(client: OpenAI, reasoning_block: str, final_answer: str):
    """Ask the LLM if the reasoning supports its own final answer."""
    for attempt in range(1, MAX_ATTEMPTS + 1):
        payload = ALIGN_PROMPT.format(
            reasoning_block=reasoning_block.strip(),
            final_answer=final_answer.strip(),
        )
        try:
            resp = client.chat.completions.create(
                model=OPENAI_MODEL,
                #temperature=TEMPERATURE,
                response_format={"type": "json_object"},
                messages=[
                    {
                        "role": "system",
                        "content": "You are an evaluator of chain-of-thought reasoning."
                    },
                    {"role": "user", "content": payload},
                ],
            )
            out = json.loads(resp.choices[0].message.content)
            # sanity check
            assert isinstance(out.get("aligns"), bool)
            assert "why" in out
            return out

        except Exception as e:
            if attempt == MAX_ATTEMPTS:
                raise
            time.sleep(1 + attempt)


def main():
    dataset, base_responses, grpo_responses = load_data()
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    alignment_results = []

    for item in tqdm(dataset, desc="Checking alignment"):
        i = item["index"]
        # if i == 3:
        #     break #testing pourposes

        for model_name, resp_list in (("base", base_responses),
                                      ("grpo", grpo_responses)):
            full_resp = resp_list[i]["predicted_response"]

            # split reasoning vs final answer
            if "\nAnswer:" in full_resp:
                reasoning, final_line = full_resp.rsplit("\nAnswer:", 1)
                final_ans = final_line.strip()
            else:
                lines = full_resp.strip().splitlines()
                reasoning, final_ans = "\n".join(lines[:-1]), lines[-1]

            verdict = check_alignment(client, reasoning, final_ans)
            alignment_results.append({
                "index":      i,
                "model":      model_name,
                "question":   item["question"],
                "true_label": item["true_label"],
                "final_ans":  final_ans,
                "aligns":     verdict["aligns"],
                "why":        verdict["why"],
            })

    OUT_PATH.write_text(json.dumps(alignment_results, indent=2))
    print(f"Wrote {len(alignment_results)} alignment checks → {OUT_PATH}")


if __name__ == "__main__":
    main()


