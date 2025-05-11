import re
import json
import argparse
from pathlib import Path
from typing import List
from tqdm import tqdm
from PIL import Image

import torch
from datasets import load_from_disk
from transformers import AutoProcessor, AutoModelForImageTextToText
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from statsmodels.stats.proportion import proportion_confint


def make_conversation(example, reasoning_model=False):
    if reasoning_model:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": f"file://{example['image_path']}"},
                    {"type": "text",  
                     "text": f"{example['problem']} First output the thinking process in <think>...</think> tags, then the answer in <answer>...</answer> tags."}
                ],
            }
        ]
        solution = f"<answer> {example['solution']} </answer>"
    else:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": f"file://{example['image_path']}"},
                    {"type": "text",  "text": example['problem']}
                ],
            }
        ]
        solution = example['solution']

    return {"messages": messages, "solution": solution}


def normalize_answer(ans: str) -> str:
    return re.sub(r'[^a-zA-Z0-9]', '', ans).lower()


def extract_answer(text: str) -> str:
    m = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    return normalize_answer(m.group(1)) if m else normalize_answer(text)


def generate_responses(dataset, model, processor):
    model.eval()
    device = model.device
    outputs = []
    for item in tqdm(dataset, desc="Generating responses"):
        # Build text prompt
        text = processor.apply_chat_template(
            item['messages'], tokenize=False, add_generation_prompt=True
        )
        # Load image(s)
        imgs = []
        for msg in item['messages']:
            for c in msg['content']:
                if c.get('type') == 'image':
                    path = c['image'].replace('file://', '')
                    imgs.append(Image.open(path).convert('RGB'))
        # Tokenize and prepare inputs
        inputs = processor(
            text=[text],
            images=[imgs],
            padding=True,
            return_tensors="pt"
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        # Generation
        with torch.no_grad():
            gen_ids = model.generate(**inputs, max_new_tokens=256)
        # Trim input length
        input_ids = inputs['input_ids']
        trimmed = [out[len(inp):] for out, inp in zip(gen_ids, input_ids)]
        preds = processor.tokenizer.batch_decode(trimmed, skip_special_tokens=True)
        outputs.append({
            'true_response': item['solution'],
            'predicted_response': preds[0]
        })
    return outputs


def compute_scores(responses, reasoning_model=False):
    gts, preds = [], []
    for r in responses:
        gt = extract_answer(r['true_response']) if reasoning_model else normalize_answer(r['true_response'])
        pr = extract_answer(r['predicted_response']) if reasoning_model else normalize_answer(r['predicted_response'])
        gts.append(gt); preds.append(pr)
    acc = accuracy_score(gts, preds) * 100
    prec = precision_score(gts, preds, average='weighted', zero_division=0)
    rec = recall_score(gts, preds, average='weighted', zero_division=0)
    f1 = f1_score(gts, preds, average='weighted', zero_division=0)
    n = len(gts)
    correct = sum(1 for a, b in zip(gts, preds) if a == b)
    ci_low, ci_up = (None, None)
    if n:
        ci_low, ci_up = proportion_confint(count=correct, nobs=n, alpha=0.05, method='wilson')
    return {
        'accuracy': acc,
        'accuracy_ci_95': (ci_low, ci_up),
        'precision_weighted': prec,
        'recall_weighted': rec,
        'f1_weighted': f1
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--input_data_dir', type=str, required=True)
    parser.add_argument('--output_data_dir', type=str, required=True)
    parser.add_argument('--reasoning', action='store_true')
    args = parser.parse_args()

    # Load and preprocess dataset
    raw = load_from_disk(args.input_data_dir)['validation']
    formatted = raw.map(
        lambda x: {
            'problem': f"Is the following statement true: {x['caption']}",
            'solution': str(x['label'] == 1)
        },
        remove_columns=['caption', 'label', 'relation', 'subj', 'obj'],
        desc='Preprocessing dataset'
    )
    dataset = [make_conversation(ex, args.reasoning) for ex in formatted]

    out_dir = Path(args.output_data_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    gen_path = out_dir / 'generated_responses.json'
    if gen_path.exists():
        with open(gen_path, 'r') as f:
            responses = json.load(f)
    else:
        # Load SmolVLM
        model = AutoModelForImageTextToText.from_pretrained(
            args.model_path, trust_remote_code=True, device_map='auto'
        )
        processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
        responses = generate_responses(dataset, model, processor)
        with open(gen_path, 'w') as f:
            json.dump(responses, f, indent=2)

    scores = compute_scores(responses, args.reasoning)
    with open(out_dir / 'scores.json', 'w') as f:
        json.dump(scores, f, indent=2)
    print('Evaluation complete. Scores:', scores)

if __name__ == '__main__':
    main()