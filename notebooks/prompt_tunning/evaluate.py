import re
import json
import argparse
import torch

from typing import List
from peft import PeftConfig, PeftModel
from tqdm import tqdm
from pathlib import Path

from transformers import AutoTokenizer, Qwen2_5_VLForConditionalGeneration, AutoProcessor
from datasets import load_from_disk
from qwen_vl_utils import process_vision_info

import math
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score
)
from statsmodels.stats.proportion import proportion_confint

def make_conversation(example):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": f"file://{example['image_path']}"},
                {"type": "text", "text": f"{example['problem']}"},
            ],
        }
    ]
    # {"type": "text", "text": f"{example['problem']} First output the thinking process in <think> </think> tags and then output the final answer in <answer> </answer> tags."},
    # return {"messages": messages, "solution": f"<answer> {example['solution']} </answer>"}
    return {"messages": messages, "solution": f"{example['solution']}"}

def generate_responses(dataset, model, processor):
    responses = []

    for data in tqdm(dataset):
        try:
            text = processor.apply_chat_template(
                data['messages'], tokenize=False, add_generation_prompt=True
            )
            image_inputs, _ = process_vision_info(data['messages'])
            inputs = processor(
                text=[text],
                images=image_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to(model.device)

            # Inference: Generation of the output
            generated_ids = model.generate(**inputs, max_new_tokens=512)
            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False,
                do_sample=False
            )
            responses.append({"true_response": data['solution'], "predicted_response": output_text[0]})
        except Exception as e:
            print(e)
            print("messages")
            print(data['messages'])
            print("gnd_response")
            print(data['solution'])
            raise Exception

    return responses

def normalize_answer(answer):
    """Normalizes an answer by stripping whitespace, converting to lowercase, and removing punctuation."""
    return re.sub(r'[^a-zA-Z0-9]', '', answer).lower()

def extract_answer(text: str) -> List[str]:
    # Extract the final answer within <answer> </answer> tags
    match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    if match:
        return normalize_answer(match.group(1))
    else:
        return ''

def compute_accuracy(preds: List[str], true_answers: List[str]) -> float:
    exact_matches = sum(pred == true_answers[idx] for idx, pred in enumerate(preds))
    return 100 * exact_matches / len(preds) if preds else 0

def compute_scores(generated_responses):
    all_ground_truth = []
    all_predictions = []
    
    # Normalize & collect
    for data in generated_responses:
        gt   = normalize_answer(data['true_response'])
        pred = normalize_answer(data['predicted_response'])
        all_ground_truth.append(gt)
        all_predictions.append(pred)
    
    # Basic metrics
    accuracy   = accuracy_score(all_ground_truth, all_predictions)
    precision  = precision_score(all_ground_truth, all_predictions, average='weighted')
    recall     = recall_score(all_ground_truth, all_predictions, average='weighted')
    f1_weight  = f1_score(all_ground_truth, all_predictions, average='weighted')
    
    # Count correct for CI
    n = len(all_ground_truth)
    correct_count = sum(1 for gt, pred in zip(all_ground_truth, all_predictions) if gt == pred)
    
    # 95% Wilson CI for accuracy
    if n > 0:
        ci_lower, ci_upper = proportion_confint(
            count=correct_count,
            nobs=n,
            alpha=0.05,
            method='wilson'
        )
    else:
        ci_lower = ci_upper = None

    return {
        "accuracy": accuracy,
        "accuracy_ci_95": (ci_lower, ci_upper),
        "precision_weighted": precision,
        "recall_weighted": recall,
        "f1_weighted": f1_weight
    }

def main(args):
    # preprocess the dataset
    dataset = load_from_disk(args.input_data_dir)['validation']

    dataset = dataset.map(lambda sample: {
        "problem": f'Is the following statement true: {sample["caption"]}',
        "solution": str(sample["label"]==1)
    }, remove_columns=["caption", "label", "relation", "subj", "obj"], desc="Preprocessing dataset")

    # apply formatting
    dataset = [make_conversation(sample) for sample in dataset]

    # Generate and evaluate
    output_path = Path(args.output_data_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if False and (output_path / 'generated_responses.json').exists():
        with open(output_path / 'generated_responses.json', 'r') as f:
            generated_responses = json.load(f)
    else:

        peft_config = PeftConfig.from_pretrained(args.softprompt_model)

        base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(peft_config.base_model_name_or_path, torch_dtype="auto", device_map="auto")
        processor = AutoProcessor.from_pretrained(peft_config.base_model_name_or_path, trust_remote_code=True, use_fast=True)

        # 3. Load PEFT wrapper (applies soft prompts)
        model = PeftModel.from_pretrained(base_model, args.softprompt_model, torch_dtype="auto", device_map="auto")
        model.word_embeddings = model.base_model.get_input_embeddings()

        # Get all named parameters of the model
        for name, param in model.named_parameters():
            if "prompt_embeddings" in name:  # usually something like 'prompt_embeddings.weight'
                print(f"{name}: shape {param.shape}")
                print(f"Min value: {param.data.min().item()}")
                print(f"Max value: {param.data.max().item()}")
                break
        else:
            print("Soft prompt embeddings not found.")


        with torch.inference_mode():
            generated_responses = generate_responses(dataset, model, processor)
    
        with open(output_path / 'generated_responses.json', 'w') as f:
            json.dump(generated_responses, f, indent=4) 

    scores = compute_scores(generated_responses)
    
    with open(output_path / 'scores.json', 'w') as f:
        json.dump(scores, f, indent=4)

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--input_data_dir", type=str, default="/scratch/izar/saydalie/vlm-r1/data/vsr")
    parser.add_argument("--output_data_dir", type=str, default="/home/saydalie/project/VLM-R1/results")
    parser.add_argument("--softprompt_model", type=str, default="/fill/out")

    args = parser.parse_args()

    main(args)