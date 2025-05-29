import re
import json
import argparse
import torch
import os

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

# This should match your training script!
SOFTPROMPT_LEN = 5

def make_conversation(example):
    if 'image_path' in example and example['image_path'] is not None and os.path.exists(example['image_path']):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": f"file://{example['image_path']}"},
                    {"type": "text", "text": f"{example['problem']}"},
                ],
            }
        ]
    else:
        messages = [
            {
                "role": "user", 
                "content": [
                    {"type": "text", "text": f"{example['problem']}"},
                ],
            }
        ]
    
    return {"messages": messages, "solution": f"{example['solution']}"}

def generate_responses(dataset, model, processor):
    responses = []
    
    print(f"Processing {len(dataset)} examples...")
    print(f"Soft prompt length: {SOFTPROMPT_LEN}")

    for i, data in enumerate(tqdm(dataset[:100] if len(dataset) > 100 else dataset)):  # Limit for debugging
        try:
            # Apply chat template
            text = processor.apply_chat_template(
                data['messages'], tokenize=False, add_generation_prompt=True
            )
            
            # Process vision info
            image_inputs, video_inputs = process_vision_info(data['messages'])
            
            # Prepare inputs - EXACTLY like in training
            inputs = processor(
                text=[text],
                images=image_inputs if image_inputs else None,
                padding=True,
                return_tensors="pt",
            )
            
            # Move to device
            inputs = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v 
                     for k, v in inputs.items()}

            if i < 3:  # Debug first few examples
                print(f"\nExample {i}:")
                print(f"Input text: {text}")
                print(f"Input IDs shape: {inputs['input_ids'].shape}")
                print(f"First few input IDs: {inputs['input_ids'][0, :10].tolist()}")

            with torch.no_grad():
                # FIXED: Use generation approach instead of trying to extract specific logits
                # This is more reliable for soft prompt models
                
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=10,  # Increased to capture full True/False
                    do_sample=False,
                    temperature=1.0,
                    pad_token_id=processor.tokenizer.pad_token_id,
                    eos_token_id=processor.tokenizer.eos_token_id
                )
                
                # Extract only the generated tokens (after the input sequence)
                input_length = inputs['input_ids'].shape[1]
                generated_tokens = generated_ids[0][input_length:]
                generated_text = processor.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

                if i < 5:  # Debug info
                    print(f"Generated text: '{generated_text}'")
                    print(f"Input length: {input_length}, Generated tokens: {len(generated_tokens)}")

                # Parse the generated response for True/False
                generated_lower = generated_text.lower().strip()
                
                # Look for explicit True/False at the beginning
                if generated_lower.startswith('true'):
                    predicted_response = "True"
                    confidence = 1.0
                elif generated_lower.startswith('false'):
                    predicted_response = "False"
                    confidence = 1.0
                # Also check if True/False appears anywhere in the response
                elif 'true' in generated_lower and 'false' not in generated_lower:
                    predicted_response = "True"
                    confidence = 0.8
                elif 'false' in generated_lower and 'true' not in generated_lower:
                    predicted_response = "False"
                    confidence = 0.8
                # Fallback: look at the first word/token
                else:
                    first_word = generated_text.split()[0] if generated_text.split() else ""
                    if first_word.lower() in ['yes', '1', 'correct', 'right']:
                        predicted_response = "True"
                        confidence = 0.5
                    elif first_word.lower() in ['no', '0', 'incorrect', 'wrong']:
                        predicted_response = "False"
                        confidence = 0.5
                    else:
                        # Default to True if unclear
                        predicted_response = "True"
                        confidence = 0.1
                        if i < 10:
                            print(f"Warning: Unclear response '{generated_text}', defaulting to True")

            responses.append({
                "true_response": data['solution'], 
                "predicted_response": predicted_response,
                "example_id": i,
                "confidence": confidence,
                "generated_text": generated_text,
                "method": "generation"
            })
            
        except Exception as e:
            print(f"Error processing example {i}: {e}")
            import traceback
            traceback.print_exc()
            responses.append({
                "true_response": data['solution'], 
                "predicted_response": "True",
                "example_id": i,
                "error": str(e)
            })

    return responses

def normalize_answer(answer):
    if answer is None:
        return ""
    return str(answer).strip().lower()

def compute_scores(generated_responses):
    all_ground_truth = []
    all_predictions = []
    errors = 0
    
    for data in generated_responses:
        if "error" in data:
            errors += 1
            continue
            
        gt = normalize_answer(data['true_response'])
        pred = normalize_answer(data['predicted_response'])
        all_ground_truth.append(gt)
        all_predictions.append(pred)
    
    if len(all_ground_truth) == 0:
        return {"error": "No valid predictions"}
    
    accuracy = accuracy_score(all_ground_truth, all_predictions)
    precision = precision_score(all_ground_truth, all_predictions, average='weighted', zero_division=0)
    recall = recall_score(all_ground_truth, all_predictions, average='weighted', zero_division=0)
    f1_weight = f1_score(all_ground_truth, all_predictions, average='weighted', zero_division=0)
    
    n = len(all_ground_truth)
    correct_count = sum(1 for gt, pred in zip(all_ground_truth, all_predictions) if gt == pred)
    
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
        "accuracy": float(accuracy),
        "accuracy_ci_95": [float(ci_lower), float(ci_upper)] if ci_lower is not None else None,
        "precision_weighted": float(precision),
        "recall_weighted": float(recall),
        "f1_weighted": float(f1_weight),
        "total_examples": len(generated_responses),
        "valid_examples": n,
        "errors": errors
    }

def main(args):
    print(f"Loading dataset from: {args.input_data_dir}")
    
    try:
        dataset = load_from_disk(args.input_data_dir)['validation']
    except KeyError:
        try:
            dataset = load_from_disk(args.input_data_dir)['test']
        except KeyError:
            dataset = load_from_disk(args.input_data_dir)['train']
            print("Warning: Using train split")

    print(f"Original dataset size: {len(dataset)}")
    
    dataset = dataset.map(lambda sample: {
        "problem": f'Is the following statement true: {sample["caption"]}',
        "solution": str(sample["label"]==1)
    }, remove_columns=["caption", "label", "relation", "subj", "obj"], desc="Preprocessing dataset")

    dataset = [make_conversation(sample) for sample in dataset]
    print(f"Processed dataset size: {len(dataset)}")

    output_path = Path(args.output_data_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Loading model from: {args.softprompt_model}")
    
    peft_config = PeftConfig.from_pretrained(args.softprompt_model)
    print(f"Base model: {peft_config.base_model_name_or_path}")
    print(f"PEFT config: {peft_config}")

    processor = AutoProcessor.from_pretrained(
        args.softprompt_model, 
        trust_remote_code=True, 
        use_fast=True
    )
    
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    print("Loading base model...")
    base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        peft_config.base_model_name_or_path, 
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )

    print("Loading soft prompt adapter...")
    model = PeftModel.from_pretrained(
        base_model, 
        args.softprompt_model,
        torch_dtype=torch.bfloat16
    )
    
    # CRITICAL: This line from your training script
    model.word_embeddings = model.base_model.get_input_embeddings()
    model.eval()

    # FIXED: Better verification of soft prompt parameters
    soft_prompt_found = False
    print("\nChecking for soft prompt parameters...")
    for name, param in model.named_parameters():
        if any(keyword in name.lower() for keyword in ['prompt', 'virtual', 'embedding']):
            print(f"✓ Found parameter: {name}, shape: {param.shape}")
            if param.requires_grad:
                print(f"  -> Trainable: Yes")
            else:
                print(f"  -> Trainable: No")
            print(f"  -> Stats - Min: {param.data.min().item():.4f}, Max: {param.data.max().item():.4f}, Mean: {param.data.mean().item():.4f}")
            soft_prompt_found = True
    
    if not soft_prompt_found:
        print("❌ WARNING: No soft prompt parameters found!")
        print("Available parameters:")
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(f"  {name}: {param.shape}")
    else:
        print("✓ Soft prompt parameters loaded successfully!")

    print("Generating responses...")
    with torch.inference_mode():
        generated_responses = generate_responses(dataset, model, processor)

    print(f"Saving results to: {output_path}")
    with open(output_path / 'generated_responses.json', 'w') as f:
        json.dump(generated_responses, f, indent=2)

    scores = compute_scores(generated_responses)
    print("\nResults:")
    for key, value in scores.items():
        print(f"{key}: {value}")
    
    with open(output_path / 'scores.json', 'w') as f:
        json.dump(scores, f, indent=2)

    print(f"\nEvaluation complete! Results saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_data_dir", type=str, required=True)
    parser.add_argument("--output_data_dir", type=str, required=True) 
    parser.add_argument("--softprompt_model", type=str, required=True)

    args = parser.parse_args()
    main(args)