# Copyright 2025 The HuggingFace Team. All rights reserved.
# Adapted for soft prompt tuning
import re
import os
import sys
import pathlib
import logging
import random
import wandb
from tqdm import tqdm

from typing import List

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score
)
from statsmodels.stats.proportion import proportion_confint

import torch
import datasets
from datasets import load_from_disk

import transformers
from transformers import set_seed, AutoProcessor, Qwen2_5_VLForConditionalGeneration

from trl import ModelConfig, SFTConfig, ScriptArguments, SFTTrainer, TrlParser

from peft import PromptTuningConfig, TaskType, get_peft_model

from qwen_vl_utils import process_vision_info

logger = logging.getLogger(__name__)
processor = None

# Format into conversation
def make_conversation(example):
    assert 'image_path' in example and example['image_path'] is not None
    return {
        'messages': [
            {
                'role': 'user',
                'content': [
                    {'type': 'image', 'image': f"file://{example['image_path']}"},
                    {'type': 'text', 'text': example['problem']}
                ]
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": example['solution']}],
            }
        ],
        'solution': example['solution']
    }

SOFTPROMPT_LEN = 5
# Collation function
def collate_fn(examples):
    # Apply chat template to text
    texts = [
        processor.apply_chat_template(example["messages"], tokenize=False, add_generation_prompt=True)
        for example in examples
    ]

    # Process images (or videos)
    image_inputs = []
    for example in examples:
        imgs, vids = process_vision_info(example["messages"])
        image_inputs.append(imgs)

    # Tokenize multimodal input
    batch = processor(
        text=texts,
        images=image_inputs,
        return_tensors="pt",
        padding=True,
    )

    # Clone input_ids to use as labels
    labels = batch["input_ids"].clone()

    # Mask out pad tokens
    labels[labels == processor.tokenizer.pad_token_id] = -100

    # Mask out special image tokens (they aren't part of the loss)
    image_token_id = processor.tokenizer.convert_tokens_to_ids(processor.image_token)
    labels[labels == image_token_id] = -100

    # 🔥 Mask out soft prompt tokens (first N tokens)
    labels[:, :SOFTPROMPT_LEN] = -100

    batch["labels"] = labels
    return batch



class PatchedSFTTrainer(SFTTrainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits[:, SOFTPROMPT_LEN:, :]  # Trim the soft prompt logits
        loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
        loss = loss_fct(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
        return (loss, outputs) if return_outputs else loss


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

# Main training function
def main(script_args, training_args, model_args):
    set_seed(training_args.seed)

    wandb.init(
        project="vlm_r1",  # replace this with your actual project name
        name=training_args.run_name if hasattr(training_args, "run_name") else None,
        config={
            "soft_prompt_length": SOFTPROMPT_LEN,
            "script_args": vars(script_args),
            "training_args": vars(training_args),
            "model_args": vars(model_args),
        },
    )

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f" distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Model parameters {model_args}")
    logger.info(f"Script parameters {script_args}")
    logger.info(f"Data parameters {training_args}")

    # Load dataset
    dataset = load_from_disk(script_args.dataset_name)['train']
    # dataset = dataset.select(random.sample(range(len(dataset)), 1000))

    # Preprocess
    dataset = dataset.map(lambda sample: {
        "problem": f'Is the following statement true: {sample["caption"]}',
        "solution": str(sample["label"]==1)
    }, remove_columns=["caption", "label", "relation", "subj", "obj"], desc="Preprocessing dataset")

    dataset = [make_conversation(sample) for sample in dataset]
    # dataset = dataset[:10]
    print(f"Dataset size: {len(dataset)}")

    # Load processor
    global processor
    processor = AutoProcessor.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=model_args.trust_remote_code,
        use_fast=True,
    )
    logger.info("Using AutoProcessor for vision-language model.")
    if hasattr(processor, "pad_token") and processor.pad_token is None:
        processor.pad_token = processor.eos_token
    elif hasattr(processor.tokenizer, "pad_token") and processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    # Load model
    logger.info("*** Initializing model kwargs ***")
    model_kwargs = dict(
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation,
        torch_dtype=model_args.torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True
    )
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_args.model_name_or_path, **model_kwargs
    )

    # Set soft prompt tuning config
    peft_config = PromptTuningConfig(
        task_type=TaskType.CAUSAL_LM,
        # prompt_tuning_init="TEXT",  # or "RANDOM"
        # prompt_tuning_init_text="Answer the following question:",
        num_virtual_tokens=SOFTPROMPT_LEN,
        tokenizer_name_or_path=model_args.model_name_or_path,
    )

    # model.embeddings.word_embeddings = model.get_input_embeddings()
    model = get_peft_model(model, peft_config)
    model.word_embeddings = model.base_model.get_input_embeddings()



    # Trainer setup
    training_args.dataset_kwargs = {"skip_prepare_dataset": True}
    training_args.remove_unused_columns = False

    trainer = PatchedSFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=None,
        processing_class=processor.tokenizer,
        data_collator=collate_fn,
        peft_config=peft_config
    )

    # Training
    logger.info("*** Train ***")
    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        # trainer.train(resume_from_checkpoint=True)
        trainer.train()
    else:
        trainer.train()

    logger.info("*** Save model and soft prompt ***")
    torch.cuda.synchronize()
    # Save the soft prompt adapter and tokenizer
    soft_prompt_dir = os.path.join(training_args.output_dir, "soft_prompt")
    model.save_pretrained(soft_prompt_dir)
    processor.save_pretrained(soft_prompt_dir)

    # (Optional) Also save the full base model + trainer state
    trainer.save_model(os.path.join(training_args.output_dir, "final"))

    # Push to Hub if needed
    if training_args.push_to_hub:
        trainer.push_to_hub()
    
    # Eval


    dataset = load_from_disk(script_args.dataset_name)['validation']
    dataset = dataset.map(lambda sample: {
        "problem": f'Is the following statement true: {sample["caption"]}',
        "solution": str(sample["label"]==1)
    }, remove_columns=["caption", "label", "relation", "subj", "obj"], desc="Preprocessing dataset")

    # apply formatting
    dataset = [make_conversation(sample) for sample in dataset]
    dataset = dataset[:30] 
    model.eval()

    with torch.inference_mode():
        generated_responses = generate_responses(dataset, model, processor)

    scores = compute_scores(generated_responses)
    # WandB log scores
    wandb.log(scores)
    print(f"Scores: {scores}")
    logger.info(f"Scores: {scores}")

    


if __name__ == "__main__":
    parser = TrlParser((ScriptArguments, SFTConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)
