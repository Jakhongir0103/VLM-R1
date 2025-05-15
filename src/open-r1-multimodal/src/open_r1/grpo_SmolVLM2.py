import os
import sys
import pathlib
import logging
import random

import torch
from PIL import Image  # for loading images

import datasets
from datasets import load_from_disk

import transformers
from transformers import set_seed, AutoProcessor, AutoModelForImageTextToText, Idefics3ForConditionalGeneration, Idefics3Processor
from transformers import AutoModelForSequenceClassification, AutoTokenizer, PreTrainedModel

from trl import ModelConfig, ScriptArguments, TrlParser, get_peft_config, GRPOTrainer, GRPOConfig
from vlm_modules.qwen_module import Qwen2VLModule

import re
from typing import List
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from statsmodels.stats.proportion import proportion_confint
from transformers import EvalPrediction

logger = logging.getLogger(__name__)

processor = None

def normalize_answer(answer: str) -> str:
    # retain letters only and lowercase
    return re.sub(r'[^a-zA-Z]', '', answer).lower()

def extract_answer(text: str) -> str:
    """
    Extract the model or label text and map Yes/No/True/False to canonical "true"/"false".
    """
    m = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    raw = m.group(1) if m else text
    raw = raw.strip().lower()
    # map variants
    if re.search(r"\b(true|yes)\b", raw):
        return "true"
    if re.search(r"\b(false|no)\b", raw):
        return "false"
    # fallback normalization
    return normalize_answer(raw)

def compute_scores_list(true_list: List[str], pred_list: List[str]):
    n = len(true_list)
    correct = sum(t == p for t, p in zip(true_list, pred_list))
    acc = 100 * correct / n if n else 0
    prec = precision_score(true_list, pred_list, average="weighted", zero_division=0)
    rec  = recall_score(true_list, pred_list, average="weighted", zero_division=0)
    f1   = f1_score(true_list, pred_list, average="weighted", zero_division=0)
    low, high = proportion_confint(count=correct, nobs=n, alpha=0.05, method="wilson") if n else (None, None)
    return {
        "eval_accuracy": acc,
        "eval_accuracy_ci_95_low": low,
        "eval_accuracy_ci_95_high": high,
        "eval_precision": prec,
        "eval_recall": rec,
        "eval_f1": f1,
    }

def compute_metrics(eval_pred: EvalPrediction):
    logits_or_ids, label_ids = eval_pred
    if logits_or_ids.ndim == 3:
        preds = torch.argmax(torch.tensor(logits_or_ids), dim=-1).numpy()
    else:
        preds = logits_or_ids
    pred_texts  = processor.tokenizer.batch_decode(preds,      skip_special_tokens=True)
    label_texts = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
    pred_ans  = [extract_answer(t) for t in pred_texts]
    label_ans = [extract_answer(t) for t in label_texts]
    return compute_scores_list(label_ans, pred_ans)

def make_conversation(example):
    # identical to before, message-format stays the same
    if 'image_path' in example and example['image_path'] is not None:
        return {
            'messages': [
                {
                    'role': 'user',
                    'content': [
                        {'type': 'image', 'image': f"file://{example['image_path']}"},
                        {'type': 'text',  'text': example['problem']}
                    ]
                },
                {
                    'role': 'assistant',
                    'content': [{'type': 'text', 'text': example['solution']}]
                }
            ]
        }
    else:
        return {
            'messages': [
                {'role': 'user',      'content': [{'type': 'text', 'text': example['problem']}]},
                {'role': 'assistant', 'content': [{'type': 'text', 'text': example['solution']}]}  
            ]
        }

def collate_fn(examples):
    # 1) build text prompts exactly as before
    texts = [
        processor.apply_chat_template(example["messages"], tokenize=False, add_generation_prompt=True)
        for example in examples
    ]

    # 2) load raw PIL images from the file:// URIs in messages
    image_inputs = []
    for example in examples:
        imgs = []
        for msg in example["messages"]:
            for c in msg["content"]:
                if c["type"] == "image":
                    path = c["image"].replace("file://", "")
                    imgs.append(Image.open(path).convert("RGB"))
        image_inputs.append(imgs)

    # 3) pack into tensors just like HuggingFace’s VLM tutorial
    batch = processor(
        text=texts,
        images=image_inputs,
        return_tensors="pt",
        padding=True,
    )

    # 4) labels: mask out pads and leave rest to generate
    labels = batch["input_ids"].clone()
    labels[labels == processor.tokenizer.pad_token_id] = -100
    batch["labels"] = labels

    return batch

def find_all_linear_names(model, multimodal_keywords):
    cls = torch.nn.Linear
    lora_module_names = set()
    for name, module in model.named_modules():
        if any(mm_keyword in name for mm_keyword in multimodal_keywords):
            continue
        if isinstance(module, cls):
            lora_module_names.add(name)
    lora_module_names = {
        name for name in lora_module_names
        if "embed_tokens" not in name and "lm_head" not in name
    }
    return list(lora_module_names)

def main(script_args, training_args, model_args):
    set_seed(training_args.seed)
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
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, "
        f"n_gpu: {training_args.n_gpu}, distributed: {bool(training_args.local_rank != -1)}, "
        f"16-bit: {training_args.fp16}"
    )
    logger.info(f"Model parameters {model_args}")
    logger.info(f"Script parameters {script_args}")
    logger.info(f"Data parameters {training_args}")

    dataset = load_from_disk(script_args.dataset_name)["train"]
    print(f"Train Dataset Size: {len(dataset)}")

    random_indices = random.sample(range(len(dataset)), min(1000, len(dataset)))
    dataset = dataset.select(random_indices)

    # preprocess the dataset
    dataset = dataset.map(lambda sample: {
        "problem": f'Is the following statement true: {sample["caption"]}',
        "solution": str(sample["label"]==1)
    }, remove_columns=["caption", "label", "relation", "subj", "obj"], desc="Preprocessing dataset")

    # Reuse same question prompt as for Qwen2.5-VL
    question_prompt = Qwen2VLModule.get_question_template(task_type="default")

    # Format into conversation
    def make_conversation(example):
        if 'image_path' in example and example['image_path'] is not None:
            return {
                'image_path': example['image_path'],  # Store path instead of loaded image
                'problem': example['problem'],
                'solution': f"<answer> {example['solution']} </answer>",
                'prompt': [{
                    'role': 'user',
                    'content': [
                        {'type': 'image', 'text': None},
                        {'type': 'text', 'text': question_prompt.format(Question=example['problem'])}
                    ]
                }]
            }
        else:
            return {
                'problem': example['problem'],
                'solution': f"<answer> {example['solution']} </answer>",
                'prompt': [{
                    'role': 'user',
                    'content': [
                        {'type': 'text', 'text': question_prompt.format(Question=example['problem'])}
                    ]
                }]
            }

    # Map the conversations
    dataset = dataset.map(make_conversation)

    global processor
    processor = Idefics3Processor.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=model_args.trust_remote_code
    )
    logger.info("Loaded SmolVLM AutoProcessor.")

    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    model_kwargs = dict(
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation,
        torch_dtype=model_args.torch_dtype,
        use_cache=not training_args.gradient_checkpointing
    )
    model = Idefics3ForConditionalGeneration.from_pretrained(
        model_args.model_name_or_path,
        **model_kwargs
    )

    training_args.dataset_kwargs = {"skip_prepare_dataset": True}
    training_args.remove_unused_columns = False

    peft_config = get_peft_config(model_args)
    if peft_config is not None:
        peft_config.target_modules = find_all_linear_names(model, ["vision", "image", "pixel"])

    # --- Prepare reward functions & processing classes ---
    reward_funcs = [
        Qwen2VLModule.format_reward_rec,
        Qwen2VLModule.format_reward,
        Qwen2VLModule.iou_reward
    ]
    # 1) ensure list
    if not isinstance(reward_funcs, list):
        reward_funcs = [reward_funcs]
    # 2) load from strings if needed
    model_init_kwargs = dict(trust_remote_code=model_args.trust_remote_code)
    for i, rf in enumerate(reward_funcs):
        if isinstance(rf, str):
            reward_funcs[i] = AutoModelForSequenceClassification.from_pretrained(
                rf, num_labels=1, **model_init_kwargs
            )

    # 3) infer tokenizers / processing classes
    reward_processing_classes = [None] * len(reward_funcs)
    for i, rf in enumerate(reward_funcs):
        if isinstance(rf, PreTrainedModel):
            rpc = reward_processing_classes[i] or AutoTokenizer.from_pretrained(
                rf.config._name_or_path
            )
            if rpc.pad_token_id is None:
                rpc.pad_token = rpc.eos_token
            rf.config.pad_token_id = rpc.pad_token_id
            reward_processing_classes[i] = rpc

    # 4) pass both lists into trainer
    trainer = GRPOTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=None,
        processing_class=processor.tokenizer,
        peft_config=peft_config,
        reward_funcs=reward_funcs,
        reward_processing_classes=reward_processing_classes
    )

    logger.info("*** Train ***")
    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    logger.info("*** Save model ***")
    torch.cuda.synchronize()
    trainer.save_model(os.path.join(training_args.output_dir, "final"))
    if training_args.push_to_hub:
        trainer.push_to_hub()

if __name__ == "__main__":
    parser = TrlParser((ScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)
