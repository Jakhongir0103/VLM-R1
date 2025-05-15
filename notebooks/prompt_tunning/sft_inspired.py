# Copyright 2025 The HuggingFace Team. All rights reserved.
# Adapted for soft prompt tuning

import os
import sys
import pathlib
import logging
import random
import wandb

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
    if 'image_path' in example and example['image_path'] is not None:
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
            ]
        }
    else:
        return {
            'messages': [
                {
                    'role': 'user',
                    'content': [{'type': 'text', 'text': example['problem']}]
                },
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": example['solution']}],
                }
            ]
        }

SOFTPROMPT_LEN = 50
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

    # Save
    logger.info("*** Save model ***")
    torch.cuda.synchronize()
    trainer.save_model(os.path.join(training_args.output_dir, 'final'))
    if training_args.push_to_hub:
        trainer.push_to_hub()

if __name__ == "__main__":
    parser = TrlParser((ScriptArguments, SFTConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)
