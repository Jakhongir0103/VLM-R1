# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Implementation is adapted from: https://huggingface.co/learn/cookbook/fine_tuning_vlm_trl
"""

import os
import sys
import pathlib
import logging
import random
import json

from dataclasses import dataclass, field

import torch

import datasets
import transformers
from transformers import set_seed, AutoProcessor, Qwen2_5_VLForConditionalGeneration

from trl import SFTConfig, ModelConfig, ScriptArguments, SFTTrainer, TrlParser, get_peft_config
from peft import get_peft_model

from qwen_vl_utils import process_vision_info

from open_r1.utils.format_prompt import format_prompt

logger = logging.getLogger(__name__)

processor = None

@dataclass
class SFTScriptArguments(ScriptArguments):
    explanation_type: str = field(default='bbox', metadata={"help": "Possible values: [original, bbox]"})

# Format into conversation
def make_conversation(sample, data_dir, explanation_type):
    # https://github.com/QwenLM/Qwen2.5-VL/blob/fe0d43a3b74d70b40d28062c8b44d05978a0ed98/qwen-vl-utils/src/qwen_vl_utils/vision_process.py#L112C1-L113C1

    sample_formatted = format_prompt(sample, explanation_type=explanation_type)
    image_path = os.path.join(data_dir, sample['img_filename'])

    return {
        'messages': [
            {
                'role': 'user',
                'content': [
                    {'type': 'image', 'image': f"file://{image_path}"},
                    {'type': 'text', 'text': sample_formatted['prompt']}
                ]
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": sample_formatted['response']}],
            }
        ]
    }
        
def collate_fn(examples):
    texts = [
        processor.apply_chat_template(example["messages"], tokenize=False, add_generation_prompt=True)
        for example in examples
    ]
    image_inputs = []
    for example in examples:
        imgs, vids = process_vision_info(example["messages"])
        image_inputs.append(imgs)
    batch = processor(
        text=texts,
        images=image_inputs,
        return_tensors="pt",
        padding=True,
    )
    labels = batch["input_ids"].clone()
    labels[labels == processor.tokenizer.pad_token_id] = -100
    image_token_id = processor.tokenizer.convert_tokens_to_ids(processor.image_token)
    labels[labels == image_token_id] = -100
    batch["labels"] = labels

    return batch

def find_all_linear_names(model, multimodal_keywords):
    cls = torch.nn.Linear
    lora_module_names = set()
    for name, module in model.named_modules():
        # LoRA is not applied to the vision modules
        if any(mm_keyword in name for mm_keyword in multimodal_keywords):
            continue
        if isinstance(module, cls):
            lora_module_names.add(name)
    lora_module_names = {name for name in lora_module_names if "embed_tokens" not in name and "lm_head" not in name} # needed for 16-bit
    return list(lora_module_names)

def main(script_args, training_args, model_args):
    # Set seed for reproducibility
    set_seed(training_args.seed)

    # Setup logging
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

    # Log on each process a small summary
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f" distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Model parameters {model_args}")
    logger.info(f"Script parameters {script_args}")
    logger.info(f"Data parameters {training_args}")

    # Load dataset
    # `dataset_name`: dataset path to `.json` file
    with open(script_args.dataset_name + '/train.json', "r") as f:
        data = json.load(f)
    data = list(data.values())

    # Remove double questions: 3142 -> 2248
    data = [d for d in data if not d['has_multiple_questions']]

    # Filter out large images: 2248 -> 1885
    data = [d for d in data if d['img_size'][0] * d['img_size'][1] <= 1638400]
    print(len(data))
    
    # TODO: used when we split the data into SFT and GRPO
    random.shuffle(data)
    # data = data[:1000]

    # Map the conversations
    data = [make_conversation(sample, script_args.dataset_name, script_args.explanation_type) for sample in data]

    print(data[0])

    # Load tokenizer
    global processor
    processor = AutoProcessor.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=model_args.trust_remote_code
    )
    logger.info("Using AutoProcessor for vision-language model.")
    if hasattr(processor, "pad_token") and processor.pad_token is None:
        processor.pad_token = processor.eos_token
    elif hasattr(processor.tokenizer, "pad_token") and processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    
    # Model init kwargs
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

    # Initialize the SFT Trainer
    training_args.dataset_kwargs = {
        "skip_prepare_dataset": True,
    }
    training_args.remove_unused_columns = False

    # Get LoRA configs
    peft_config = get_peft_config(model_args)
    print(peft_config)
    if peft_config is not None:
        target_modules = find_all_linear_names(model, ['visual'])
        peft_config.target_modules = target_modules

        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

    if training_args.gradient_checkpointing:
        model.enable_input_require_grads()

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=data,
        eval_dataset=None,
        processing_class=processor.tokenizer,
        data_collator=collate_fn,
        peft_config=peft_config
    )

    # Training loop
    logger.info("*** Train ***")
    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    # Save model
    logger.info("*** Save model ***")
    torch.cuda.synchronize()
    trainer.save_model(os.path.join(training_args.output_dir, 'final'))
    if training_args.push_to_hub:
        trainer.push_to_hub()

if __name__ == "__main__":
    parser = TrlParser((SFTScriptArguments, SFTConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)