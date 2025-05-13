import os
import sys
import pathlib
import logging
import random

import torch
import datasets
from datasets import load_from_disk

from torch.nn.utils.prune import remove
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
                    "content": [{"type": "text", "text": example['desired_output']}],
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
                    "content": [{"type": "text", "text": example['desired_output']}],
                }
            ]
        }

SOFTPROMPT_LEN = 10
# Collation function

def collate_fn(examples):
    return {
        "training_inputs": {
            k: torch.stack([
                torch.tensor(ex["training_inputs"][k]).squeeze(0) if not isinstance(ex["training_inputs"][k], torch.Tensor) else ex["training_inputs"][k].squeeze(0)
                for ex in examples
            ])
            for k in examples[0]["training_inputs"]
        },
        "reasoning_logits_offset": torch.tensor([ex["reasoning_logits_offset"] for ex in examples]),
        "reasoning_logits": torch.stack([
            torch.tensor(ex["reasoning_logits"]) if not isinstance(ex["reasoning_logits"], torch.Tensor) else ex["reasoning_logits"]
            for ex in examples
        ])
    }





class PatchedSFTTrainer(SFTTrainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # inputs["training_inputs"]["attention_mask"] = inputs["training_inputs"]["attention_mask"].squeeze(1)

        outputs = model(**inputs['training_inputs'])
        logits_without_softprompt = outputs.logits[:, SOFTPROMPT_LEN:, :]  # Trim the soft prompt logits
        logits = logits_without_softprompt[:, inputs['reasoning_logits_offset']:]
        target_logits = inputs['reasoning_logits']

        loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
        loss = loss_fct(logits, target_logits)
        return (loss, outputs) if return_outputs else loss


# Main training function
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
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f" distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Model parameters {model_args}")
    logger.info(f"Script parameters {script_args}")
    logger.info(f"Data parameters {training_args}")

    # Load dataset
    dataset = load_from_disk(os.environ['DATA_PATH'] + "/vsr_prompt_tuning")['train']
    print(f"Dataset size: {len(dataset)}")
    sample = dataset[0]


    # dataset = [make_conversation(sample) for sample in dataset]

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
        peft_config=peft_config,
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
