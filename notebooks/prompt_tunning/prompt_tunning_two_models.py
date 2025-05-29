import os
import sys
import pathlib
import logging
import random
import wandb

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

SOFTPROMPT_LEN = 20

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
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reasoning_model = 
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # inputs["training_inputs"]["attention_mask"] = inputs["training_inputs"]["attention_mask"].squeeze(1)
        offset = inputs['reasoning_logits_offset']

        outputs = model(**inputs['training_inputs'])
        logits_without_softprompt = outputs.logits[:, SOFTPROMPT_LEN:, :]  # Trim the soft prompt logits
        logits = logits_without_softprompt[:, offset:]
        target_logits = inputs['reasoning_logits'][:, offset:]

        assert logits.shape == target_logits.shape, f"Logits shape {logits.shape} does not match target shape {target_logits.shape}"
        
        relevant_mask = (inputs['training_inputs']['input_ids'] != processor.tokenizer.pad_token_id)[offset:]


        loss_fct = torch.nn.MSELoss()
        loss = loss_fct(logits, target_logits.to(logits.dtype))

        return (loss, outputs) if return_outputs else loss


# Main training function

def main(script_args, training_args, model_args):
    set_seed(training_args.seed)

    # Initialize Weights & Biases
    wandb.init(
        project="vlm_r1",  # replace this with your actual project name
        name=training_args.run_name if hasattr(training_args, "run_name") else None,
        config={
            "script_args": vars(script_args),
            "training_args": vars(training_args),
            "model_args": vars(model_args),
        },
    )

    wandb.log({
        "device": str(training_args.device),
        "n_gpu": training_args.n_gpu,
        "fp16": training_args.fp16,
        "distributed": training_args.local_rank != -1,
    })

    # Load dataset
    dataset = load_from_disk(os.environ['DATA_PATH'] + "/vsr")['train']
    wandb.log({"dataset_size": len(dataset)})


    # Load processor
    global processor
    processor = AutoProcessor.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=model_args.trust_remote_code,
        use_fast=True,
    )
    if hasattr(processor, "pad_token") and processor.pad_token is None:
        processor.pad_token = processor.eos_token
    elif hasattr(processor.tokenizer, "pad_token") and processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    wandb.log({"processor_type": type(processor).__name__})

    # Load model
    model_kwargs = dict(
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation,
        torch_dtype=model_args.torch_dtype,
        use_cache=not training_args.gradient_checkpointing
    )
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_args.model_name_or_path, **model_kwargs
        device_map="cuda:0"
    )

    # Set soft prompt tuning config
    peft_config = PromptTuningConfig(
        task_type=TaskType.CAUSAL_LM,
        num_virtual_tokens=SOFTPROMPT_LEN,
        tokenizer_name_or_path=model_args.model_name_or_path,
    )

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
    wandb.log({"training_started": True})
    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train()
    else:
        trainer.train()

    # Save
    torch.cuda.synchronize()
    trainer.save_model(os.path.join(training_args.output_dir, 'final'))
    wandb.log({"model_saved": True})
    if training_args.push_to_hub:
        trainer.push_to_hub()

if __name__ == "__main__":
    parser = TrlParser((ScriptArguments, SFTConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)
