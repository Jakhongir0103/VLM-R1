from transformers import set_seed, AutoProcessor, AutoModelForImageTextToText
from peft import PeftModel, PeftConfig
import torch
import os
import argparse

def merge_lora_adapter(baseline_model_path: str, adapter_model_path: str, output_path: str):
    """
    Merge a LoRA adapter into the base model and save the merged model.

    Args:
        baseline_model_path (str): Path to the base model (Hugging Face repo or local directory).
        adapter_model_path (str): Path to the LoRA adapter.
        output_path (str): Directory where the merged model should be saved.
    """
    # Load the base model and tokenizer
    print("[INFO] Loading base model")
    model = AutoModelForImageTextToText.from_pretrained(
        baseline_model_path,
        torch_dtype=torch.bfloat16
    )
    tokenizer = AutoProcessor.from_pretrained(
        baseline_model_path
    )

    # Load the adapter configuration and wrap the base model with the adapter
    print("[INFO] Loading LoRA adapter")
    model = PeftModel.from_pretrained(model, adapter_model_path)

    # Merge the LoRA weights into the base model
    print("[INFO] Merging adapter into base model")
    model = model.merge_and_unload()

    # Save the merged model
    print(f"[INFO] Saving merged model to {output_path}")
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    print("[INFO] Merge completed.")

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline_model_path", type=str, default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--adapter_model_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    args = parser.parse_args()

    merge_lora_adapter(
        baseline_model_path=args.baseline_model_path,
        adapter_model_path=args.adapter_model_path,
        output_path=args.output_path
    )