from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from datasets import load_from_disk
from qwen_vl_utils import process_vision_info

model_name_or_path = "Qwen/Qwen2.5-VL-3B-Instruct"
dataset_path = "/scratch/izar/smaldone/vlm_r1/data/vsr"

# default: Load the model on the available device(s)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_name_or_path, torch_dtype="auto", device_map="auto"
)

# default processor
processor = AutoProcessor.from_pretrained(model_name_or_path)

# preprocess the dataset
dataset = load_from_disk(dataset_path)['validation']

dataset = dataset.map(lambda sample: {
    "problem": f'Is the following statement true: {sample["caption"]}',
    "solution": str(sample["label"]==1)
}, remove_columns=["caption", "label", "relation", "subj", "obj"], desc="Preprocessing dataset")

def make_conversation(example):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": f"file://{example['image_path']}"},
                {"type": "text", "text": f"{example['problem']} First output the thinking process in <think> </think> tags and then output the final answer in <answer> </answer> tags."},
            ],
        }
    ]
    return {"messages": messages, "solution": example['solution']}

# apply formatting
dataset = [make_conversation(sample) for sample in dataset]

# ONLY FOR DEBUGGING    
dataset = dataset[1]
print("INPUT:")
print(dataset)
print("="*20)

# Preparation for inference
text = processor.apply_chat_template(
    dataset['messages'], tokenize=False, add_generation_prompt=True
)
image_inputs, video_inputs = process_vision_info(dataset['messages'])
inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
)
inputs = inputs.to(model.device)

# Inference: Generation of the output
generated_ids = model.generate(**inputs, max_new_tokens=256)
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print("OUTPUT:")
print(output_text)
print("="*20)