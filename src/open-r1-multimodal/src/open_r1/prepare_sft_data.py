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
import json
import argparse
import random

from transformers import set_seed

from open_r1.utils.format_prompt import format_prompt

# Format into conversation
def make_conversation(sample, data_dir, explanation_type, system_prompt):
    # https://github.com/QwenLM/Qwen2.5-VL/blob/fe0d43a3b74d70b40d28062c8b44d05978a0ed98/qwen-vl-utils/src/qwen_vl_utils/vision_process.py#L112C1-L113C1

    sample_formatted = format_prompt(sample, explanation_type=explanation_type)
    image_path = os.path.join(data_dir, sample['img_filename'])
    
    messages = [
        {
            'role': 'user',
            'content': "<image>" + sample_formatted['prompt']
        },
        {
            "role": "assistant",
            "content": sample_formatted['response']
        }
    ]
    images = [image_path]

    if system_prompt:
        bbox_system_prompt = "As an AI assistant, you specialize in accurate image object detection, delivering coordinates in plain text format 'x1,y1,x2,y2 object'."
        return {'messages': messages, 'images': images, 'system_prompt': bbox_system_prompt}
    else:
        return {'messages': messages, 'images': images}

def main(args):
    # Set seed for reproducibility
    set_seed(args.seed)

    # Load dataset
    with open(args.data_dir + '/DrivingVQA/train.json', "r") as f:
        data = list(json.load(f).values())

    # Remove double questions: 3142 -> 2248
    data = [d for d in data if not d['has_multiple_questions']]

    # Filter out large images: 2248 -> 1885
    data = [d for d in data if d['img_size'][0] * d['img_size'][1] <= 3686400]

    # TODO: used when we split the data into SFT and GRPO
    random.shuffle(data)

    # driving vqa -- Keep 1/2
    # data = data[:1000] # for SFT
    # data = data[1000:] # for GRPO

    # Map the conversations
    data = [make_conversation(sample, args.data_dir, args.explanation_type, args.system_prompt) for sample in data]

    # Save
    with open(args.data_dir + f"/aokvqa/sft/{args.output_name}.json", "w+") as f:
        json.dump(data, f, indent=4)

    print(data[0])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--explanation_type", type=str, required=True, help="bbox | original")
    parser.add_argument("--output_name", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--system_prompt", action="store_true")
    args = parser.parse_args()

    main(args)