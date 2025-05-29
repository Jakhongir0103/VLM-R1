def make_conversation_from_prompt(example, prompt_template, answer = None):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": f"file://{example['image_path']}"},
                {"type": "text", "text": prompt_template(example)},
            ],
        }
    ]
    if answer is not None:
        messages.append(
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": answer},
                ],
            }
        )
    return {"messages": messages, "solution": example['solution']}

# apply formatting
def reasoning_prompt_template(example):
    return f"{example['problem']} First output the thinking process in <think> </think> tags and then output the final answer in <answer> </answer> tags."

def baseline_prompt_template(example):
    return f"{example['problem']} First output the thinking process in <think> </think> tags and then output the final answer in <answer> </answer> tags."