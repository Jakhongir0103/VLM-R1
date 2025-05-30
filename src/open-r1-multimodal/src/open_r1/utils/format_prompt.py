# TODO: change bbox formate in `interleaved_explanation`

import re
from typing import Dict, Tuple, Any

def xywh_to_xyxy(box):
    """Convert COCO format [x_min, y_min, width, height] to [x_min, y_min, x_max, y_max]."""
    x_min, y_min, width, height = box
    x_max = x_min + width
    y_max = y_min + height
    return [x_min, y_min, x_max, y_max]

def get_possible_answers(sample: Dict[str, Any]) -> Tuple[str, str]:
    """
    Extract possible answers from the sample data.

    Parameters:
    - sample (dict): The sample data containing possible answers.

    Returns:
    - tuple: A tuple containing possible answers for question 1 and question 2 (if applicable).
    """
    possible_answers = sample["possible_answers"]

    if sample["has_multiple_questions"]:
        answers_between_first_and_second_question = ['A', 'B']

        possible_answers_1_list = []
        possible_answers_2_list = []

        for key, value in possible_answers.items():
            answer_text = f"({key}) {value}"
            if key in answers_between_first_and_second_question:
                possible_answers_1_list.append(answer_text)
            else:
                possible_answers_2_list.append(answer_text)

        return " ".join(possible_answers_1_list), " ".join(possible_answers_2_list)
    else:
        possible_answers_list = [f"({key}) {value}" for key, value in possible_answers.items()]
        return " ".join(possible_answers_list), ""

def remove_bbox(text: str) -> str:
    """
    Removes list-like objects in square brackets (bbox) from the input string.
    
    Parameters:
    - text (str): The input string containing list-like objects.
    
    Returns:
    - str: The cleaned string with list-like objects removed.
    """
    # Regular expression pattern to match [number, number, ...]
    pattern = r"\[\s*\d+(?:\.\d+)?(?:\s*,\s*\d+(?:\.\d+)?)*\s*\]"
    cleaned_text = re.sub(pattern, "", text)
    cleaned_text = re.sub(r'\s{2,}', ' ', cleaned_text)
    return cleaned_text

def get_prompt_components(sample: Dict[str, Any], explanation_type: str) -> Dict[str, str]:
    """
    Get various prompt components based on the sample and explanation type.

    Parameters:
    - sample (dict): The sample data.
    - explanation_type (str): Type of explanation to use; [original, bbox]

    Returns:
    - dict: Dictionary containing prompt components.
    """
    # Main instruction
    main_instruction_prompt = "Select all correct answers to the following question from the available options."

    # Extract possible answers for the questions
    possible_answers_question_1, possible_answers_question_2 = get_possible_answers(sample)

    # Get explanation based on type
    if explanation_type == "original":
        think_instruction_prompt = "First think about the question in the mind using all relevant entities from the scene that are necessary to answer the question. Then, provide with the thinking process in <think> </think> tags and then output the final answer in <answer> </answer> tags."
        # explanation = sample['explanation']
        explanation = remove_bbox(sample['interleaved_explanation'])
    elif explanation_type == "bbox":
        # https://github.com/QwenLM/Qwen2.5-VL/blob/main/cookbooks/spatial_understanding.ipynb
        think_instruction_prompt = "First think about the question in the mind using all relevant entities from the scene that are necessary to answer the question, along with their bounding boxes coordinates in plain text format 'x1,y1,x2,y2 object'. Then, provide with the thinking process in <think> </think> tags and then output the final answer in <answer> </answer> tags."
        explanation = sample['interleaved_explanation']

        for relevant_entity in sample['relevant_entities']:
            entity = list(relevant_entity.keys())[0]
            bbox = relevant_entity[entity]
            bbox_values = [str(x) for x in bbox]
            rounded_bbox_values = [str(int(x)) for x in xywh_to_xyxy(bbox)]

            # Replace bbox format
            bbox_pattern = r"\[" + r"\s*,\s*".join(map(re.escape, bbox_values)) + r"\]"
            explanation = re.sub(bbox_pattern, f"[{','.join(rounded_bbox_values)} {entity}]", explanation)

    # Format question text
    if sample['has_multiple_questions']:
        main_instruction_prompt = main_instruction_prompt.replace("question", "questions")
        think_instruction_prompt = think_instruction_prompt.replace("question", "questions")
        main_instruction_prompt += " Choose at least one answer per question."
        q_text = (f"Question: {sample['questions'][0]}\nOptions: {possible_answers_question_1}.\n"
                  f"Question: {sample['questions'][1]}\nOptions: {possible_answers_question_2}.")
    else:
        if isinstance(sample['questions'], str):
            question = sample['questions']
        elif isinstance(sample['questions'], list):
            question = sample['questions'][0]
        else:
            raise Exception(f"Unknown question type: {type(sample['questions'])}")
        q_text = f"Question: {question}\nOptions: {possible_answers_question_1}."

    return {
        'main_instruction_prompt': main_instruction_prompt,
        'q_text': q_text,
        'answers_text': ", ".join(sample['true_answers']),
        'explanation': explanation,
        'think_instruction_prompt': think_instruction_prompt
    }

def format_prompt(sample: Dict[str, Any], explanation_type: str) -> Dict[str, str]:
    """
    Parameters:
    - sample (dict): The sample data.
    - explanation_type (str): Type of explanation to use; [original, bbox]

    Returns:
    - dict: Dictionary containing `prompt` and `response`.
    """
    prompt_components = get_prompt_components(sample, explanation_type=explanation_type)

    prompt = prompt_components['main_instruction_prompt'] + ' ' + prompt_components['q_text'] + '\n' + prompt_components['think_instruction_prompt']
    response = f"<think>{prompt_components['explanation']}</think> <answer>{prompt_components['answers_text']}</answer>"

    return {
        'prompt': prompt,
        'response': response
    }
