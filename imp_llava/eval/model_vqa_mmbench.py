import argparse
import json
import math
import os

from PIL import Image
import pandas as pd
import shortuuid
import torch
from tqdm import tqdm

from imp_llava.constants import (
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    IMAGE_TOKEN_INDEX,
)
from imp_llava.conversation import SeparatorStyle, conv_templates
from imp_llava.model.builder import load_pretrained_model
from imp_llava.mm_utils import (
    KeywordsStoppingCriteria,
    load_image_from_base64,
    process_images,
    tokenizer_image_token,
)
from imp_llava.utils import disable_torch_init


ALL_OPTIONS = ['A', 'B', 'C', 'D']


def split_list(lst, n):
    """
    Split a list into n roughly equal-sized chunks.
    
    Args:
        lst (list): The list to split.
        n (int): Number of chunks.
        
    Returns:
        list of lists: Split chunks.
    """
    chunk_size = math.ceil(len(lst) / n)
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    """
    Retrieve the k-th chunk from a list split into n chunks.
    
    Args:
        lst (list): The list to split.
        n (int): Total number of chunks.
        k (int): Index of the chunk to retrieve.
        
    Returns:
        list: The k-th chunk.
    """
    chunks = split_list(lst, n)
    return chunks[k]


def is_none(value):
    """
    Check if a value is considered None or invalid.
    
    Args:
        value: The value to check.
        
    Returns:
        bool: True if the value is None, NaN, or equivalent string.
    """
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    if isinstance(value, str):
        return value.strip().lower() in {'nan', 'none'}
    return False


def get_options(row, options):
    """
    Extract available options from a row.
    
    Args:
        row (pd.Series): The row containing option data.
        options (list): List of option keys to extract.
        
    Returns:
        list: Parsed options.
    """
    parsed_options = []
    for option in options:
        option_value = row.get(option)
        if is_none(option_value):
            break
        parsed_options.append(option_value)
    return parsed_options


def eval_model(args):
    """
    Evaluate the model on the provided questions and save the answers.
    
    Args:
        args (argparse.Namespace): Command-line arguments.
    """
    # Disable initial torch settings for faster loading
    disable_torch_init()
    
    # Load the pretrained model and tokenizer
    model_path = os.path.expanduser(args.model_path)
    model_name = load_pretrained_model.get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path, args.model_base, model_name
    )

    keywords = ['</s>']

    # Load questions from file
    questions = pd.read_table(os.path.expanduser(args.question_file))
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)

    # Prepare answers file
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    
    with open(answers_file, "w") as ans_file:
        # Adjust conversation mode if necessary
        if ('plain' in model_name and 
            'finetune' not in model_name.lower() and 
            'mmtag' not in args.conv_mode):
            args.conv_mode += '_mmtag'
            print(
                f'It seems that this is a plain model, but it is not using a mmtag prompt. '
                f'Auto-switching to {args.conv_mode}.'
            )

        # Iterate over each question
        for _, row in tqdm(questions.iterrows(), total=len(questions), desc="Processing Questions"):
            options = get_options(row, ALL_OPTIONS)
            current_option_chars = ALL_OPTIONS[:len(options)]

            num_rounds = len(options) if args.all_rounds else 1

            for round_idx in range(num_rounds):
                idx = row['index']
                question = row['question']
                hint = row.get('hint')
                image = load_image_from_base64(row.get('image'))
                
                if not is_none(hint):
                    question = f"{hint}\n{question}"
                
                # Append options to the question
                for option_char, option in zip(ALL_OPTIONS[:len(options)], options):
                    question += f'\n{option_char}. {option}'
                
                prompt_text = question
                if model.config.mm_use_im_start_end:
                    prompt_text = f"{DEFAULT_IM_START_TOKEN}{DEFAULT_IMAGE_TOKEN}{DEFAULT_IM_END_TOKEN}\n{prompt_text}"
                else:
                    prompt_text = f"{DEFAULT_IMAGE_TOKEN}\n{prompt_text}"

                # Add single prediction prompt if specified
                if args.single_pred_prompt:
                    if args.lang.lower() == 'cn':
                        prompt_text += '\n请直接回答选项字母。'
                    else:
                        prompt_text += "\nAnswer with the option's letter from the given choices directly."

                # Create conversation template
                conv = conv_templates[args.conv_mode].copy()
                conv.append_message(conv.roles[0], prompt_text)
                conv.append_message(conv.roles[1], None)
                prompt = conv.get_prompt()

                # Tokenize input
                input_ids = tokenizer_image_token(
                    prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt'
                ).unsqueeze(0).cuda()

                stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

                # Process image
                image_tensor = process_images([image], image_processor, model.config)[0]

                # Determine stopping string
                stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2

                # Generate model output
                with torch.inference_mode():
                    output_ids = model.generate(
                        input_ids,
                        images=image_tensor.unsqueeze(0).half().cuda(),
                        do_sample=True if args.temperature > 0 else False,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        num_beams=args.num_beams,
                        stopping_criteria=[stopping_criteria],
                        max_new_tokens=1024,
                        use_cache=True
                    )

                # Decode the output
                input_token_len = input_ids.shape[1]
                n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
                if n_diff_input_output > 0:
                    print(f'[Warning] {n_diff_input_output} output_ids differ from input_ids')

                outputs = tokenizer.batch_decode(
                    output_ids[:, input_token_len:], skip_special_tokens=True
                )[0].strip()

                # Remove stopping string if present
                if outputs.endswith(stop_str):
                    outputs = outputs[:-len(stop_str)].strip()

                # Prepare answer record
                ans_record = {
                    "question_id": idx,
                    "round_id": round_idx,
                    "prompt": question,
                    "text": outputs,
                    "options": options,
                    "option_char": current_option_chars,
                    "answer_id": shortuuid.uuid(),
                    "model_id": model_name,
                    "metadata": {}
                }

                # Write answer to file
                ans_file.write(json.dumps(ans_record) + "\n")
                ans_file.flush()

                # Rotate options for next round if applicable
                options = options[1:] + options[:1]
                current_option_chars = current_option_chars[1:] + current_option_chars[:1]


def main():
    """Parse arguments and initiate model evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate a pretrained model on questions with images.")
    
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m",
                        help="Path to the pretrained model.")
    parser.add_argument("--model-base", type=str, default=None,
                        help="Base model name.")
    parser.add_argument("--image-folder", type=str, default="",
                        help="Folder containing images (if applicable).")
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl",
                        help="Path to the question file in JSONL format.")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl",
                        help="Path to save the answers in JSONL format.")
    parser.add_argument("--conv-mode", type=str, default="llava_v1",
                        help="Conversation mode/template to use.")
    parser.add_argument("--num-chunks", type=int, default=1,
                        help="Number of chunks to split the questions into.")
    parser.add_argument("--chunk-idx", type=int, default=0,
                        help="Index of the chunk to process.")
    parser.add_argument("--temperature", type=float, default=0.2,
                        help="Sampling temperature for generation.")
    parser.add_argument("--top_p", type=float, default=None,
                        help="Top-p (nucleus) sampling probability.")
    parser.add_argument("--num_beams", type=int, default=1,
                        help="Number of beams for beam search.")
    parser.add_argument("--all-rounds", action="store_true",
                        help="Whether to perform all rounds of option processing.")
    parser.add_argument("--single-pred-prompt", action="store_true",
                        help="Whether to use a single prediction prompt.")
    parser.add_argument("--lang", type=str, default="en",
                        help="Language for the prompt (e.g., 'en' or 'cn').")
    
    args = parser.parse_args()
    eval_model(args)


if __name__ == "__main__":
    main()
