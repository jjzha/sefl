import argparse
import json
import time
from tqdm import tqdm
from datetime import datetime
from typing import List, Dict
import logging

import autogen
from autogen import ConversableAgent

from datasets import load_dataset

from src.hf_api_model import APIModelClient
from src.custom_model import CustomModelClient


def setup_logging(
    log_file: str = f"_logs/{datetime.now().strftime('%Y_%m_%d_%H:%M:%S')}_chat_simulation_llama.log",
):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )


def output_writer(output_file_path: str, chat_history: List[Dict[str, str]]) -> None:
    try:
        with open(output_file_path, "a", encoding="utf-8") as f_out:
            json.dump({"conversation": chat_history}, f_out)
            f_out.write("\n")
    except IOError as e:
        logging.error(f"Error writing to output file: {e}")


def create_agent(
    args: argparse.Namespace, name: str, system_message: str, config_list: List[Dict]
) -> ConversableAgent:
    return ConversableAgent(
        name=name,
        system_message=system_message,
        llm_config={
            "config_list": config_list,
            "max_tokens": args.max_new_tokens,
            "cache_seed": 44,
        },
    )


def load_config_list(config_file: str, model_name: str) -> List[Dict]:
    return autogen.config_list_from_json(
        config_file,
        filter_dict={"model": [model_name]},
    )


def process_dataset(
    dataset,
    args: argparse.Namespace,
    student_agent: ConversableAgent,
    teacher_agent: ConversableAgent,
):
    num_conversations = 0
    with tqdm(total=len(dataset), desc="Processing dataset") as pbar:
        while args.output_conversations_num > num_conversations:
            for batch in dataset.iter(batch_size=args.batch_size):
                for text in batch[args.column_name]:
                    if len(text.split()) > 5000:
                        continue
                    try:
                        chat_result = student_agent.initiate_chat(
                            teacher_agent,
                            message=f"{text} {args.prompt}",
                            summary_method="reflection_with_llm",
                            max_turns=args.max_turns,
                        )
                        output_writer(args.output_file, chat_result.chat_history)
                        pbar.update(1)
                        num_conversations += 1
                    except Exception as e:
                        logging.error(f"Cannot process: {e}")
                        time.sleep(3600)
                        continue
        
        else:
            logging.info(f"Reached maximum number of conversations {args.output_conversations_num}. Quitting...")
            time.sleep(5)
            exit(1)


def main(args: argparse.Namespace):
    setup_logging()

    # config_list_student = load_config_list(
    #     "configs/API_CONFIG_LIST_1", "Qwen/Qwen2.5-72B-Instruct"
    # )
    # config_list_teacher = load_config_list(
    #     "configs/API_CONFIG_LIST_2", "Qwen/Qwen2.5-72B-Instruct"
    # )
    config_list_student = load_config_list(
        "configs/API_CONFIG_LIST_1", "meta-llama/Llama-3.1-70B-Instruct"
    )
    config_list_teacher = load_config_list(
        "configs/API_CONFIG_LIST_2", "meta-llama/Llama-3.1-70B-Instruct"
    )

    student_agent = create_agent(
        args=args,
        name="Student",
        system_message="You are a diligent student who solves all assignments efficiently. Your key traits are: 1. Direct and Concise Answers: Answer questions directly and concisely; use appropriate academic language. 2. Show Your Work: Demonstrate your problem-solving process; provide step-by-step solutions when necessary. 3. Encourage Learning: Focus on assisting with academic tasks; promote understanding through your answers. 4. Intentional Mistakes: Make some obvious mistakes that the teacher can give feedback on; ensure mistakes are explicit and noticeable. 5. Response Format: When responding to the teacher's assignment, give your answer and make explicit errors in your answer in valid JSON Lines (JSONL) format without any additional text, using the structure: {'answer': 'Your answer here', 'error_1': 'Description of the first mistake', 'error_2': 'Description of the second mistake'}. Do not write anything else.",
        config_list=config_list_student,
    )
    teacher_agent = create_agent(
        args=args,
        name="Instructor",
        system_message="You are a skilled teacher specializing in creating concise, effective assignments and providing constructive, targeted feedback. Your key responsibilities are: 1. Assignment Creation: Create short, clear assignments across various subjects; provide brief, focused instructions. 2. Feedback Provision: Offer constructive feedback on completed work; explain concepts succinctly when needed; do not give grades, only feedback for each mistake. 3. Encouragement and Adaptation: Encourage critical thinking and creativity; adapt to different learning styles and levels. 4. Response Format: When creating an assignment, give your answer in valid JSON format using {'assignment': 'Your assignment text here', 'task': 'Specific task instructions here'}; when providing feedback on a student's reply, respond in valid JSONL format with {'answer': 'Your global feedback here', 'feedback_1': 'Feedback on the first mistake', 'feedback_2': 'Feedback on the second mistake'}. Your goal is to facilitate learning through well-designed tasks and helpful guidance.",
        config_list=config_list_teacher,
    )

    student_agent.register_model_client(model_client_cls=APIModelClient)
    teacher_agent.register_model_client(model_client_cls=APIModelClient)

    if args.dataset_name:
        dataset = load_dataset(
            args.dataset_name, data_files="sample/10BT/003_0000.parquet", split="train"
        )
        process_dataset(dataset, args, student_agent, teacher_agent)
    else:
        chat_result = teacher_agent.initiate_chat(
            student_agent,
            message=args.prompt,
            summary_method="reflection_with_llm",
            max_turns=args.max_turns,
        )
        output_writer(args.output_file, chat_result.chat_history)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM Chat Simulation Script")

    # Data arguments
    parser.add_argument(
        "--batch_size", type=int, default=4, help="Batch size for dataset processing"
    )
    parser.add_argument(
        "--dataset_name",
        default="HuggingFaceFW/fineweb-edu",
        help="HuggingFace dataset name",
    )
    parser.add_argument(
        "--column_name", default="text", help="Column name for text in dataset"
    )

    # Generation arguments
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=256,
        help="Maximum number of new tokens to generate",
    )
    parser.add_argument("--prompt", default="", help="Initial prompt for chat")
    parser.add_argument(
        "--max_turns",
        type=int,
        default=5,
        help="Maximum number of turns in the conversation",
    )

    # Output arguments
    parser.add_argument(
        "--output_file",
        default=f"_generation_output/out_{datetime.now().strftime('%Y_%m_%d_%H:%M:%S')}_llama.json",
        help="Path to output file",
    )
    parser.add_argument(
        "--output_conversations_num",
        default=10000,
        help="Numbers of conversations to have.",
        type=int,
    )

    args = parser.parse_args()
    main(args)
