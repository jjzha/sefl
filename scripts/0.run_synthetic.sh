#!/bin/sh

python src/run_agents.py \
        --max_turns 2 \
        --max_new_tokens 8192 \
        --prompt "\n\nCreate a short and concise one-question university level assignment given the text, be creative. Give your answer in valid jsonl format: {assignment: <text>, task_1: <text>, task_2: <text>, ...} nothing else."