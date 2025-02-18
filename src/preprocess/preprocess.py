import json
from pprint import pprint


def process_generations(chats):
    chat_data = {"assignment": [], "conversation": [], "model": "Qwen/Qwen2.5-72B-Instruct"}
    for chat in chats:
        content = json.loads(chat["content"])
        name = chat["name"]
        keys = list(content.keys())

        if content.get("assignment"):
            chat_data["assignment"].append(content)
            continue
        try:
            if name == "Instructor":
                keys.remove("answer")
                role = "assistant"
                chat_data["conversation"].append({
                "role": role,
                "content": content["answer"],
                "feedback": [{f'feedback_{key[-1]}': content[key]} for key in keys]
            })
            else:
                keys.remove("answer")
                role = "user"
                chat_data["conversation"].append({
                "role": role,
                "content": content["answer"],
                "error": [{f'error_{key[-1]}': content[key]} for key in keys]

            })
        except:
            pass


    errors = [list(error.values())[0] for error in chat_data["conversation"][0]["error"]]
    feedback = [list(feedback.values())[0] for feedback in chat_data["conversation"][1]["feedback"]]

    if len(errors) == len(feedback) and all(e and f for e, f in zip(errors, feedback)):
        return chat_data


with open("_generation_output/out_2024_11_01_13:57:21_qwen.json") as f_in, open("_processed_output/output_qwen.json", "w") as f_out:
    cnt = 0
    for doc in f_in:
        if not cnt > 5000:
            try:
                conversation = json.loads(doc)
                filtered_conversation = conversation["conversation"][1:]
                chat_data = process_generations(filtered_conversation)

                if chat_data:
                    f_out.write(json.dumps(chat_data))
                    f_out.write("\n")
            
            except:
                pass

            cnt += 1
            