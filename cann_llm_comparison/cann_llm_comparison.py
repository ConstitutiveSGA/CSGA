import os

import chatting_llm_azure


def main():
    config = set_config()
    llm = chatting_llm_azure.ChattingLLMAzure()
    llm.set_up()
    messages = [
        {"role":    "system",
         "content": "You are a helpful assistant."},
        {"role":    "user",
         "content": "Hello"}
    ]
    response, finish_reason = llm.chat(messages)
    pass


def set_config():
    return {
        "input_dir":  os.path.join("..", "..", "input"),
        "data_file":  "",
        "output_dir": os.path.join("..", "..", "output"),
    }


if __name__ == "__main__":
    main()
