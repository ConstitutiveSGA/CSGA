import os
import dotenv
import openai
import tiktoken


class ChattingLLMAzure():

    def __init__(self):
        self._model        = "o3-mini"
        self._temperature  = 1.0
        self._env          = None
        self._tokenization = None
        self._client       = None


    def set_up(self):
        self._env          = self._load_env()
        self._tokenization = {
                "context_length":     self._select_context_length(), 
                "encoding":           tiktoken.get_encoding("cl100k_base"),
                "tokens_per_message": 3,
                "tokens_per_name":    1
        }
        self._client = openai.AzureOpenAI(
            azure_endpoint = self._env["endpoint"],
            api_version    = self._select_api_version(),
            api_key        = self._env["key"],

        )


    def chat(self, system_message, user_message):
        messages = self._assemble_messages(system_message, user_message)
        self._check_chat_length(messages)
        response = self._generate_response(messages)
        response = self._clean_response_from_special_chars(response)
        return response


    def get_model(self):
        return self._model


    def _load_env(self):
        dotenv.load_dotenv()
        return {
            "endpoint": os.getenv("AZURE_ENDPOINT"),
            "key":      os.getenv("AZURE_API_KEY"),
        }


    def _select_api_version(self):
        match self._model:
            case "gpt-35-turbo-16k"             | \
                 "gpt-4o"                       | \
                 "Meta-Llama-3.1-70B-Instruct"  | \
                 "Meta-Llama-3.1-405B-Instruct" | \
                 "DeepSeek-R1":
                return "2024-05-01-preview"
            case "o1-preview" | \
                 "o1"         | \
                 "o3-mini":
                return "2024-12-01-preview"
            case _:
                raise ValueError(f"API version for model {self._env['model']} unset!")


    def _select_context_length(self):
        match self._model:
            case "gpt-35-turbo-16k":
                return 16000
            case "gpt-4o"                       | \
                 "o1-preview"                   | \
                 "o1"                           | \
                 "Meta-Llama-3.1-70B-Instruct"  | \
                 "Meta-Llama-3.1-405B-Instruct" | \
                 "DeepSeek-R1":
                return 128000
            case "o3-mini":
                return 200000
            case _:
                raise ValueError(f"Context length for model {self._env['model']} unset!")


    def _check_chat_length(self, messages):
        input_tokens = 0
        for message in messages:
            input_tokens += self._tokenization["tokens_per_message"]
            input_tokens += len(self._tokenization["encoding"].encode(message["role"]))
            input_tokens += len(self._tokenization["encoding"].encode(message["content"][0]["type"]))
            input_tokens += len(self._tokenization["encoding"].encode(message["content"][0]["text"]))

        input_tokens += 3  # every reply is primed with <|start|>assistant<|message|>

        if input_tokens + 4000 > self._tokenization["context_length"]:
            raise ValueError("Chat is too long, would exceed the context window!")


    def _assemble_messages(self, system_message, user_message):
        match self._model:
            case "gpt-35-turbo-16k"             | \
                 "gpt-4o"                       | \
                 "Meta-Llama-3.1-70B-Instruct"  | \
                 "Meta-Llama-3.1-405B-Instruct" | \
                 "DeepSeek-R1":
                return [
                    {"role": "system", "content": [{"type": "text", "text": system_message}]},
                    {"role": "user",   "content": [{"type": "text", "text": user_message}]},
                ]
            case "o1-preview" | \
                 "o1":
                return [
                    {"role": "user",   "content": [{"type": "text", "text": f"## Scenario:\n{system_message}\n" + user_message}]},
                ]
            case "o3-mini":
                return [
                    {"role": "developer", "content": [{"type": "text", "text": system_message}]},
                    {"role": "user",      "content": [{"type": "text", "text": user_message}]},
                ]
            case _:
                raise ValueError(f"Assembly of messages for model {self._env['model']} unset!")


    def _generate_response(self, messages):
        match self._model:
            case "gpt-35-turbo-16k"             | \
                 "gpt-4o"                       | \
                 "Meta-Llama-3.1-70B-Instruct"  | \
                 "Meta-Llama-3.1-405B-Instruct" | \
                 "DeepSeek-R1":
                completion = self._client.chat.completions.create(
                    model       = self._model,
                    messages    = messages,
                    max_tokens  = 4096,
                    temperature = self._temperature,
                )
            case "o1-preview" | \
                 "o1":
                completion = self._client.chat.completions.create(
                    model                 = self._model,
                    messages              = messages,
                    max_completion_tokens = 32768,
                )
            case "o3-mini":
                completion = self._client.chat.completions.create(
                    model                 = self._model,
                    messages              = messages,
                    max_completion_tokens = 100000,
                    reasoning_effort      = "high",
                )
            case _:
                raise ValueError(f"Response for model {self._env['model']} unset!")

        response      = completion.choices[0].message.content
        finish_reason = completion.choices[0].finish_reason
        if finish_reason != "stop":
            raise ValueError(f"Call to LLM finished with reason: {finish_reason}")

        return response


    def _clean_response_from_special_chars(self, response):
        allowed_non_alnum_chars = [" ", ",", ".", ";", ":", "!", "%", "&", "?", "|", "-", "_", "/",
                                   '"', "'", "´", "`", "{", "}", "[", "]", "(", ")", "<", ">", "=",
                                   "+", "*", "#", "\n"]
        cleaned_response = []
        for char in response:
            if char.isalnum() or char in allowed_non_alnum_chars:
                cleaned_response.append(char)
        return "".join(cleaned_response)
