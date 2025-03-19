import os
import dotenv
import openai
import tiktoken


class ChattingLLMOpenAI():

    def __init__(self, platform):
        self._platform          = platform
        self._model             = "gpt-4o"
        self._temperature       = 0.6
        self._max_output_length = None
        self._env               = None
        self._tokenization      = None
        self._client            = None


    def set_up(self):
        self._tokenization = {
                "context_length":     self._select_context_length(), 
                "encoding":           tiktoken.get_encoding("cl100k_base"),
                "tokens_per_message": 3,
                "tokens_per_name":    1
        }
        self._max_output_length = self._select_max_output_length()

        match self._platform:
            case "azure":
                self._set_up_azure_client()
            case "openrouter":
                self._set_up_openrouter_client()
            case _:
                raise ValueError("Invalid LLM platform.")
        

    def _set_up_azure_client(self):
        self._env    = self._load_env()
        self._client = openai.AzureOpenAI(
            azure_endpoint = self._env["azure_endpoint"],
            api_version    = self._select_api_version(),
            api_key        = self._env["azure_key"],

        )


    def _set_up_openrouter_client(self):
        self._env    = self._load_env()
        self._client = openai.OpenAI(
            base_url = self._env["openrouter_endpoint"],
            api_key  = self._env["openrouter_key"],
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
            "azure_endpoint":      os.getenv("AZURE_ENDPOINT"),
            "azure_key":           os.getenv("AZURE_API_KEY"),
            "openrouter_endpoint": os.getenv("OPENROUTER_ENDPOINT"),
            "openrouter_key":      os.getenv("OPENROUTER_API_KEY"),
        }


    def _select_api_version(self):
        match self._model:
            case "gpt-35-turbo-16k"            | \
                 "gpt-4o"                      | \
                 "Meta-Llama-3.1-70B-Instruct" | \
                 "Meta-Llama-3.1-405B-Instruct":
                return "2024-05-01-preview"
            case "o1-preview" | \
                 "o1"         | \
                 "o3-mini":
                return "2024-12-01-preview"
            case _:
                raise ValueError(f"API version for model {self._model} unset!")


    def _select_context_length(self):
        match self._model:
            case "gpt-35-turbo-16k":
                return 16000
            case "gpt-4o"                       | \
                 "o1-preview"                   | \
                 "o1"                           | \
                 "Meta-Llama-3.1-70B-Instruct"  | \
                 "Meta-Llama-3.1-405B-Instruct" | \
                 "deepseek/deepseek-r1":
                return 128000
            case "deepseek/deepseek-r1-distill-qwen-32b":
                return 131072
            case "o3-mini":
                return 200000
            case _:
                raise ValueError(f"Context length for model {self._model} unset!")


    def _select_max_output_length(self):
        match self._model:
            case "gpt-35-turbo-16k"            | \
                 "gpt-4o"                      | \
                 "Meta-Llama-3.1-70B-Instruct" | \
                 "Meta-Llama-3.1-405B-Instruct":
                return 4096
            case "o1-preview"           | \
                 "o1"                   | \
                 "deepseek/deepseek-r1" | \
                 "deepseek/deepseek-r1-distill-qwen-32b":
                return 32768
            case "o3-mini":
                return 100000
            case _:
                raise ValueError(f"Max output length for model {self._model} unset!")


    def _check_chat_length(self, messages):
        match self._platform:
            case "azure":
                input_tokens = self._count_tokens_in_azure_format(messages)
            case "openrouter":
                input_tokens = self._count_tokens_in_openrouter_format(messages)
            case _:
                raise ValueError("Invalid LLM platform.")

        if input_tokens + self._max_output_length > self._tokenization["context_length"]:
            raise ValueError("Chat is too long, would exceed the context window!")


    def _count_tokens_in_azure_format(self, messages):
        input_tokens = 0
        
        for message in messages:
            input_tokens += self._tokenization["tokens_per_message"]
            input_tokens += len(self._tokenization["encoding"].encode(message["role"]))
            input_tokens += len(self._tokenization["encoding"].encode(message["content"][0]["type"]))
            input_tokens += len(self._tokenization["encoding"].encode(message["content"][0]["text"]))

        input_tokens += 3  # every reply is primed with <|start|>assistant<|message|>

        return input_tokens


    def _count_tokens_in_openrouter_format(self, messages):
        input_tokens = 0
        
        for message in messages:
            input_tokens += self._tokenization["tokens_per_message"]
            for value in message.values():
                input_tokens += len(self._tokenization["encoding"].encode(value))
        
        input_tokens += 3  # every reply is primed with <|start|>assistant<|message|>

        return input_tokens


    def _assemble_messages(self, system_message, user_message):
        match self._model:
            case "gpt-35-turbo-16k"             | \
                 "gpt-4o"                       | \
                 "Meta-Llama-3.1-70B-Instruct"  | \
                 "Meta-Llama-3.1-405B-Instruct":
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
            case "deepseek/deepseek-r1" | \
                 "deepseek/deepseek-r1-distill-qwen-32b":
                return [
                    {"role": "user", "content": f"## Scenario:\n{system_message}\n" + user_message},
                ]
            case _:
                raise ValueError(f"Assembly of messages for model {self._model} unset!")


    def _generate_response(self, messages):
        match self._model:
            case "gpt-35-turbo-16k"             | \
                 "gpt-4o"                       | \
                 "Meta-Llama-3.1-70B-Instruct"  | \
                 "Meta-Llama-3.1-405B-Instruct" | \
                 "deepseek/deepseek-r1"         | \
                 "deepseek/deepseek-r1-distill-qwen-32b":
                completion = self._client.chat.completions.create(
                    model       = self._model,
                    messages    = messages,
                    max_tokens  = self._max_output_length,
                    temperature = self._temperature,
                )
            case "o1-preview" | \
                 "o1":
                completion = self._client.chat.completions.create(
                    model                 = self._model,
                    messages              = messages,
                    max_completion_tokens = self._max_output_length,
                )
            case "o3-mini":
                completion = self._client.chat.completions.create(
                    model                 = self._model,
                    messages              = messages,
                    max_completion_tokens = self._max_output_length,
                    reasoning_effort      = "high",
                )
            case _:
                raise ValueError(f"Response for model {self._model} unset!")

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
