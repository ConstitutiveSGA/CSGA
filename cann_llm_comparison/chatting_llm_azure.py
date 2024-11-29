import os
import dotenv
import tiktoken
import azure.ai.inference
import azure.core.credentials


class ChattingLLMAzure():

    def __init__(self):
        self._temperature  = 1.0
        self._max_tokens   = 4000
        self._env          = None
        self._tokenization = None
        self._client       = None


    def set_up(self):
        self._env          = self._load_env()
        self._tokenization = {
                "context_length":     self._select_context_length(), 
                "encoding":           self._select_encoding(),
                "tokens_per_message": 3,
                "tokens_per_name":    1
        }
        self._client = azure.ai.inference.ChatCompletionsClient(
            endpoint   = self._env["endpoint"],
            credential = azure.core.credentials.AzureKeyCredential(self._env["key"]),
        )


    def chat(self, messages):
        self._check_chat_length(messages)
        response = self._generate_response(messages)
        response = self._clean_response_from_special_chars(response)
        return response


    def get_model(self):
        return self._env["model"]


    def _load_env(self):
        dotenv.load_dotenv()
        return {
            "model":       os.getenv("AZURE_MODEL"),
            "endpoint":    os.getenv("AZURE_ENDPOINT"),
            "key":         os.getenv("AZURE_API_KEY"),
            "api_version": os.getenv("AZURE_API_VERSION"),
        }


    def _select_encoding(self):
        match self._env["model"]:
            case "gpt-35-turbo-16k" | \
                 "gpt-4o"           | \
                 "o1-preview":
                return tiktoken.encoding_for_model(self._env["model"])
            case "Meta-Llama-3.1-70B-Instruct" | \
                 "Meta-Llama-3.1-405B-Instruct":
                return tiktoken.get_encoding("cl100k_base")
            case _:
                raise ValueError(f"Encoding for model {self._env['model']} unset!")


    def _select_context_length(self):
        match self._env["model"]:
            case "gpt-35-turbo-16k":
                return 16000
            case "gpt-4o"                      | \
                 "o1-preview"                  | \
                 "Meta-Llama-3.1-70B-Instruct" | \
                 "Meta-Llama-3.1-405B-Instruct":
                return 128000
            case _:
                raise ValueError(f"Context length for model {self._env['model']} unset!")


    def _check_chat_length(self, messages):
        input_tokens = 0
        for message in messages:
            input_tokens += self._tokenization["tokens_per_message"]
            for key, value in message.items():
                input_tokens += len(self._tokenization["encoding"].encode(value))
                if key == "name":
                    input_tokens += self._tokenization["tokens_per_name"]
        input_tokens += 3  # every reply is primed with <|start|>assistant<|message|>

        if input_tokens + self._max_tokens > self._tokenization["context_length"]:
            raise ValueError("Chat is too long, would exceed the context window!")


    def _generate_response(self, messages):
        match self._env["model"]:
            case "gpt-35-turbo-16k"            | \
                 "gpt-4o"                      | \
                 "Meta-Llama-3.1-70B-Instruct" | \
                 "Meta-Llama-3.1-405B-Instruct":
                response_struct = self._client.complete(
                    messages    = messages,
                    temperature = self._temperature,
                    max_tokens  = self._max_tokens,
                    model       = self._env["model"],
                )
            case "o1-preview":
                response_struct = self._client.complete(
                    messages    = messages,
                    model       = self._env["model"],
                )
            case _:
                raise ValueError(f"Response for model {self._env['model']} unset!")

        response      = response_struct.choices[0].message.content
        finish_reason = response_struct.choices[0].finish_reason
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
