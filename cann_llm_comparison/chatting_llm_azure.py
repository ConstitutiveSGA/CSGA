"""
Module for interacting with Azure's OpenAI language model.

This module defines the ChattingLLMAzure class, which handles the setup and interaction
with Azure's OpenAI language model to generate chat responses.

Classes:
    ChattingLLMAzure: A class to interact with Azure's OpenAI language model for generating chat
                      responses.
"""

import os
import openai
import dotenv
import tiktoken


class ChattingLLMAzure():
    """
    A class to interact with Azure's OpenAI language model for generating chat responses.

    Attributes:
        _model_type (str):            The type of the language model to use.
        _max_output_tokens (int):     The maximum number of tokens for the output.
        _env (dict):                  Environment variables for Azure OpenAI.
        _tokenization (dict):         Tokenization settings for the language model.
        _client (openai.AzureOpenAI): The client for interacting with Azure OpenAI.
    """

    def __init__(self):
        """
        Initializes the ChattingLLMAzure.
        """
        self._model_type        = None
        self._max_output_tokens = 5000
        self._env               = None
        self._tokenization      = None
        self._client            = None


    def set_up(self):
        """
        Sets up the environment and tokenization settings for the language model.
        """
        self._env        = self._load_env()
        self._model_type = self._select_model_type()
        self._tokenization = {
                "context_length":     self._select_context_length(), 
                "encoding":           tiktoken.encoding_for_model(self._model_type),
                "tokens_per_message": 3,
                "tokens_per_name":    1
        }
        self._client = openai.AzureOpenAI(
            azure_endpoint   = self._env["azure_endpoint"],
            azure_deployment = self._env["deployment"],
            api_version      = self._env["api_version"],
            api_key          = self._env["api_key"],
        )


    def chat(self, messages):
        """
        Generates a chat response from the language model.

        Args:
            messages (list): A list of messages to send to the language model.

        Returns:
            tuple: A tuple containing the response and the finish reason.
        """
        self._check_chat_length(messages)
        response = self._generate_response(messages)
        response = self._clean_response_from_special_chars(response)
        return response


    def _load_env(self):
        dotenv.load_dotenv()
        return {
            "azure_endpoint": os.getenv("OPENAI_ENDPOINT"),
            "deployment":     os.getenv("OPENAI_DEPLOYMENT"),
            "api_key":        os.getenv("OPENAI_TOKEN"),
            "api_version":    os.getenv("OPENAI_API_VERSION"),
        }


    def _select_model_type(self):
        if self._env["deployment"] == "standard-gpt-4o":
            return "gpt-4o"
        if self._env["deployment"] == "standard-gpt-o1-preview":
            return "o1-preview"
        raise ValueError(f"Model type for deployment {self._env['deployment']} unset!")


    def _select_context_length(self):
        if self._model_type == "gpt-4o":
            return 128000
        if self._model_type == "o1-preview":
            return 128000
        raise ValueError(f"Context length for model {self._model_type} unset!")


    def _check_chat_length(self, messages):
        input_tokens = 0
        for message in messages:
            input_tokens += self._tokenization["tokens_per_message"]
            for key, value in message.items():
                input_tokens += len(self._tokenization["encoding"].encode(value))
                if key == "name":
                    input_tokens += self._tokenization["tokens_per_name"]
        input_tokens += 3  # every reply is primed with <|start|>assistant<|message|>

        if input_tokens + self._max_output_tokens > self._tokenization["context_length"]:
            raise ValueError("Chat is too long, would exceed the context window!")


    def _generate_response(self, messages):
        chat_completion_choices = 1
        if   self._model_type == "gpt-4o":
            response_structure = self._generate_4o_response(messages, chat_completion_choices)
        elif self._model_type == "o1-preview":
            response_structure = self._generate_o1_response(messages, chat_completion_choices)
        else:
            raise ValueError(f"Model type {self._model_type} not recognized!")

        response      = response_structure.choices[0].message.content
        finish_reason = response_structure.choices[0].finish_reason
        if finish_reason != "stop":
            raise ValueError(f"Call to LLM finished with reason: {finish_reason}")

        return response


    def _generate_4o_response(self, messages, chat_completion_choices):
        return self._client.chat.completions.create(
            model       = self._env["deployment"],
            messages    = messages,
            max_tokens  = self._max_output_tokens,
            n           = chat_completion_choices,
            temperature = 1.0
        )


    def _generate_o1_response(self, messages, chat_completion_choices):
        return self._client.chat.completions.create(
            model                 = self._env["deployment"],
            messages              = messages,
            max_completion_tokens = self._max_output_tokens,
            n                     = chat_completion_choices,
        )


    def _clean_response_from_special_chars(self, response):
        allowed_non_alnum_chars = [" ", ",", ".", ";", ":", "!", "%", "&", "?", "|", "-", "_", "/",
                                   '"', "'", "´", "`", "{", "}", "[", "]", "(", ")", "<", ">", "=",
                                   "+", "*", "#", "\n"]
        cleaned_response = []
        for char in response:
            if char.isalnum() or char in allowed_non_alnum_chars:
                cleaned_response.append(char)
        return "".join(cleaned_response)
