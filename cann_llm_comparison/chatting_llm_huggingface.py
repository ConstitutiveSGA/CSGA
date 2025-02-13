import os
import torch
import transformers


class ChattingLLMHuggingface():

    def __init__(self):
        self._model_name     = "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
        self._context_length = None
        self._temperature    = 0.6
        self._cache_dir      = os.path.join("..", "..", "cache")
        self._device         = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model          = None
        self._tokenizer      = None


    def set_up(self):
        self._context_length = self._select_context_length()

        self._tokenizer = transformers.AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path = self._model_name,
            cache_dir                     = self._cache_dir,
        )

        self._model = transformers.AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path = self._model_name,
            cache_dir                     = self._cache_dir,
            use_safetensors               = True,
            quantization_config           = transformers.BitsAndBytesConfig(
                load_in_4bit              = True,
                bnb_4bit_use_double_quant = True,
                bnb_4bit_quant_type       = "nf4",
                bnb_4bit_compute_dtype    = torch.bfloat16,
            )
        ).to(self._device)


    def chat(self, system_message, user_message):
        messages = self._assemble_messages(system_message, user_message)
        messages = self._tokenize_messages(messages)
        self._check_chat_length(messages)
        response = self._generate_response(messages)
        response = self._clean_response_from_special_chars(response)
        return response


    def get_model(self):
        return self._model


    def _select_context_length(self):
        match self._model_name:
            case "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B":
                return 131072 
            case _:
                raise ValueError(f"Context length for model {self._model_name} unset!") 


    def _assemble_messages(self, system_message, user_message):
        match self._model_name:
            case "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B":
                return [
                    {"role": "user", "content": f"## Scenario:\n{system_message}\n" + user_message},
                ]
            case _:
                raise ValueError(f"Assembly of messages for model {self._model_name} unset!") 


    def _tokenize_messages(self, messages):
        return self._tokenizer.apply_chat_template(
            conversation          = messages,
            add_generation_prompt = True,
            tokenize              = True,
            padding               = True,
            return_tensors        = "pt",
            return_dict           = True,
        ).to(self._device)


    def _check_chat_length(self, messages):
        if messages.data["input_ids"].shape[1] + 32768 > self._context_length:
            raise ValueError("Chat is too long, would exceed the context window!")


    def _generate_response(self, messages):
        response = self._model.generate(
            input_ids      = messages.data["input_ids"],
            attention_mask = messages.data["attention_mask"],
            max_new_tokens = 32768,
            pad_token_id   = self._tokenizer.pad_token_id
        )
        return self._tokenizer.decode(response[0])


    def _clean_response_from_special_chars(self, response):
        allowed_non_alnum_chars = [" ", ",", ".", ";", ":", "!", "%", "&", "?", "|", "-", "_", "/",
                                   '"', "'", "´", "`", "{", "}", "[", "]", "(", ")", "<", ">", "=",
                                   "+", "*", "#", "\n"]
        cleaned_response = []
        for char in response:
            if char.isalnum() or char in allowed_non_alnum_chars:
                cleaned_response.append(char)
        return "".join(cleaned_response)
