import re
import copy

import loader
import exporter
import evaluator
import prompt_writer
import chatting_llm_azure
import chatting_llm_huggingface


class ScientificGenerativeAgent():

    def __init__(self, config):
        self._config        = config
        self._prompt_writer = prompt_writer.PromptWriter(config)
        self._loader        = loader.Loader(             config)
        self._evaluator     = evaluator.Evaluator(       config)
        self._exporter      = exporter.Exporter(         config)
        match self._config["llm_platform"]:
            case "azure":
                self._llm = chatting_llm_azure.ChattingLLMAzure()
            case "huggingface":
                self._llm = chatting_llm_huggingface.ChattingLLMHuggingface()
            case _:
                raise ValueError("Invalid LLM platform.")

        self._iterations    = 5
        self._top_k         = 3
        self._top_k_models  = []


    def set_up(self):
        self._llm.set_up()
        self._exporter.set_up()
        self._loader.load()


    def run(self):
        system_prompt = self._prompt_writer.write_system_prompt()
        user_prompt   = self._prompt_writer.write_user_prompt(loader=self._loader)
        fit_code      = self._prompt_writer.write_fit_code()

        for iteration in range(self._iterations):
            def _recursive_generate_and_evaluate_model(attempts=0, max_attempts=10):
                try:
                    return self._generate_and_evaluate_model(
                        system_prompt, user_prompt, fit_code
                    )
                except Exception as e:
                    print(f"Repeating iteration {iteration} due to error: {e}")
                    if attempts < max_attempts:
                        return _recursive_generate_and_evaluate_model(attempts+1, max_attempts)
                    else:
                        print("Max attempts reached. Operation failed.")
                        return None
            model, model_code = _recursive_generate_and_evaluate_model()
            loss, loss_line   = self._evaluator.evaluate(iteration, self._loader, model)

            self._save_model(model, model_code, loss, loss_line)

        model, model_code, _, _ = self._load_best_model()
        self._exporter.export(
            loader     = self._loader,
            model      = model,
            model_code = model_code,
            prompts    = [system_prompt, user_prompt, fit_code],
            llm        = self._llm.get_model()
        )


    def _generate_and_evaluate_model(self, system_prompt, user_prompt, fit_code):
        # Outer-Level Optimization: Ask LLM for forward equation based on previous iterations
        previous = ""
        for idx, (_, top_k_code, _, top_k_loss_line) in enumerate(self._top_k_models):
            previous += (f"### Previous iteration #{            idx}:\n\n{top_k_code     }\n\n"
                         f"### Feedback on previous iteration #{idx}:\n\n{top_k_loss_line}\n\n")
        user_prompt = previous + user_prompt
        response = self._llm.chat(system_prompt, user_prompt)

        # Execution of proposed code
        model_code = re.findall(r"```python(.*?)```", response, re.DOTALL)[0].strip()
        namespace = {}
        exec(model_code + fit_code, namespace)
        model = namespace["Physics"]()

        # Inner-Level Optimization: Optimize parameters
        model.fit(x = self._loader.get_train_data_x(), y = self._loader.get_train_data_y())

        return model, model_code


    def _save_model(self, model, model_code, loss, loss_line):
        self._top_k_models.append([copy.deepcopy(model), model_code, loss, loss_line])
        # Sort models by loss
        self._top_k_models.sort(key=lambda x: x[2])
        # Keep only the top k models
        self._top_k_models = self._top_k_models[:self._top_k]


    def _load_best_model(self):
        return self._top_k_models[0]
