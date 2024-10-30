import os
import re
import torch
import numpy
import pandas
import datetime
import sklearn.model_selection

import chatting_llm_azure


class ScientificGenerativeAgent():

    def __init__(self, config):
        self._config          = config
        self._full_output_dir = None
        self._llm             = chatting_llm_azure.ChattingLLMAzure()
        self._train_data_x    = None
        self._train_data_y    = None
        self._test_data_x     = None
        self._test_data_y     = None
        self._top_k           = 3
        self._top_k_models    = []
        

    def set_up(self):
        self._create_output_dir()
        self._llm.set_up()
        self._load_data()   


    def run(self):
        system_prompt = self._write_system_prompt()
        user_prompt   = self._write_user_prompt()
        fit_code      = self._write_fit_code()
        
        iterations = 25
        for iteration in range(iterations):
            # Generate and evaluate model
            def _recursive_generate_and_evaluate_model(attempts=0, max_attempts=3):
                try:
                    return self._generate_and_evaluate_model(
                        system_prompt, user_prompt, fit_code
                    )
                except Exception as _:
                    print(f"Repeating iteration {iteration} due to error!")
                    if attempts < max_attempts:
                        return _recursive_generate_and_evaluate_model(attempts+1, max_attempts)
                    else:
                        print("Max attempts reached. Operation failed.")
                        return None
            model_code, train_predictions, train_loss = _recursive_generate_and_evaluate_model()
            
            # Save model
            self._save_model(model_code, train_loss)

            # Export model
            self._export_model(iteration, model_code, train_predictions, train_loss)


    def _generate_and_evaluate_model(self, system_prompt, user_prompt, fit_code):
        # Outer-Level Optimization: Ask LLM for forward equation
        extension = ""
        for idx, (top_k_model_code, top_k_model_loss) in enumerate(self._top_k_models):
            extension += f"### Previous iteration #{idx}:\n\n{                   top_k_model_code}\n\n"
            extension += f"### Feedback on previous iteration #{idx}:\n\nLoss = {top_k_model_loss}\n\n"
        user_prompt = extension + user_prompt
        messages = [
            {"role":"system", "content":system_prompt}, {"role":"user", "content":user_prompt}
        ]    
        response, _ = self._llm.chat(messages)
                
        # Execution of proposed code
        model_code = re.findall(r"```python(.*?)```", response, re.DOTALL)[0].strip()
        namespace = {}
        exec(model_code + fit_code, namespace)
        model = namespace["Physics"]()

        # Inner-Level Optimization: Optimize parameters
        model.fit(
            x = torch.tensor(self._train_data_x.values, dtype=torch.float32), 
            y = torch.tensor(self._train_data_y.values, dtype=torch.float32)
        )

        # Evaluate model
        train_predictions = pandas.Series(model.forward(
            cauchy_green_deformation = torch.tensor(self._train_data_x.values, dtype=torch.float32)
        ).detach().numpy())
        train_loss = numpy.mean((self._train_data_y.values - train_predictions) ** 2)

        return model_code, train_predictions, train_loss


    def _load_data(self):
        path = os.path.join(self._config["input_dir"], self._config["data_file"])
        data = pandas.read_csv(
            filepath_or_buffer = path,
            delimiter          = ",",
            header             = 0,
            index_col          = False,
        )
        train_data, test_data = sklearn.model_selection.train_test_split(
            data,
            test_size    = 0.2,
            random_state = 42
        )
        self._train_data_x = train_data["strains"].reset_index( drop=True)
        self._train_data_y = train_data["stresses"].reset_index(drop=True)
        self._test_data_x  = test_data["strains"].reset_index(  drop=True)
        self._test_data_y  = test_data["stresses"].reset_index( drop=True)


    def _create_output_dir(self):
        datetime_dir = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        self._full_output_dir = os.path.join(os.getcwd(), self._config["output_dir"], datetime_dir)
        os.mkdir(self._full_output_dir)
        

    def _save_model(self, model_code, train_loss):
        self._top_k_models.append((model_code, train_loss))
        # Sort models by loss
        self._top_k_models.sort(key=lambda x: x[1])
        # Keep only the top k models
        self._top_k_models = self._top_k_models[:self._top_k]


    def _export_model(self, iteration, model_code, train_predictions, train_loss):
        path = os.path.join(self._full_output_dir, f"iteration_{iteration}_model_code.txt")
        with open(path, mode="w", encoding="utf-8", errors="ignore") as file:
            file.write(model_code)

        path = os.path.join(self._full_output_dir, f"iteration_{iteration}_predictions.csv")
        pandas.DataFrame({
            "train_data_x":      self._train_data_x,
            "train_data_y":      self._train_data_y,
            "train_predictions": train_predictions
        }).to_csv(path, index=False)

        path = os.path.join(self._full_output_dir, f"iteration_{iteration}_loss.txt")
        with open(path, "w") as file:
            file.write(str(train_loss))


    def _write_system_prompt(self):
        return '''
You are an intelligent AI assistant for coding, physical simulation, and scientific discovery. Follow the user’s requirements carefully and make sure you understand them.
Your expertise is strictly limited to physical simulation, material science, mathematics, and coding. Keep your answers short and to the point.
Do not provide any information that is not requested. Always document your code as comments to explain the reason behind them. Use Markdown to format your solution.
You are very familiar with Python and PyTorch. Do not use any external libraries other than the libraries used in the examples.
'''

    def _write_user_prompt(self):
        return '''
### Code Requirements

1. The programming language is always python. 
2. The only library allowed is PyTorch. Follow the examples provided by the user and check the PyTorch documentation to learn how to use PyTorch.
3. Separate the code into continuous physical parameters that can be tuned with differentiable optimization and the symbolic constitutive law represented by PyTorch code. Define them respectively in the ‘__init__‘ function and the ‘forward‘ function. Keep the continuous physical parameters in the list ‘self.params‘.
4. The input and output of the ‘forward‘ function (cauchy_green_deformation and kirchhoff_stress) are batches of scalar values, stored in torch tensors to allow parameter optimization. Do not try to perform matrix or tensor operations like ‘trace‘ on them. Treat them as what they are - stacked scalars!
5. The proposed code should strictly follow the structure and function signatures below:

```python
import torch


class Physics(torch.nn.Module):

    def __init__(self):
        """
        Define trainable continuous physical parameters for differentiable optimization.
        Tentatively initialize the parameters with default values.
        """
        super().__init__()
        
        self.params = [
            """
            Define the physical parameters as torch.nn.Parameter objects and strictly keep them in 
            the list ‘self.params‘. Do not unpack this list. Define as many parameters as list 
            elements as necessary. For each element, replace [float] with the desired default value.
            
            torch.nn.Parameter(torch.tensor([float], requires_grad=True))
            ...
            """
        ]      
       

    def forward(self, cauchy_green_deformation) -> torch.tensor:
        """
        Compute Second Piola-Kirchhoff Stress Values from Right Cauchy-Green Deformation Values. 

        Args:
            x (torch.tensor): Right Cauchy-Green Deformation Values. Scalars stored in 1D torch tensor.

        Returns:
            torch.tensor: Second Piola-Kirchhoff Stress Values. Scalars stored in 1D torch tensor.
        """
        return kirchhoff_stress
```

### Solution Requirements

1. Analyze step-by-step what the potential problem is in the previous iterations based on the feedback. Think about why the results from previous constitutive laws mismatched with the ground truth. Do not give advice about how to optimize. Focus on the formulation of the constitutive law. Start this section with "### Analysis". Analyze all iterations individually, and start the subsection for each iteration with "#### Iteration N", where N stands for the index. Remember to analyze every iteration in the history. Be creative! Do not try any approach you find in a previous iteration a second time.
2. Think step-by-step what you need to do in this iteration. Think about how to separate your algorithm into a continuous physical parameter part and a symbolic constitutive law part. Describe your plan in pseudo-code, written out in great detail. Remember to update the default values of the trainable physical parameters based on previous optimizations. Start this section with "### Step-by-Step Plan".
3. Output the code in a single code block "‘‘‘python ... ‘‘‘" with detailed comments in the code block. Do not add any trailing comments before or after the code block. Start this section with "### Code".
'''

    def _write_fit_code(self):
        return '''
    def fit(self, x, y, epochs=50, lr=0.01):
        """
        Trains the regression model.

        Args:
            x (torch.tensor): Input tensor.
            y (torch.tensor): Target tensor.
            epochs (int, optional): Number of epochs to train. Default is 1000.
            lr (float, optional): Learning rate for the optimizer. Default is 0.01.

        This method trains the model using Stochastic Gradient Descent (SGD) and prints the loss
        every 100 epochs.
        """
        optimizer = torch.optim.SGD(self.params, lr=lr)
        
        for epoch in range(epochs):
            optimizer.zero_grad()
            y_pred = self.forward(x)
            loss   = torch.nn.MSELoss()(y_pred, y)
            loss.backward()
            optimizer.step()
            
            if (epoch+1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
'''
