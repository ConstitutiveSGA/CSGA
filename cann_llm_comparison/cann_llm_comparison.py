import os
import torch
import pandas
import matplotlib.pyplot
import sklearn.model_selection

import chatting_llm_azure
import scientific_generative_agent


def main():
    config = _set_config()

    pre_response= """
import torch
"""
    response = """
class RegressionModel(torch.nn.Module):
    def __init__(self):
        super(RegressionModel, self).__init__()
        
        self.params = [
            torch.nn.Parameter(torch.randn(1, requires_grad=True)),
            torch.nn.Parameter(torch.randn(1, requires_grad=True)),
        ]        
        self.loss_fn = torch.nn.MSELoss()


    def forward(self, x):
        a, b = self.params
        return a * x + b
"""
    post_response = """
    def fit(self, x, y, epochs=1000, lr=0.01):
        optimizer = torch.optim.SGD(self.params, lr=lr)
        
        for epoch in range(epochs):
            optimizer.zero_grad()
            y_pred = self.forward(x)
            loss   = self.loss_fn(y_pred, y)
            loss.backward()
            optimizer.step()
            
            if (epoch+1) % 100 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
"""
    namespace = {}
    exec(pre_response + response + post_response, namespace)
    RegressionModel = namespace['RegressionModel']
    
    data   = _load_data(config)
    train_data, test_data = sklearn.model_selection.train_test_split(data, test_size=0.2, random_state=42)
    
    model  = RegressionModel()
    model.fit(
        x = torch.tensor(train_data["strains"].values,  dtype=torch.float32), 
        y = torch.tensor(train_data["stresses"].values, dtype=torch.float32)
    )
    train_preds = pandas.Series(model.forward(
        x = torch.tensor(train_data["strains"].values, dtype=torch.float32)
    ).detach().numpy())
    
    matplotlib.pyplot.scatter(
        x     = train_data["strains"], 
        y     = train_data["stresses"], 
        color = "blue", 
        label = "Training data",
    )
    matplotlib.pyplot.scatter(
        x     = train_data["strains"], 
        y     = train_preds,
        color = "red",
        label = "Predictions"
    )
    matplotlib.pyplot.title("Regression Model Predictions")
    matplotlib.pyplot.xlabel("Strain")
    matplotlib.pyplot.ylabel("Stress")
    matplotlib.pyplot.legend()
    matplotlib.pyplot.show()

    
    
    
    # llm = chatting_llm_azure.ChattingLLMAzure()
    # llm.set_up()
    # messages = [
    #     {"role":    "system",
    #      "content": "You are a helpful assistant."},
    #     {"role":    "user",
    #      "content": "Hello"}
    # ]
    # response, finish_reason = llm.chat(messages)


def _set_config():
    return {
        "input_dir":  os.path.join("..", "input"),
        "data_file":  "treloar_uniaxial_tension.csv",
        "output_dir": os.path.join("..", "output"),
    }


def _load_data(config):
    return pandas.read_csv(
        filepath_or_buffer = os.path.join(config["input_dir"], config["data_file"]),
        delimiter          = ",",
        header             = 0,
        index_col          = False,
    )


if __name__ == "__main__":
    main()
