import os
import datetime
import torch
import pandas
import torcheval.metrics
import matplotlib.pyplot

class Exporter():

    def __init__(self, config):
        self._config     = config
        self._output_dir = None


    def set_up(self):
        self._create_output_dir()


    def export(self, loader, model, model_code, prompts, llm):
        match self._config["problem"]:
            case "synthetic_a":
                self._export_synthetic_a(loader, model, model_code, prompts, llm)
            case "synthetic_b":
                self._export_synthetic_b(loader, model, model_code, prompts, llm)
            case "brain":
                self._export_brain(loader, model, model_code, prompts, llm)
            case _:
                raise ValueError("Invalid problem type.")


    def _export_synthetic_a(self, loader, model, model_code, prompts, llm):
        self._export_problem()
        self._export_model(
            model      = model,
            model_code = model_code
        )
        self._export_prompts(prompts=prompts)
        self._export_llm(llm=llm)
        self._export(
            loader         = loader,
            key            = "uni-x",
            model          = model,
            prepare_data   = self._prepare_synthetic_data_for_export,
            strain_slc_idx = [0,0], # extract epsilon_1_1
            stress_slc_idx = [0,0], # extract   sigma_1_1
            xlabel         = "Stretch [F_11]",
            ylabel         = "Stress [P_11]",
            inverted_x     = False,
            inverted_y     = False,
            title          = "Synthetic Data [A] Uniaxial Tension"
        )
        self._export(
            loader         = loader,
            key            = "equi",
            model          = model,
            prepare_data   = self._prepare_synthetic_data_for_export,
            strain_slc_idx = [0,0], # extract epsilon_1_1
            stress_slc_idx = [0,0], # extract   sigma_1_1
            xlabel         = "Stretch [F_11]",
            ylabel         = "Stress [P_11]",
            inverted_x     = False,
            inverted_y     = False,
            title          = "Synthetic Data [A] Equibiaxial Tension"
        )
        self._export(
            loader         = loader,
            key            = "strip-x",
            model          = model,
            prepare_data   = self._prepare_synthetic_data_for_export,
            strain_slc_idx = [0,0], # extract epsilon_1_1
            stress_slc_idx = [0,0], # extract   sigma_1_1
            xlabel         = "Stretch [F_11]",
            ylabel         = "Stress [P_11]",
            inverted_x     = False,
            inverted_y     = False,
            title          = "Synthetic Data [A] Strip-biaxial Tension"
        )


    def _export_synthetic_b(self, loader, model, model_code, prompts, llm):
        self._export_problem()
        self._export_model(
            model      = model,
            model_code = model_code
        )
        self._export_prompts(prompts=prompts)
        self._export_llm(llm=llm)
        self._export(
            loader         = loader,
            key            = "uni-x",
            model          = model,
            prepare_data   = self._prepare_synthetic_data_for_export,
            strain_slc_idx = [0,0], # extract epsilon_1_1
            stress_slc_idx = [0,0], # extract   sigma_1_1
            xlabel         = "Stretch [RCG_11]",
            ylabel         = "Stress [S_11]",
            inverted_x     = False,
            inverted_y     = False,
            title          = "Synthetic Data [B] Uniaxial Tension"
        )
        self._export(
            loader         = loader,
            key            = "equi",
            model          = model,
            prepare_data   = self._prepare_synthetic_data_for_export,
            strain_slc_idx = [0,0], # extract epsilon_1_1
            stress_slc_idx = [0,0], # extract   sigma_1_1
            xlabel         = "Stretch [RCG_11]",
            ylabel         = "Stress [S_11]",
            inverted_x     = False,
            inverted_y     = False,
            title          = "Synthetic Data [B] Equibiaxial Tension"
        )
        self._export(
            loader         = loader,
            key            = "strip-x",
            model          = model,
            prepare_data   = self._prepare_synthetic_data_for_export,
            strain_slc_idx = [0,0], # extract epsilon_1_1
            stress_slc_idx = [0,0], # extract   sigma_1_1
            xlabel         = "Stretch [RCG_11]",
            ylabel         = "Stress [S_11]",
            inverted_x     = False,
            inverted_y     = False,
            title          = "Synthetic Data [B] Strip-biaxial Tension"
        )


    def _prepare_synthetic_data_for_export(self, loader, key, model, strain_slc_idx, stress_slc_idx):
        train_data_x      = loader.get_train_data_x()[key]
        test_data_x       = loader.get_test_data_x( )[key]
        train_data_y      = loader.get_train_data_y()[key]
        test_data_y       = loader.get_test_data_y( )[key]
        train_predictions = model.forward(train_data_x).detach()
        test_predictions  = model.forward(test_data_x ).detach()

        train_data_x      = train_data_x[     :, strain_slc_idx[0], strain_slc_idx[1]]
        test_data_x       =  test_data_x[     :, strain_slc_idx[0], strain_slc_idx[1]]
        train_data_y      = train_data_y[     :, stress_slc_idx[0], stress_slc_idx[1]]
        test_data_y       =  test_data_y[     :, stress_slc_idx[0], stress_slc_idx[1]]
        train_predictions = train_predictions[:, stress_slc_idx[0], stress_slc_idx[1]]
        test_predictions  =  test_predictions[:, stress_slc_idx[0], stress_slc_idx[1]]

        return train_data_x, test_data_x, train_data_y, test_data_y, train_predictions, test_predictions


    def _export_brain(self, loader, model, model_code, prompts, llm):
        self._export_problem()
        self._export_model(
            model      = model,
            model_code = model_code
        )
        self._export_prompts(prompts=prompts)
        self._export_llm(llm=llm)
        self._export(
            loader         = loader,
            key            = "tens",
            model          = model,
            prepare_data   = self._prepare_brain_data_for_export,
            strain_slc_idx = [0,0], # extract epsilon_1_1
            stress_slc_idx = [0,0], # extract   sigma_1_1
            xlabel         = "Stretch [F_11]",
            ylabel         = "Stress [P_11]",
            inverted_x     = False,
            inverted_y     = False,
            title          = "Brain Data Tension"
        )
        self._export(
            loader         = loader,
            key            = "comp",
            model          = model,
            prepare_data   = self._prepare_brain_data_for_export,
            strain_slc_idx = [0,0], # extract epsilon_1_1
            stress_slc_idx = [0,0], # extract   sigma_1_1
            xlabel         = "Stretch [F_11]",
            ylabel         = "Stress [P_11]",
            inverted_x     = True,
            inverted_y     = True,
            title          = "Brain Data Compression"
        )
        self._export(
            loader         = loader,
            key            = "simple_shear",
            model          = model,
            prepare_data   = self._prepare_brain_data_for_export,
            strain_slc_idx = [0,1], # extract epsilon_1_2
            stress_slc_idx = [0,1], # extract   sigma_1_2
            xlabel         = "Stretch [F_12]",
            ylabel         = "Stress [P_12]",
            inverted_x     = False,
            inverted_y     = False,
            title          = "Brain Data Shear"
        )


    def _prepare_brain_data_for_export(self, loader, key, model, strain_slc_idx, stress_slc_idx):
        train_data_x      = loader.get_train_data_x()[key]
        test_data_x       = loader.get_test_data_x( )[key]
        train_data_y      = loader.get_train_data_y()[key]
        test_data_y       = loader.get_test_data_y( )[key]
        train_predictions = model.forward(train_data_x).detach()
        test_predictions  = model.forward(test_data_x ).detach()

        train_data_x      = train_data_x[     :, strain_slc_idx[0], strain_slc_idx[1]]
        test_data_x       =  test_data_x[     :, strain_slc_idx[0], strain_slc_idx[1]]
        train_data_y      = train_data_y.squeeze(1)
        test_data_y       =  test_data_y.squeeze(1)
        train_predictions = train_predictions[:, stress_slc_idx[0], stress_slc_idx[1]]
        test_predictions  =  test_predictions[:, stress_slc_idx[0], stress_slc_idx[1]]

        return train_data_x, test_data_x, train_data_y, test_data_y, train_predictions, test_predictions


    def _export(self, loader, key, model, prepare_data, strain_slc_idx, stress_slc_idx,
                xlabel, ylabel, inverted_x, inverted_y, title):
        (
            train_data_x,
             test_data_x,
            train_data_y,
             test_data_y,
            train_predictions,
             test_predictions
        ) = prepare_data(
            loader         = loader,
            key            = key,
            model          = model,
            strain_slc_idx = strain_slc_idx,
            stress_slc_idx = stress_slc_idx
        )

        train_mse = torch.nn.MSELoss()(train_predictions, train_data_y).item()
        test_mse  = torch.nn.MSELoss()( test_predictions,  test_data_y).item()
        r2 = torcheval.metrics.functional.r2_score(
            input  = torch.cat((train_predictions, test_predictions)),
            target = torch.cat((train_data_y,      test_data_y)),
        ).item()

        self._export_loss(
            file      = "loss",
            key       = key.replace("-", "_").title(),
            train_mse = train_mse,
            test_mse  = test_mse,
            r2        = r2,
        )
        self._export_data(
            file        = "training_data",
            key         = key.replace("-", "_").title(),
            data_x      = train_data_x,
            data_y      = train_data_y,
            predictions = train_predictions
        )
        self._export_data(
            file        = "test_data",
            key         = key.replace("-", "_").title(),
            data_x      = test_data_x,
            data_y      = test_data_y,
            predictions = test_predictions
        )

        self._plot_predictions(
            file              = "predictions",
            key               = key.replace("-", "_").title(),
            train_data_x      = train_data_x,
            train_data_y      = train_data_y,
            train_predictions = train_predictions,
            test_data_x       = test_data_x,
            test_data_y       = test_data_y,
            test_predictions  = test_predictions,
            r2                = r2,
            xlabel            = xlabel,
            ylabel            = ylabel,
            inverted_x        = inverted_x,
            inverted_y        = inverted_y,
            title             = title
        )


    def _export_problem(self):
        path = os.path.join(self._output_dir, "problem.txt")
        with open(path, mode="w", encoding="utf-8") as file:
            file.write(f"Problem: {self._config['problem'].replace('_', ' ').title()}")


    def _export_model(self, model, model_code):
        path = os.path.join(self._output_dir, "model_code.txt")
        with open(path, mode="w", encoding="utf-8") as file:
            file.write(model_code)

        path = os.path.join(self._output_dir, "model_parameters.txt")
        with open(path, mode="w", encoding="utf-8") as file:
            for index, param in enumerate(model.params):
                file.write(f"Parameter {index}: {param.item():.8f}\n")


    def _export_prompts(self, prompts):
        path = os.path.join(self._output_dir, "prompts.txt")
        with open(path, mode="w", encoding="utf-8") as file:
            [file.write(f"{prompt}\n\n") for prompt in prompts]
    
    
    def _export_llm(self, llm):
        path = os.path.join(self._output_dir, "llm.txt")
        with open(path, mode="w", encoding="utf-8") as file:
            file.write(llm)


    def _export_loss(self, file, key, train_mse, test_mse, r2):
        path = os.path.join(self._output_dir, f"{key}_{file}.txt")
        with open(path, mode="w", encoding="utf-8") as file:
            file.write(f"Train MSE: {train_mse:.4f}\nTest MSE: {test_mse:.4f}\n")
            file.write(f"R2: {r2:.4f}")


    def _export_data(self, file, key, data_x, data_y, predictions):
        path = os.path.join(self._output_dir, f"{key}_{file}.csv")
        pandas.DataFrame({
            "data_x":           data_x.numpy(),
            "data_y":           data_y.numpy(),
            "predictions": predictions.numpy()
        }).to_csv(path_or_buf=path, index=False)


    def _plot_predictions(self, file, key, train_data_x, train_data_y, train_predictions,
                          test_data_x, test_data_y, test_predictions, r2,
                          xlabel, ylabel, inverted_x, inverted_y, title):
        path = os.path.join(self._output_dir, f"{key}_{file}.png")
        matplotlib.pyplot.figure()
        matplotlib.pyplot.scatter(
            x     = train_data_x.numpy(),
            y     = train_data_y.numpy(),
            color = "deepskyblue",
            label = "Training data"
        )
        matplotlib.pyplot.scatter(
            x     = train_data_x.numpy(),
            y     = train_predictions.numpy(),
            color = "darkseagreen",
            label = "Training predictions"
        )
        matplotlib.pyplot.scatter(
            x     = test_data_x.numpy(),
            y     = test_data_y.numpy(),
            color = "aqua",
            label = "Test data"
        )
        matplotlib.pyplot.scatter(
            x     = test_data_x.numpy(),
            y     = test_predictions.numpy(),
            color = "lawngreen",
            label = "Test predictions"
        )
        matplotlib.pyplot.xlabel(xlabel)
        matplotlib.pyplot.ylabel(ylabel)
        if inverted_x:
            matplotlib.pyplot.gca().invert_xaxis()
        if inverted_y:
            matplotlib.pyplot.gca().invert_yaxis()
        matplotlib.pyplot.title(title)
        handles, labels = matplotlib.pyplot.gca().get_legend_handles_labels()
        handles.append(matplotlib.pyplot.Line2D([], [], color='none'))
        labels.append(f"R2: {r2:.2f}")
        matplotlib.pyplot.legend(handles, labels)
        matplotlib.pyplot.savefig(path)
        matplotlib.pyplot.close()


    def _create_output_dir(self):
        datetime_dir     = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        self._output_dir = os.path.join(os.getcwd(), self._config["output_dir"], datetime_dir)
        os.mkdir(self._output_dir)
