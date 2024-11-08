import os
import torch
import pandas
import datetime
import matplotlib.pyplot

class Exporter():

    def __init__(self, config):
        self._config     = config
        self._output_dir = None

    
    def set_up(self):
        self._create_output_dir()


    def export(self, loader, model, model_code):
        match self._config["problem"]:
            case "treloar_uniaxial_tension":
                self._export_treloar_uniaxial_tension(loader, model, model_code)
            case "treloar_biaxial_tension":
                self._export_treloar_biaxial_tension(loader, model, model_code)
            case "treloar_shear":
                self._export_treloar_shear(loader, model, model_code)
            case "synthetic_a_uniaxial_tension":
                self._export_synthetic_a_uniaxial_tension(loader, model, model_code)
            case "synthetic_a_biaxial_tension":
                self._export_synthetic_a_biaxial_tension(loader, model, model_code)
            case "synthetic_a_shear":
                self._export_synthetic_a_shear(loader, model, model_code)
            case "synthetic_b_uniaxial_tension":
                self._export_synthetic_b_uniaxial_tension(loader, model, model_code)
            case "synthetic_b_biaxial_tension":
                self._export_synthetic_b_biaxial_tension(loader, model, model_code)
            case "synthetic_b_shear":
                self._export_synthetic_b_shear(loader, model, model_code)
            case "brain_tension":
                self._export_brain_tension(loader, model, model_code)
            case "brain_compression":
                self._export_brain_compression(loader, model, model_code)
            case "brain_shear":
                self._export_brain_shear(loader, model, model_code)
            case _:
                raise ValueError("Invalid problem type.")


    def _export_treloar_uniaxial_tension(self, loader, model, model_code):
        self._export_treloar(loader, model, model_code, "Treloar Uniaxial Tension")


    def _export_treloar_biaxial_tension(self, loader, model, model_code):
        self._export_treloar(loader, model, model_code, "Treloar Biaxial Tension")


    def _export_treloar_shear(self, loader, model, model_code):
        self._export_treloar(loader, model, model_code, "Treloar Pure Shear")


    def _export_treloar(self, loader, model, model_code, title):
        train_predictions = model.forward(loader.get_train_data_x()).detach()
        test_predictions  = model.forward(loader.get_test_data_x()).detach()
        train_loss = torch.nn.MSELoss()(train_predictions, loader.get_train_data_y()).item()
        test_loss  = torch.nn.MSELoss()(test_predictions,  loader.get_test_data_y()).item()

        self._export_problem()
        self._export_model(model, model_code)
        self._export_loss(train_loss, test_loss)
        self._export_data(
            file        = "training_data", 
            data_x      = loader.get_train_data_x(),
            data_y      = loader.get_train_data_y(),
            predictions = train_predictions
        )
        self._export_data(
            file        = "test_data", 
            data_x      = loader.get_test_data_x(),
            data_y      = loader.get_test_data_y(),
            predictions = test_predictions
        )
        self._plot_predictions(
            train_data_x      = loader.get_train_data_x(),
            train_data_y      = loader.get_train_data_y(),
            train_predictions = train_predictions,
            test_data_x       = loader.get_test_data_x(),
            test_data_y       = loader.get_test_data_y(),
            test_predictions  = test_predictions,
            xlabel            = "Stretch",
            ylabel            = "Stress",
            inverted_x        = False,
            inverted_y        = False,
            title             = title
        )


    def _export_synthetic_a_uniaxial_tension(self, loader, model, model_code):
        self._export_synthetic(
            loader         = loader, 
            model          = model,
            model_code     = model_code,
            strain_slc_idx = [0,0], # extract epsilon_1_1
            stress_slc_idx = [0,0], # extract   sigma_1_1
            xlabel         = "Stretch [F_11]",
            ylabel         = "Stress [P_11]",
            title          = "Synthetic Data [A] Uniaxial Tension"
        )


    def _export_synthetic_a_biaxial_tension(self, loader, model, model_code):
        self._export_synthetic(
            loader         = loader, 
            model          = model,
            model_code     = model_code,
            strain_slc_idx = [0,0], # extract epsilon_1_1
            stress_slc_idx = [0,0], # extract   sigma_1_1
            xlabel         = "Stretch [F_11]",
            ylabel         = "Stress [P_11]",
            title          = "Synthetic Data [A] Biaxial Tension"
        )


    def _export_synthetic_a_shear(self, loader, model, model_code):
        self._export_synthetic(
            loader         = loader, 
            model          = model,
            model_code     = model_code,
            strain_slc_idx = [0,1], # extract epsilon_1_2
            stress_slc_idx = [0,0], # extract   sigma_1_1
            xlabel         = "Stretch [F_12]",
            ylabel         = "Stress [P_11]",
            title          = "Synthetic Data [A] Pure Shear"
        )


    def _export_synthetic_b_uniaxial_tension(self, loader, model, model_code):
        self._export_synthetic(
            loader         = loader, 
            model          = model,
            model_code     = model_code,
            strain_slc_idx = [0,0], # extract epsilon_1_1
            stress_slc_idx = [0,0], # extract   sigma_1_1
            xlabel         = "Strain [RCG_11]",
            ylabel         = "Stress [S_11]",
            title          = "Synthetic Data [A] Uniaxial Tension"
        )


    def _export_synthetic_b_biaxial_tension(self, loader, model, model_code):
        self._export_synthetic(
            loader         = loader, 
            model          = model,
            model_code     = model_code,
            strain_slc_idx = [0,0], # extract epsilon_1_1
            stress_slc_idx = [0,0], # extract   sigma_1_1
            xlabel         = "Strain [RCG_11]",
            ylabel         = "Stress [S_11]",
            title          = "Synthetic Data [A] Biaxial Tension"
        )


    def _export_synthetic_b_shear(self, loader, model, model_code):
        self._export_synthetic(
            loader         = loader, 
            model          = model,
            model_code     = model_code,
            strain_slc_idx = [0,1], # extract epsilon_1_2
            stress_slc_idx = [0,0], # extract   sigma_1_1
            xlabel         = "Strain [RCG_12]",
            ylabel         = "Stress [S_11]",
            title          = "Synthetic Data [B] Pure Shear"
        )


    def _export_synthetic(self, loader, model, model_code, strain_slc_idx, stress_slc_idx,
                          xlabel, ylabel, title):
        train_predictions = model.forward(loader.get_train_data_x()).detach()
        test_predictions  = model.forward(loader.get_test_data_x()).detach()
        train_predictions = train_predictions[:, stress_slc_idx[0], stress_slc_idx[1]].unsqueeze(1)
        test_predictions  = test_predictions[ :, stress_slc_idx[0], stress_slc_idx[1]].unsqueeze(1)
        train_d_y = loader.get_train_data_y()[:, stress_slc_idx[0], stress_slc_idx[1]].unsqueeze(1)
        test_d_y  = loader.get_test_data_y()[:,  stress_slc_idx[0], stress_slc_idx[1]].unsqueeze(1)
        train_loss = torch.nn.MSELoss()(train_predictions, train_d_y).item()
        test_loss  = torch.nn.MSELoss()(test_predictions,   test_d_y).item()

        self._export_problem()
        self._export_model(model, model_code)
        self._export_loss(train_loss, test_loss)
        self._export_data(
            file        = "training_data", 
            data_x      = loader.get_train_data_x()[:, strain_slc_idx[0], strain_slc_idx[1]],
            data_y      =         train_d_y.squeeze(1),
            predictions = train_predictions.squeeze(1)
        )
        self._export_data(
            file        = "test_data", 
            data_x      = loader.get_test_data_x()[:, strain_slc_idx[0], strain_slc_idx[1]],
            data_y      =         test_d_y.squeeze(1),
            predictions = test_predictions.squeeze(1)
        )

        self._plot_predictions(
            train_data_x      = loader.get_train_data_x()[:, strain_slc_idx[0], strain_slc_idx[1]],
            train_data_y      =         train_d_y.squeeze(1),
            train_predictions = train_predictions.squeeze(1),
            test_data_x       = loader.get_test_data_x()[:, strain_slc_idx[0], strain_slc_idx[1]],
            test_data_y       =         test_d_y.squeeze(1),
            test_predictions  = test_predictions.squeeze(1),
            xlabel            = xlabel,
            ylabel            = ylabel,
            inverted_x        = False,
            inverted_y        = False,
            title             = title
        )


    def _export_brain_tension(self, loader, model, model_code):
        slc_idx     = [0,0] # extract epsilon_1_1 / sigma_1_1
        predictions = self._export_brain(loader, model, model_code, slc_idx)
        
        self._plot_predictions(
            train_data_x      = loader.get_train_data_x()[:, slc_idx[0], slc_idx[1]],
            train_data_y      = loader.get_train_data_y().squeeze(1),
            train_predictions =      predictions["train"].squeeze(1),
            test_data_x       = loader.get_test_data_x()[:, slc_idx[0], slc_idx[1]],
            test_data_y       = loader.get_test_data_y().squeeze(1),
            test_predictions  =      predictions["test"].squeeze(1),
            xlabel            = "Stretch [F_11]",
            ylabel            = "Stress [P_11]",
            inverted_x        = False,
            inverted_y        = False,
            title             = "Brain Tension"
        )


    def _export_brain_compression(self, loader, model, model_code):
        slc_idx     = [0,0] # extract epsilon_1_1 / sigma_1_1
        predictions = self._export_brain(loader, model, model_code, slc_idx)
        
        self._plot_predictions(
            train_data_x      = loader.get_train_data_x()[:, slc_idx[0], slc_idx[1]],
            train_data_y      = loader.get_train_data_y().squeeze(1),
            train_predictions =      predictions["train"].squeeze(1),
            test_data_x       = loader.get_test_data_x()[:, slc_idx[0], slc_idx[1]],
            test_data_y       = loader.get_test_data_y().squeeze(1),
            test_predictions  =      predictions["test"].squeeze(1),
            xlabel            = "Stretch [F_11]",
            ylabel            = "Stress [P_11]",
            inverted_x        = True,
            inverted_y        = True,
            title             = "Brain Compression"
        )


    def _export_brain_shear(self, loader, model, model_code):       
        slc_idx     = [0,1] # extract epsilon_1_2 / sigma_1_2
        predictions = self._export_brain(loader, model, model_code, slc_idx)
        
        self._plot_predictions(
            train_data_x      = loader.get_train_data_x()[:, slc_idx[0], slc_idx[1]],
            train_data_y      = loader.get_train_data_y().squeeze(1),
            train_predictions =      predictions["train"].squeeze(1),
            test_data_x       = loader.get_test_data_x()[:, slc_idx[0], slc_idx[1]],
            test_data_y       = loader.get_test_data_y().squeeze(1),
            test_predictions  =      predictions["test"].squeeze(1),
            xlabel            = "Stretch [F_12]",
            ylabel            = "Stress [P_12]",
            inverted_x        = False,
            inverted_y        = False,
            title             = "Brain Shear"
        )


    def _export_brain(self, loader, model, model_code, slc_idx):
        train_predictions = model.forward(loader.get_train_data_x()).detach()
        test_predictions  = model.forward(loader.get_test_data_x()).detach()
        train_predictions = train_predictions[:, slc_idx[0], slc_idx[1]].unsqueeze(1)
        test_predictions  = test_predictions[ :, slc_idx[0], slc_idx[1]].unsqueeze(1)
        train_loss = torch.nn.MSELoss()(train_predictions, loader.get_train_data_y()).item()
        test_loss  = torch.nn.MSELoss()(test_predictions,  loader.get_test_data_y()).item()

        self._export_problem()
        self._export_model(model, model_code)
        self._export_loss(train_loss, test_loss)
        self._export_data(
            file        = "training_data", 
            data_x      = loader.get_train_data_x()[:, slc_idx[0], slc_idx[1]],
            data_y      = loader.get_train_data_y().squeeze(1),
            predictions =         train_predictions.squeeze(1)
        )
        self._export_data(
            file        = "test_data", 
            data_x      = loader.get_test_data_x()[:, slc_idx[0], slc_idx[1]],
            data_y      = loader.get_test_data_y().squeeze(1),
            predictions =         test_predictions.squeeze(1)
        )

        return {"train": train_predictions, "test": test_predictions}
        

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


    def _export_loss(self, train_loss, test_loss):
        path = os.path.join(self._output_dir, "loss.txt")
        with open(path, mode="w", encoding="utf-8") as file:
            file.write(f"Train loss: {train_loss:.4f}\nTest loss: {test_loss:.4f}")


    def _export_data(self, file, data_x, data_y, predictions):
        path = os.path.join(self._output_dir, f"{file}.csv")
        pandas.DataFrame({
            "data_x":           data_x.numpy(),
            "data_y":           data_y.numpy(),
            "predictions": predictions.numpy()
        }).to_csv(path_or_buf=path, index=False)


    def _plot_predictions(self, train_data_x, train_data_y, train_predictions, test_data_x, 
                          test_data_y, test_predictions, xlabel, ylabel, inverted_x, inverted_y,
                          title):
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
        matplotlib.pyplot.legend()
        path = os.path.join(self._output_dir, "predictions.png")
        matplotlib.pyplot.savefig(path)
        matplotlib.pyplot.close()


    def _create_output_dir(self):
        datetime_dir     = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        self._output_dir = os.path.join(os.getcwd(), self._config["output_dir"], datetime_dir)
        os.mkdir(self._output_dir)
