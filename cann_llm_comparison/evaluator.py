import torch


class Evaluator():

    def __init__(self, config):
        self._config = config


    def evaluate(self, iteration, loader, model):
        match self._config["problem"]:
            case "treloar_uniaxial_tension" | \
                 "treloar_biaxial_tension"  | \
                 "treloar_shear":
                return self._evaluate_treloar(iteration, loader, model)
            case "synthetic_a_uniaxial_tension" | \
                 "synthetic_a_biaxial_tension"  | \
                 "synthetic_a_shear"            | \
                 "synthetic_b_uniaxial_tension" | \
                 "synthetic_b_biaxial_tension"  | \
                 "synthetic_b_shear":
                return self._evaluate_synthetic(iteration, loader, model)
            case "brain_tension":
                return self._evaluate_brain_tension(iteration, loader, model)
            case "brain_compression":
                return self._evaluate_brain_compression(iteration, loader, model)
            case "brain_shear":
                return self._evaluate_brain_shear(iteration, loader, model)
            case _:
                raise ValueError("Invalid problem type.")


    def _evaluate_treloar(self, iteration, loader, model):
        train_predictions = model.forward(loader.get_train_data_x()).detach()
        test_predictions  = model.forward(loader.get_test_data_x()).detach()
        return self._evaluate(iteration, loader, train_predictions, test_predictions)
        

    def _evaluate_synthetic(self, iteration, loader, model):
        train_predictions = model.forward(loader.get_train_data_x()).detach()
        test_predictions  = model.forward(loader.get_test_data_x()).detach()
        return self._evaluate(iteration, loader, train_predictions, test_predictions)


    def _evaluate_brain_tension(self, iteration, loader, model):     # extract sigma_1_1
        train_predictions = model.forward(loader.get_train_data_x()).detach()[:,0,0].unsqueeze(1)
        test_predictions  = model.forward(loader.get_test_data_x()).detach( )[:,0,0].unsqueeze(1)
        return self._evaluate(iteration, loader, train_predictions, test_predictions)
        

    def _evaluate_brain_compression(self, iteration, loader, model): # extract sigma_1_1
        train_predictions = model.forward(loader.get_train_data_x()).detach()[:,0,0].unsqueeze(1)
        test_predictions  = model.forward(loader.get_test_data_x()).detach( )[:,0,0].unsqueeze(1)
        return self._evaluate(iteration, loader, train_predictions, test_predictions)


    def _evaluate_brain_shear(self, iteration, loader, model):       # extract sigma_1_2
        train_predictions = model.forward(loader.get_train_data_x()).detach()[:,0,1].unsqueeze(1)
        test_predictions  = model.forward(loader.get_test_data_x()).detach( )[:,0,1].unsqueeze(1)
        return self._evaluate(iteration, loader, train_predictions, test_predictions)

    
    def _evaluate(self, iteration, loader, train_predictions, test_predictions):
        train_loss = torch.nn.MSELoss()(train_predictions, loader.get_train_data_y()).item()
        test_loss  = torch.nn.MSELoss()(test_predictions,  loader.get_test_data_y()).item()
        self._print(iteration, train_loss, test_loss)
        return train_loss


    def _print(self, iteration, train_loss, test_loss):
        print(f"Iteration {iteration}: (Train loss / Test loss) - "
              f"({train_loss:.4f} / {test_loss:.4f})")
