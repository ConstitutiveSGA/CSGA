import torch


class Evaluator():

    def __init__(self, config):
        self._config = config


    def evaluate(self, iteration, loader, model):
        match self._config["problem"]:
            case "synthetic_a" | "synthetic_b":
                return self._evaluate_synthetic(iteration, loader, model)
            case "brain":
                return self._evaluate_brain(iteration, loader, model)
            case _:
                raise ValueError("Invalid problem type.")


    def _evaluate_synthetic(self, iteration, loader, model):
        train_data_x = torch.cat(list(loader.get_train_data_x().values()))
        train_data_y = torch.cat(list(loader.get_train_data_y().values()))
        test_data_x  = torch.cat(list(loader.get_test_data_x( ).values()))
        test_data_y  = torch.cat(list(loader.get_test_data_y( ).values()))

        train_preds = model.forward(train_data_x).detach()
        test_preds  = model.forward(test_data_x ).detach()

        train_loss = torch.nn.MSELoss()(train_preds, train_data_y).item()
        test_loss  = torch.nn.MSELoss()( test_preds,  test_data_y).item()

        self._print(iteration, train_loss, test_loss)

        return train_loss


    def _evaluate_brain(self, iteration, loader, model):
        train_data_x = loader.get_train_data_x()
        train_data_y = loader.get_train_data_y()
        test_data_x  = loader.get_test_data_x()
        test_data_y  = loader.get_test_data_y()

        train_pred                 = {}
        train_pred["tens"]         = model.forward(train_data_x["tens"        ]).detach()[:,0,0]
        train_pred["comp"]         = model.forward(train_data_x["comp"        ]).detach()[:,0,0]
        train_pred["simple_shear"] = model.forward(train_data_x["simple_shear"]).detach()[:,0,1]
        train_pred                 = torch.cat(list(train_pred.values()))
        test_pred                  = {}
        test_pred["tens"]          = model.forward(test_data_x["tens"        ]).detach()[:,0,0]
        test_pred["comp"]          = model.forward(test_data_x["comp"        ]).detach()[:,0,0]
        test_pred["simple_shear"]  = model.forward(test_data_x["simple_shear"]).detach()[:,0,1]
        test_pred                  = torch.cat(list(test_pred.values()))

        train_data_y = torch.cat(list(train_data_y.values())).squeeze(1)
        test_data_y  = torch.cat(list( test_data_y.values())).squeeze(1)

        train_loss = torch.nn.MSELoss()(train_pred, train_data_y).item()
        test_loss  = torch.nn.MSELoss()( test_pred,  test_data_y).item()

        self._print(iteration, train_loss, test_loss)

        return train_loss


    def _print(self, iteration, train_loss, test_loss):
        print(f"Iteration {iteration}: (Train loss / Test loss) - "
              f"({train_loss:.4f} / {test_loss:.4f})")
