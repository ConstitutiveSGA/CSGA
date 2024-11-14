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
        train_data_x = loader.get_train_data_x()
        train_data_y = loader.get_train_data_y()

        train_pred            = {}
        train_pred["uni-x"]   = model.forward(train_data_x["uni-x"  ]).detach()[:,0,0]
        train_pred["equi"]    = model.forward(train_data_x["equi"   ]).detach()[:,0,0]
        train_pred["strip-x"] = model.forward(train_data_x["strip-x"]).detach()[:,0,0]

        train_loss            = {}
        train_loss["uni-x"]   = torch.nn.MSELoss()(train_pred["uni-x"]  , train_data_y["uni-x"][  :,0,0]).item()
        train_loss["equi"]    = torch.nn.MSELoss()(train_pred["equi"]   , train_data_y["equi"][   :,0,0]).item()
        train_loss["strip-x"] = torch.nn.MSELoss()(train_pred["strip-x"], train_data_y["strip-x"][:,0,0]).item()

        train_loss_line = (f"MSE [Uniaxial Tension]: {     train_loss['uni-x'  ]:.4f} / "
                           f"MSE [Biaxial Tension]: {      train_loss['equi'   ]:.4f} / "
                           f"MSE [Strip-Biaxial Tension]: {train_loss['strip-x']:.4f}")

        print(f"Iteration {iteration}: {train_loss_line}")

        return sum(train_loss.values()) / len(train_loss), train_loss_line


    def _evaluate_brain(self, iteration, loader, model):
        train_data_x = loader.get_train_data_x()
        train_data_y = loader.get_train_data_y()

        train_pred                 = {}
        train_pred["tens"]         = model.forward(train_data_x["tens"        ]).detach()[:,0,0]
        train_pred["comp"]         = model.forward(train_data_x["comp"        ]).detach()[:,0,0]
        train_pred["simple_shear"] = model.forward(train_data_x["simple_shear"]).detach()[:,0,1]

        train_loss                 = {}
        train_loss["tens"]         = torch.nn.MSELoss()(train_pred["tens"]        , train_data_y["tens"]        .squeeze(1)).item()
        train_loss["comp"]         = torch.nn.MSELoss()(train_pred["comp"]        , train_data_y["comp"]        .squeeze(1)).item()
        train_loss["simple_shear"] = torch.nn.MSELoss()(train_pred["simple_shear"], train_data_y["simple_shear"].squeeze(1)).item()

        train_loss_line = (f"MSE [Tension]: {     train_loss['tens'        ]:.4f} / "
                           f"MSE [Compression]: { train_loss['comp'        ]:.4f} / "
                           f"MSE [Simple Shear]: {train_loss['simple_shear']:.4f}")

        print(f"Iteration {iteration}: {train_loss_line}")

        return sum(train_loss.values()) / len(train_loss), train_loss_line
