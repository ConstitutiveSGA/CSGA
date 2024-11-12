import os
import torch
import numpy
import sklearn.model_selection


class Loader():

    def __init__(self, config):
        self._config       = config
        self._test_size    = 0.2
        self._random_state = 42
        self._train_data_x = {}
        self._train_data_y = {}
        self._test_data_x  = {}
        self._test_data_y  = {}


    def load(self):
        match self._config["problem"]:
            case "synthetic_a":
                self._load_synthetic_a()
            case "synthetic_b":
                self._load_synthetic_b()
            case "brain":
                self._load_brain()
            case _:
                raise ValueError("Invalid problem type.")


    def _load_synthetic_a(self):
        self._load_synthetic(file = "GenMR_F_P.npy")


    def _load_synthetic_b(self):
        self._load_synthetic(file = "GenMR_RCG_S.npy")


    def _load_synthetic(self, file):
        path = os.path.join(self._config["input_dir"], "synthetic", file)
        data = numpy.load(path, allow_pickle=True)[()]
        (
            self._train_data_x["uni-x"],
            self._train_data_y["uni-x"],
            self._test_data_x[ "uni-x"],
            self._test_data_y[ "uni-x"]
        ) = self._split_synthetic(data["uni-x"])
        (
            self._train_data_x["equi"],
            self._train_data_y["equi"],
            self._test_data_x[ "equi"],
            self._test_data_y[ "equi"]
        ) = self._split_synthetic(data["equi"])
        (
            self._train_data_x["strip-x"],
            self._train_data_y["strip-x"],
            self._test_data_x[ "strip-x"],
            self._test_data_y[ "strip-x"]
        ) = self._split_synthetic(data["strip-x"])


    def _split_synthetic(self, data):
        (
            train_data_x,
            train_data_y,
            test_data_x,
            test_data_y

        ) = self._split(data)
        return (
            torch.tensor(train_data_x, dtype=torch.float32),
            torch.tensor(train_data_y, dtype=torch.float32),
            torch.tensor(test_data_x,  dtype=torch.float32),
            torch.tensor(test_data_y,  dtype=torch.float32)
        )


    def _load_brain(self):
        path = os.path.join(self._config["input_dir"], "brain", "Brain_F_P.npy")
        data = numpy.load(path, allow_pickle=True)[()]
        (
            self._train_data_x["tens"],
            self._train_data_y["tens"],
            self._test_data_x[ "tens"],
            self._test_data_y[ "tens"]
        ) = self._split(data["tens"])
        (
            self._train_data_x["comp"],
            self._train_data_y["comp"],
            self._test_data_x[ "comp"],
            self._test_data_y[ "comp"]
        ) = self._split(data["comp"])
        (
            self._train_data_x["simple_shear"],
            self._train_data_y["simple_shear"],
            self._test_data_x[ "simple_shear"],
            self._test_data_y[ "simple_shear"]
        ) = self._split(data["simple_shear"])


    def _split(self, data):
        train_data_x, test_data_x = sklearn.model_selection.train_test_split(
                data[0],
                test_size    = self._test_size,
                random_state = self._random_state
            )
        train_data_y, test_data_y = sklearn.model_selection.train_test_split(
            data[1],
            test_size    = self._test_size,
            random_state = self._random_state
        )
        return (
            train_data_x,
            train_data_y,
            test_data_x,
            test_data_y
        )


    def get_train_data_x(self):
        return self._train_data_x


    def get_train_data_y(self):
        return self._train_data_y


    def get_test_data_x(self):
        return self._test_data_x


    def get_test_data_y(self):
        return self._test_data_y
