import os
import torch
import numpy
import scipy.io
import sklearn.model_selection


class Loader():

    def __init__(self, config):
        self._config       = config
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
            case "treloar":
                self._load_treloar()
            case _:
                raise ValueError("Invalid problem type.")


    def _load_synthetic_a(self):
        self._load_synthetic(file="GenMR_F_P.npy")


    def _load_synthetic_b(self):
        self._load_synthetic(file="GenMR_RCG_S.npy")


    def _load_synthetic(self, file):
        path = os.path.join(self._config["input_dir"], "synthetic", file)
        data = numpy.load(path, allow_pickle=True)[()]
        (
            self._train_data_x["uni-x"],
            self._train_data_y["uni-x"],
            self._test_data_x[ "uni-x"],
            self._test_data_y[ "uni-x"]
        ) = self._split_synthetic(data=data["uni-x"  ], test_data_indices=[ 2, 3, 5])
        (
            self._train_data_x["equi"],
            self._train_data_y["equi"],
            self._test_data_x[ "equi"],
            self._test_data_y[ "equi"]
        ) = self._split_synthetic(data=data["equi"   ], test_data_indices=[ 5,10,11])
        (
            self._train_data_x["strip-x"],
            self._train_data_y["strip-x"],
            self._test_data_x[ "strip-x"],
            self._test_data_y[ "strip-x"]
        ) = self._split_synthetic(data=data["strip-x"], test_data_indices=[12,13,14])


    def _split_synthetic(self, data, test_data_indices):
        train_data_mask = numpy.ones( shape=(data[0].shape[0],), dtype=bool)
        test_data_mask  = numpy.zeros(shape=(data[0].shape[0],), dtype=bool)
        train_data_mask[test_data_indices] = False
        test_data_mask[ test_data_indices] = True

        train_data_x = data[0][train_data_mask,:,:]
        train_data_y = data[1][train_data_mask,:,:]
        test_data_x  = data[0][test_data_mask, :,:]
        test_data_y  = data[1][test_data_mask, :,:]

        return(
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
        ) = self._split_brain(data["tens"])
        (
            self._train_data_x["comp"],
            self._train_data_y["comp"],
            self._test_data_x[ "comp"],
            self._test_data_y[ "comp"]
        ) = self._split_brain(data["comp"])
        (
            self._train_data_x["simple_shear"],
            self._train_data_y["simple_shear"],
            self._test_data_x[ "simple_shear"],
            self._test_data_y[ "simple_shear"]
        ) = self._split_brain(data["simple_shear"])


    def _split_brain(self, data):
        test_size    = 3
        random_state = 42
        (
            train_data_x,
            test_data_x,
            train_data_y,
            test_data_y
        ) = sklearn.model_selection.train_test_split(
            data[0],
            data[1],
            test_size    = test_size,
            random_state = random_state
        )

        return (
            train_data_x,
            train_data_y,
            test_data_x,
            test_data_y
        )


    def _load_treloar(self):
        path = os.path.join(self._config["input_dir"], "treloar", "Treloar_result.mat")
        data = scipy.io.loadmat(path)
        (
            self._train_data_x["uni-x"],
            self._train_data_y["uni-x"],
            self._test_data_x[ "uni-x"],
            self._test_data_y[ "uni-x"]
        ) = self._split_treloar(data = [data["train_ut_lam"].reshape((-1,)),
                                        data["train_ut_P11"].reshape((-1,))],
                   test_data_indices = [1,2,10]
        )
        (
            self._train_data_x["equi"],
            self._train_data_y["equi"],
            self._test_data_x[ "equi"],
            self._test_data_y[ "equi"]
        ) = self._split_treloar(data = [data["train_bt_lam"].reshape((-1,)),
                                        data["train_bt_P11"].reshape((-1,))],
                   test_data_indices = [2,10]
        )
        (
            self._train_data_x["strip-x"],
            self._train_data_y["strip-x"],
            self._test_data_x[ "strip-x"],
            self._test_data_y[ "strip-x"]
        ) = self._split_treloar(data = [data["train_ps_lam"].reshape((-1,)),
                                        data["train_ps_P11"].reshape((-1,))],
                   test_data_indices = [5,7,8,9]
        )
        

    def _split_treloar(self, data, test_data_indices):
        train_data_mask = numpy.ones( shape=(data[0].shape[0],), dtype=bool)
        test_data_mask  = numpy.zeros(shape=(data[0].shape[0],), dtype=bool)
        train_data_mask[test_data_indices] = False
        test_data_mask[ test_data_indices] = True

        train_data_x = data[0][train_data_mask]
        train_data_y = data[1][train_data_mask]
        test_data_x  = data[0][ test_data_mask]
        test_data_y  = data[1][ test_data_mask]

        return(
            torch.tensor(train_data_x, dtype=torch.float32),
            torch.tensor(train_data_y, dtype=torch.float32),
            torch.tensor(test_data_x,  dtype=torch.float32),
            torch.tensor(test_data_y,  dtype=torch.float32)
        )


    def get_train_data_x(self):
        return self._train_data_x


    def get_train_data_y(self):
        return self._train_data_y


    def get_test_data_x(self):
        return self._test_data_x


    def get_test_data_y(self):
        return self._test_data_y
