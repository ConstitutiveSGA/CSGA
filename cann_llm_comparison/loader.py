import os
import torch
import numpy
import pandas
import sklearn.model_selection


class Loader():
    
    def __init__(self, config):
        self._config       = config
        self._test_size    = 0.2
        self._random_state = 42
        self._train_data_x = None
        self._train_data_y = None
        self._test_data_x  = None
        self._test_data_y  = None


    def load(self):
        match self._config["problem"]:
            case "treloar_uniaxial_tension":
                self._load_treloar_uniaxial_tension()
            case "treloar_biaxial_tension":
                self._load_treloar_biaxial_tension()
            case "treloar_shear":
                self._load_treloar_shear()
            case "synthetic_a_uniaxial_tension":
                self._load_synthetic_a_uniaxial_tension()
            case "synthetic_a_biaxial_tension":
                self._load_synthetic_a_biaxial_tension()
            case "synthetic_a_shear":
                self._load_synthetic_a_shear()
            case "synthetic_b_uniaxial_tension":
                self._load_synthetic_b_uniaxial_tension()
            case "synthetic_b_biaxial_tension":
                self._load_synthetic_b_biaxial_tension()
            case "synthetic_b_shear":
                self._load_synthetic_b_shear()
            case "brain_tension":
                self._load_brain_tension()
            case "brain_compression":
                self._load_brain_compression()
            case "brain_shear":
                self._load_brain_shear()
            case _:
                raise ValueError("Invalid problem type.")


    def _load_treloar_uniaxial_tension(self):
        self._load_treloar(file = "treloar_uniaxial_tension.csv")


    def _load_treloar_biaxial_tension(self):
        self._load_treloar(file = "treloar_equibiaxial_tension.csv")


    def _load_treloar_shear(self):
        self._load_treloar(file = "treloar_pure_shear.csv")


    def _load_treloar(self, file):        
        path = os.path.join(self._config["input_dir"], "treloar", file)
        data = pandas.read_csv(
            filepath_or_buffer = path,
            delimiter          = ",",
            header             = 0,
            index_col          = False,
        )
        train_data, test_data = sklearn.model_selection.train_test_split(
            data,
            test_size    = self._test_size,
            random_state = self._random_state
        )
        self._train_data_x = torch.tensor(train_data["strains" ].values, dtype=torch.float32)
        self._train_data_y = torch.tensor(train_data["stresses"].values, dtype=torch.float32)
        self._test_data_x  = torch.tensor(test_data[ "strains" ].values, dtype=torch.float32)
        self._test_data_y  = torch.tensor(test_data[ "stresses"].values, dtype=torch.float32)


    def _load_synthetic_a_uniaxial_tension(self):
        self._load_synthetic_a(key = "uni-x")


    def _load_synthetic_a_biaxial_tension(self):
        self._load_synthetic_a(key = "equi")


    def _load_synthetic_a_shear(self):
        self._load_synthetic_a(key = "strip-x")


    def _load_synthetic_a(self, key):
        self._load_synthetic(file = "GenMR_F_P.npy", key = key)


    def _load_synthetic_b_uniaxial_tension(self):
        self._load_synthetic_b(key = "uni-x")


    def _load_synthetic_b_biaxial_tension(self):
        self._load_synthetic_b(key = "equi")


    def _load_synthetic_b_shear(self):
        self._load_synthetic_b(key = "strip-x")


    def _load_synthetic_b(self, key):
        self._load_synthetic(file = "GenMR_RCG_S.npy", key = key)


    def _load_synthetic(self, file, key):
        path = os.path.join(self._config["input_dir"], "synthetic", file)
        data = numpy.load(path, allow_pickle=True)[()][key]
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
        self._train_data_x = torch.tensor(train_data_x, dtype=torch.float32)
        self._train_data_y = torch.tensor(train_data_y, dtype=torch.float32)
        self._test_data_x  = torch.tensor(test_data_x,  dtype=torch.float32)
        self._test_data_y  = torch.tensor(test_data_y,  dtype=torch.float32)


    def _load_brain_tension(self):
        self._load_brain(key = "tens")


    def _load_brain_compression(self):
        self._load_brain(key = "comp")


    def _load_brain_shear(self):
        self._load_brain(key = "simple_shear")


    def _load_brain(self, key):
        path = os.path.join(self._config["input_dir"], "brain", "Brain_F_P.npy")
        data = numpy.load(path, allow_pickle=True)[()][key]
        self._train_data_x, self._test_data_x = sklearn.model_selection.train_test_split(
                data[0],
                test_size    = self._test_size,
                random_state = self._random_state
            )
        self._train_data_y, self._test_data_y = sklearn.model_selection.train_test_split(
            data[1],
            test_size    = self._test_size,
            random_state = self._random_state
        )


    def get_train_data_x(self):
        return self._train_data_x


    def get_train_data_y(self):
        return self._train_data_y


    def get_test_data_x(self):
        return self._test_data_x


    def get_test_data_y(self):
        return self._test_data_y
