# Generalized Hamiltonian Neural Networks

Code for the training and evalutation of Generalized Hamiltonian Neural Networks and other structure-preserving neural networks for data of Hamiltonian systems.

For comparsion purposes [SympNets](https://arxiv.org/abs/2001.03750) are implemented
as well as [HénonNets](https://arxiv.org/abs/2007.04496).
Furthermore, physics-unaware multilayer perceptrons can be trained.

## Install

How to install this python code as a package called `ghnn`:

Clone this repository and then install:
```shell
$ git clone git@github.com:AELITTEN/GHNN.git
$ cd GHNN
$ mv new_pyproject.toml pyproject.toml
$ pip install .
```

If you want to install in editable mode:
```shell
$ git clone git@github.com:AELITTEN/GHNN.git
$ cd GHNN
$ pip install -e .
```

## Train your first NN:

Create a directory for the NN and copy the default settings of the type of NN you want to train
from `GHNN/ghnn/training/` and save it as `settings.json` in the new directory. Then train by
executing the following command in this directory:
```shell
$ python -c 'from ghnn.training.train import train_from_folder; train_from_folder()'
```
Be sure to use Python 3.

#### All possible settings:

Settings name   | Type      | Explanation / Possible setting
----------------| --------- | ------------------------------
device          | string    | Device for the training: {"cpu", "gpu"}.
seed            | int       | Random seed for the initialization of the weights and shuffling of the data.
data\_path      | string    | Path where to find the HDF5 data created by `ghnn.data.data_mapping.create_training_dataframe` or `ghnn.data.data_mapping.create_pendulum_training_dataframe`.
feature\_names  | string[]  | List of feature names that are to be used as input to the NN.
label\_names    | string[]  | List of label names that the NN is supposed to output. Usually feature\_names = label\_names.
bodies          | string[]  | If used in combination with dims feature\_names and label\_names are created automatically.
dims            | string[]  | If used in combination with bodies feature\_names and label\_names are created automatically.
max\_time       | float     | Maximal end time of trajectories: (0, inf).
t\_in\_T        | bool      | Only useful for pendulum data. Scales the time to multiples of one period during inference.
data\_frac      | float     | Fraction of the data to train on: (0, 1].
dtype           | string    | Type to use for the data and NN variables: {"float", "double"}.
path\_logs      | string    | Path where to save the Tensorboard logs.
nn\_type        | string    | Type of NN to be trained: {"MLP", "LA\_SympNet", "G\_SympNet", "GHNN", "HenonNet" or "double\_HenonNet"}.
l\_hamilt       | int       | (for GHNN) Number of learned Hamiltonians.
integrators     | string[]  | (for GHNN) Which integrators to use with the learned Hamiltonians. Can also be a single string ... then gets converted to [intregrators]\*l\_hamilt. Possible are: {"symp\_euler", "leapfrog", "stoermer\_verlet"}.
layer           | int       | (for MLP and GHNN) Number of hidden layers: [0, inf). For GHNN also [[0, inf)]\*l\_hamilt possible.
modules         | int       | (for LA\_SympNet, G\_SympNet, HenonNet and double\_HenonNet) Number of modules/hénon layer: [0, inf).
neurons         | int[]     | (for MLP and GHNN) List of number of neurons in all hidden layers: [[1, inf)]\*layer. Can also be a single int ... then gets converted to [neurons]\*layer. For GHNN also [[[1, inf)]\*layer]\*l\_hamilt possible.
units           | int[]     | (for G\_SympNet and HenonNet) List of number of units in gradient/hénon layers: [[1, inf)]\*modules. Can also be a single int ... then gets converted to [units]\*modules.
sublayer        | int[]     | (for LA\_SympNet) List of number of sublayers in linear layers: [[1, inf)]\*modules. Can also be a single int ... then gets converted to [sublayer]\*modules.
units1          | int[]     | (for double\_HenonNet)  List of number of first units in double hénon layers: [[1, inf)]\*modules. Can also be a single int ... then gets converted to [units1]\*modules.
units2          | int[]     | (for double\_HenonNet)  List of number of second units in double hénon layers: [[1, inf)]\*modules. Can also be a single int ... then gets converted to [units2]\*modules.
activations     | string[]  | List of activations in hidden layers. [\<activation\>]\*layer. \<activation\> can be: {"sigmoid", "relu", "tanh"}. Can also be a single string ... then gets converted to [activations]\*layer. For GHNN also [[\<activation\>]\*layer]\*l\_hamilt possible.
optimizer       | string    | Optimizer for the training. Can be any Keras optimizer identifier.
learning\_rate  | float     | Learning rate for the optimzer: (0, inf).
adam\_beta1     | float     | \beta\_1 for Adam optimizer: (0, 1).
adam\_beta2     | float     | \beta\_2 for Adam optimizer: (0, 1).
lr\_scheduler   | string    | The type of learning rate decrease to 0 {null, "linear", "exponential", "reduce\_on\_plateau"}.
loss            | string    | Loss used for the training. Can be any Keras metric identifier for TF and "mse" or "mae" for pytorch.
loss\_weights   | bool      | Whether weights (1/mean\_label) in the loss should be used or not.
initial\_epoch  | int       | The number of the initial epoch [0, inf).
max\_epochs     | int       | Number of the final epoch in training: [1, inf).
batch\_size     | int       | Number of datapoints in the minibatches. [1, inf/null).
period\_q       | float     | If not null q is remaped to be in [-period\_q, period\_q] during inference (0, inf/null).
model\_path     | string    | Path where to save the model after training.
