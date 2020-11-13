# Project 2
Contains all files used in project 2:

[regression.py](regression.py): Contains implementations of OLS and Ridge regression,
as well as design matrix algorithm. Also includes stochastic gradient descent versions.

[model.py](bias_variance.py): Contains implementation of feed forward neural network
with flexible number of layers/nodes/activation functions.
Also includes backpropagation algorithm. Supports n-dimensional arrays.

[gradient_descent.py](gradient_descent.py): Contains gradient descent class.

[loss_functions.py](loss_functions.py): Contains all loss/cost functions definitions,
with gradients.

[activations.py](activations.py): Contains all activation function definitions,
with gradients.

[randomnet.py](randomnet.py): Contains RandomNet, which builds a randomly cobbled
together neural network using all the activations and initializations found
in the repo. Maximum number of layers is 10, maximum number of nodes in a layer is 100,
but this is only to keep it reasonably fast. Then computes best performer on the
Franke function dataset.

task_a - task_e: Contains code (hopefully) solving each task, with run examples when relevant.
