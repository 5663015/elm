Extreme Learning Machine(ELM): Python code
===

ELM was originally proposed to train "generalized" single-hidden layer feedforward neural networks(SLFNs) with fast learning speed, good generalization capability and provides a unified learning paradigm for regression and classification. This project implemented the ELM  algorithm with python 3.5, you can download source code and install it.

Installation
---

Download source code, enter cmd in the project directory, run the following command:</br>

```c
python setup.py install
```

Then you can import elm module in python like following code:

```python
import elm
```

About ELM
---

The structure of ELM is shown in following figure.


ELM does not need BP algorithm to train the network. First, randomly initialize the input layer to the hidden layer weight, then directly calculate the hidden layer to the output layer weight matrix beta. 
The output of ELM with L hidden nodes can be written as:


Only the weights between the hidden layer and the output, beta matrix, need to be deterimined. The aim is to minimize following formulation:

If there is no regularization C, the solution of beta is:



If set regularization factor C, the solutions of beta are:

* the number of training samples is not huge(solution 1):


* the number of training samples is huge(solution 2):




API
---

`class elm.elm(hidden_units, activation_function,  x, y, C, elm_type, one_hot=True, random_type='normal')`

* hidden_units: list, shape [hidden units, output units], numbers of hidden units and output units
* activation_function: str, 'sigmoid', 'relu', 'sin', 'tanh' or 'leaky_relu'. Activation function of neurals
* x: array, shape[samples, features]. The input of neural network.
* y: array, shape[samples, ], labels
* C: float, regularization parameter
* elm_type: str, 'clf' or 'reg'. 'clf' means ELM solve classification problems, 'reg' means ELM solve regression problems.
* one_hot: bool, Ture or False, default True. The parameter is useful only when elm_type == 'clf'. If the labels need to transformed to one_hot, this parameter is set to be True.
* random_type: str, 'uniform' or 'normal', default:'normal'. Weight initialization method

`elm.elm.fit(algorithm)`

Train the model, compute beta matrix, the weight matrix from hidden layer to output layer

* Parameter:</br>
  * algorithm: str, 'no_re', 'solution1' or 'solution2'. The algorithm to compute beta matrix
* Return:</br>
  * beta: array. The weight matrix from hidden layer to output layer
  * train_score: float. The accuracy or RMSE
  * train_time: str. Time of computing beta

`elm.elm.predict(x)`

Compute the result given data

* Parameter:
  * x: array, shape[samples, features]
* Return:
  * y_: array. Predicted results

`elm.elm.score(x, y)`

Compute accuracy or RMSE given data and labels

* Parameters:
  * x: array, shape[samples, features]
  * y: array, shape[samples, ]
* Return:
  * test_score: float, accuracy or RMSE


References
---

[1] Huang G B, Zhu Q Y, Siew C K. Extreme learning machine: theory and applications[J]. Neurocomputing, 2006, 70(1-3): 489-501.</br>
[2] Huang G B, Zhou H, Ding X, et al. Extreme learning machine for regression and multiclass classification[J]. IEEE Transactions on Systems, Man, and Cybernetics, Part B (Cybernetics), 2012, 42(2): 513-529.</br>
[3] http://www.ntu.edu.sg/home/egbhuang/index.html





