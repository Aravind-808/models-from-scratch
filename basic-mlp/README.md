# Building A Neural Network From Scratch.
## Basics
### File Structure
```
NEURAL_NET/
│
├── activation_functions/
│   ├── sigmoid.py
│   └── tan_h.py
│
├── datasets/
│   └── article_level_data.csv
│
├── error_functions/
│   ├── binary_cross_entropy.py
│   └── mean_squared.py
│
├── network_architecture/
│   ├── layers.py
│   ├── network.py
│
├── article_classification.py
├── fa_solver.py
```

### Activation Functions
#### (i) tan_h
* Consists of tanh and tanh_prime (derivative of tanh). Used mostly for regression tasks, although can be used for classification.
#### (ii) sigmoid
* Consists of sigmoid and its derivative function, used for classification (Binary)
### Error Functions
#### (i) binary cross entropy
* Used for classification
#### (ii) mean squared error
* Used for regression (and classification)

### Layer Architecture
#### (i) layers.py
* has classes for FCLayer, Activation Layer, and base class
#### (ii) network.py
* Class for constructing network.

## Article Classification
* Classifies articles into AI or human generated using sigmoid activation function and binary cross entropy
## Full-Adder Solver (fa_solver)
* Solves Full adder using tanh activation function and mse.

Check code for more details about how all this works. Can't put it all here, plus you can learn and execute at the same time :)

Bye-bye!
