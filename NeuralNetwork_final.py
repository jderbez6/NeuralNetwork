### Neural Network Code
### Jose Emilio Derbez Safie

## Library import
import numpy as np
import pandas as pd
import seaborn as sns


### Function definition
# Sigmoid function 
def sigmoid(x, lam):
    res = (1)/(1 + np.exp(-x))
    return res
# Linear function
def identity(x, lam):
    return x

# function used to calculate activation values for neurons in layer
def layer_activation(layer, input_values, lr):
    output_values = []
    for i in range(len(input_values)):
        activation = layer[i].activation(input_values[i], lr)
        output_values.append(activation)
    return output_values
# function ued to calculate 
def MSE(true, predicted, root = False):
    se = np.square(np.subtract(true, predicted))
    mse = se.mean()
    rmse = np.sqrt(mse)
    if root:
        return rmse
    else:
        return mse

class Neuron:
    def __init__(self, activation_function = sigmoid):
        self.activation_function = activation_function

    def activation(self, input, lr):
        return self.activation_function(input, lr)
    
class Neural_Network:
    """Neural network class"""
    def __init__(self, n_inputs:int, n_hidden:int, n_outputs:int, hidden_activation : ["sigmoid", "linear"] ="sigmoid", output_activation : ["sigmoid", "linear"] ="sigmoid",
                 input_bias : bool = True, hidden_bias: bool = False, learning_rate : float = 0.5, momentum: float = 0.5) -> None:
        """
        Create a Neural Network object.
        Parameters:
        --------------
        n_inputs : int
            number of inputs in neural network
        n_hidden : int
            number of neurons in hidden layer
        n_outputs : int
            number of outputs prodeces by neural network
        hidden_activation : {'sigmoid', 'linear'} = 'sigmoid'
            activation function used in hidden layer
        output_activation : {'sigmoid', 'linear'} = 'sigmoid'
            activation function used in output layer
        input_bias : bool = True
            if true, include bias neuron in input layer
        hidden_bias : bool = True
            if true, include bias neuron in hidden layer
        learning_rate : float = 0.5
            learning rate parameter used in netwrok training
        momentum : float = 0.5
            momentum parameter used in netwrok training
        """

        
        if hidden_activation == "sigmoid":
            self.hidden_activation = sigmoid
        
        elif hidden_activation == "linear":
            self.hidden_activation = identity


        if output_activation == "sigmoid":
            self.output_activation = sigmoid
        
        elif output_activation == "linear":
            self.output_activation = identity

        self.lr = learning_rate
        self.momentum = momentum

        self.ib = input_bias
        self.hb = hidden_bias

        self.n_inputs = n_inputs + int(input_bias)                                              # Adding one extra if input_bias = True (n_inputs = I)
        self.n_hidden = n_hidden                                                                # (n_hidden = J)
        self.n_outputs = n_outputs                                                              # (n_outputs = K)
        
        self.input_layer = [Neuron(identity) for n in range(n_inputs)] 
        self.hidden_layer = [Neuron(self.hidden_activation) for n in range(n_hidden)]
        self.output_layer = [Neuron(self.output_activation) for n in range(n_outputs)]

        self.input = None                                           
        self.hidden = None                                                      # Values after Wx, before activation
        self.output = None                                                      # Values after Uh, before activation

        self.hidden_values = None                                               # Values after layer activation
        self.output_values = None                                               # Values after layer activation

        self.output_gradients = np.ones((self.n_outputs, self.n_hidden + int(hidden_bias)))                 # Initializing K x J matrix for storing gradients for hidden to output weights
        self.hidden_gradients = np.ones((self.n_hidden, self.n_inputs))                                     # Initializing J x I matrix for storing gradients for input to hidden weights
        
        self.hidden_weights = np.random.uniform(low = -1.0, high = 1.0, size = (self.n_hidden, self.n_inputs))                          # Initializing J x I matrix with random weights   
        self.output_weights = np.random.uniform(low = -1.0, high = 1.0, size = (self.n_outputs, self.n_hidden + int(hidden_bias)))      # Initializing K x J matrix with random weights

        self.previous_change_output = np.zeros((self.n_outputs, self.n_hidden + int(hidden_bias)))          # Initializing K x J matrix for storing previous weight change in output layer
        self.previous_change_hidden = np.zeros((self.n_hidden, self.n_inputs))                              # Initializing J x I matrix for storing previous weight change in hidden layer

        # Attributes used in training
        self.epoch = 1
        self.epoch_train_error = []
        self.epoch_validation_error = []
        self.error = None

        self.best_weights = None

    def forward_pass(self, input, output: bool = False):
        """
        Calculate network output given corresponding inputs
        Parameters:
        ----------------
        input: int or list type
            inputs of network
        output: bool = False
            if True, return network output
        """
        input = list(input)
        if self.ib:
            input.append(1.)                                                                        # Bias input = 1
        self.input = input                                          
        self.hidden = self.hidden_weights @ input                                                   # Wx
        self.hidden_values = layer_activation(self.hidden_layer, self.hidden, self.lr)              # Hidden layer activation
        if self.hb:
            self.hidden_values.append(1.)                                                           # Hidden input = 1
        self.output = self.output_weights @ self.hidden_values                                      # Uh
        self.output_values = layer_activation(self.output_layer, self.output, self.lr)              # Output layer activation
        if output:
            return self.output_values
    
    def backpropagation(self, real_values, output=False):
        """
        Calculate partial derivatives for all weights in network
        Parameters:
        ----------------
        real_values: int or list type
            real values used to calculate error of network
        output: bool = False
            if True, return network gradietns
        """

        k_gradients = []                          # Auxiliary list used to store k gradients (gradients used in both layers back prop)
        # self.error = MSE(real_values, self.output_values)
        self.error = MSE(real_values, self.output_values, root=True)

        # K gradients calculation

        if self.output_activation == sigmoid:
            for k in range(len(self.output_values)):
                if len(self.output) > 1:
                    temp = self.output_values[k] * (1 - self.output_values[k]) * (real_values[k] - self.output_values[k])
                    k_gradients.append(temp)
                else:
                    temp = self.output_values[0] * (1 - self.output_values[0]) * (real_values - self.output_values[0])
                    k_gradients.append(temp)

        elif self.output_activation == identity:
            for k in range(len(self.output_values)):
                if len(self.output) > 1:
                    temp = (real_values[k] - self.output_values[k])
                    k_gradients.append(temp)
                else:
                    temp =  (real_values - self.output_values[0])
                    k_gradients.append(temp)
            
            
        # Output gradients calculatotion

        for k in range(len(k_gradients)):
            for j in range(len(self.hidden_values)):
                temp = k_gradients[k] * self.hidden_values[j]
                self.output_gradients[k][j] = temp 

        # Hidden gradients calculattion 

        if self.hidden_activation == sigmoid:
            for i in range(len(self.input)):
                for j in range(len(self.hidden)):
                    sum = 0
                    for k in range(len(k_gradients)):
                        sum +=k_gradients[k] * self.output_weights[k][j]
                    i_gradient = self.hidden_values[j] * (1- self.hidden_values[j]) * sum
                    temp =  i_gradient * self.input[i]
                    self.hidden_gradients[j][i] = temp

        elif self.hidden_activation == identity:
            for i in range(len(self.input)):
                for j in range(len(self.hidden)):
                    sum = 0
                    for k in range(len(k_gradients)):
                        sum +=k_gradients[k] * self.output_weights[k][j]
                    i_gradient = sum
                    temp =  i_gradient * self.input[i]
                    self.hidden_gradients[j][i] = temp

        if output:
            return self.output_gradients, self.hidden_gradients
        
    def weight_updating(self):
        """
        Update weights in network after gradient calculation
        """

        # Delta weight calculation
        # For hidden to output layer
        current_change_output = np.add(self.output_gradients * self.lr, self.momentum * self.previous_change_output)
        self.previous_change_output = current_change_output

        # For input to hidden layer
        current_change_hidden = np.add(self.hidden_gradients * self.lr, self.momentum * self.previous_change_hidden)
        self.previous_change_hidden = current_change_hidden
 
        # Weight updating
        self.output_weights = np.add(self.output_weights, current_change_output)
        self.hidden_weights = np.add(self.hidden_weights, current_change_hidden)

    def train(self, train_input, train_output, validation_input, validation_output, epochs:int = 100):
        """
        Train neural network
        Parameters:
        ----------------
        train_input: int or list type
            training inputs
        train_output: int or list type
            training outputs
        validation_input: int or list type
            validation inputs
        validation_output: int or list type
            validation outputs
        epochs: int = 100
            Number of epochs to train the neural network
        """
        
        self.best_error = 100000000000000000
        while self.epoch <= (epochs):

            epoch_errors = []

            for j in range(len(train_input)):
                self.forward_pass(train_input[j])
                self.backpropagation(train_output[j])
                epoch_errors.append(self.error)
                self.weight_updating()
            
            epoch_error = np.mean(epoch_errors)
            self.epoch_train_error.append(epoch_error)

            val_error = []
            for k in range(len(validation_input)):
                pred = self.forward_pass(validation_input[k], output = True)
                val_error.append(MSE(validation_output[k], pred, root=True))
            

            val_total_error = np.mean(val_error)
            self.epoch_validation_error.append(val_total_error)
            
            if val_total_error < self.best_error:
                self.best_error = val_total_error
                self.best_weights = [self.hidden_weights, self.output_weights]

            print(f"Epoch: {self.epoch} | Train_error : {round(epoch_error, 3)} | Validation_error : {round(val_total_error, 3)}")
            self.epoch +=1

    def graph_error(self):
        """
        Plot training and validation errors through training
        """
        vis = pd.DataFrame({"epochs" : np.arange(1, self.epoch, 1), "Train_error" : self.epoch_train_error, "Validation_error" : self.epoch_validation_error}, )
        vis = vis.melt("epochs", value_vars=["Train_error", "Validation_error"])
        sns.lineplot(data = vis, x = "epochs", y = "value", hue = "variable")
        
    def save_weights(self, folder : str, best_weights = True):
        """
        Save neural network weights
        Parameters:
        ----------------
        folder: str
            folder bath to save weights
        best_weights: bool = True
            if true, save network best weights, else save current weights
        """
        if best_weights:
            np.save(f"{folder}/hidden_weights", self.best_weights[0])
            np.save(f"{folder}/output_weights", self.best_weights[1])

        else:
            np.save(f"{folder}/hidden_weights", self.hidden_weights)
            np.save(f"{folder}/output_weights", self.output_weights)


    def load_weights(self, hidden_weights, output_weights):
        """
        Load neural network weights
        Parameters:
        ----------------
        hidden_weights: np.ndarray
            hidden weights to be loaded
        output_weights: np.ndarray
            output weights to be loaded
        """
        self.hidden_weights = hidden_weights
        self.output_weights = output_weights
