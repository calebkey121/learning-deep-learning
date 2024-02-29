from neuron import Neuron
import numpy as np

class NeuralNetwork():
    def __init__(self, layout, inputs, expected, weights=None, learning_rate=0.1) -> None:
        self.layout = layout # list of number of neurons per layer, ex: [3,4] has two hidden layers, 3 in the first and 4 in the second
        self.inputs = inputs
        assert len(expected) == layout[-1] # must match size of output layer
        self.expected = expected
        self.weights = weights
        self.network = self.initialize_network() # gives us 2d list of neurons to run
        self.learning_rate = learning_rate
        self.output = None
        self.error = None


    def initialize_network(self):
        # Just returns a 2d list of initialized neurons with weights and inputs ready
        network = []
        input_len = len(self.inputs)
        inputs = [ None for _ in range(input_len) ] # we don't really care about the actual inputs, we just want the length to be correct
        # in the forward pass we make sure to update the inputs
        num_layers = len(self.layout)
        for layer in range(num_layers):
            num_neurons = self.layout[layer]
            network.append([])
            for neuron in range(num_neurons):
                # if none pass that to Neuron() and it will randomize them, otherwise you best have the right layout for your manually set weights
                weights = None if not self.weights else self.weights[layer][neuron]
                # we want tanh for hidden, sigmoid for last layer
                activation_name = "logistic_sigmoid" if layer == num_layers - 1 else "tanh"
                n = Neuron(inputs, weights, activation_name)
                network[layer].append(n)
            inputs = [ None for _ in range(num_neurons + 1) ] # input for next layer (don't forget we're adding bias)
        return network


    def forward_pass(self):
        inputs = self.inputs
        num_layers = len(self.layout)
        for layer in range(num_layers):
            layer_output = [ 1.0 ] # initialize with 1 for bias term
            num_neurons = self.layout[layer]
            for neuron in range(num_neurons):
                # hidden layers get tanh, output layer gets logistic sigmoid
                activation_function = "tanh" if layer != num_layers - 1 else "logistic_sigmoid"
                output = self.network[layer][neuron].update_inputs(inputs)
                layer_output.append(output)
            inputs = layer_output
        self.output = layer_output[1:]
        return self.output # output layer (has a bias term for hidden layers not needed for output)
    

    def backwards_pass(self):
        der_error_function = [ - ( a - b ) for a, b in zip(self.expected, self.output) ]
        print(der_error_function)
        for a, b in zip(der_error_function, self.network[-1]):
            print(a, b)
        # last layer is a bit special and it has to exist, so i do it explicitly here
        layer_errors = []
        err_last_layer = [ a * b.derivative()  for a, b in zip(der_error_function, self.network[-1]) ]
        print(err_last_layer)
        layer_errors.insert(0, err_last_layer)

        err_next_layer = err_last_layer
        curr_layer = len(self.layout) - 2 # second to last layer
        while curr_layer >= 0:
            err_current_layer = []
            for i in range(self.layout[curr_layer]):
                weights_next_layer = self.weights_to_next_layer(position=i, next_layer=curr_layer + 1)
                dot_prod = np.dot(err_next_layer, weights_next_layer)
                neuron = self.network[curr_layer][i]
                err_current_layer.append(dot_prod * neuron.derivative())
            layer_errors.insert(0, err_current_layer)
            err_next_layer = err_current_layer
            curr_layer -= 1
        # this rounding is a weird use of the book, i would guess i take it out later
        # round(err_last_layer[0], 2) * self.weights[1][0][1] * self.layers[0][0].derivative()
        print(layer_errors)
        self.update_weights(layer_errors)

    
    def weights_to_next_layer(self, position, next_layer):
        weights = []
        for neuron in self.network[next_layer]:
            weights.append(neuron.weights[position + 1]) # account for bias weight
        return weights
    

    def update_weights(self, error_terms):
        num_layers = len(self.layout)
        for layer in range(num_layers):
            num_neurons = self.layout[layer]
            for neuron in range(num_neurons):
                self.network[layer][neuron].update_weights(lr=self.learning_rate, error_term=error_terms[layer][neuron])


    def loss_function(self): # MSE
        total_loss = 0
        output_length = self.layout[-1] # length last layer
        for i in range(output_length):
            total_loss += ( self.network[-1][i].output - self.expected[i] ) ** 2
        self.error = total_loss / (output_length + 1) # Not entirely sure about the + 1 but the book say it will simplify things
        return self.error


    def update_input(self, inputs):
        self.inputs = inputs
        self.forward_pass()
            


def main():
    #x1 = -0.9
    #x2 = 0.1
    #y = [ 1 ]
    #wxg0 = 0.3
    #wxg1 = 0.6
    #wxg2 = -0.1
    #wgf0 = -0.2
    #wgf1 = 0.5
    #inputs = [ 1, x1, x2 ]
    #a = NeuralNetwork (
    #    layout=[2,1],
    #    inputs=inputs,
    #    weights=[ # weights
    #        [ # layer 1
    #            [ wxg0, wxg1, wxg2 ], # g
    #        ],
    #        [ # layer 2
    #            [ wgf0, wgf1 ]    # f
    #        ]
    #    ],
    #    expected=y
    #)
    a = NeuralNetwork (
        inputs=[1,1,1,1,1,1],
        expected=[1,1,1],
        layout=[4,3],
    )
    print(a.forward_pass())
    a.loss_function()
    a.backwards_pass()
    return

if __name__ == "__main__":
    main()
