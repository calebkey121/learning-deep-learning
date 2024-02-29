import numpy as np

def logistic_sigmoid(x):
    return 1 / (1 + np.exp(-x))

class Neuron:
    # Basically a neuron, a neuron is typically by itself and has a step function activation
    def __init__(self, inputs: list, weights: list | None = None, activate_name: str = "tanh") -> None:
        # if you set weights it must match the length of inputs, if you don't you will get random weights
        self.inputs = np.array(inputs)
        if weights:
            assert len(weights) == len(inputs)
            self.weights = np.array(weights)
        else: # randomize
            self.weights = np.random.uniform(-1.0, 1.0, len(inputs))
        self.activation_name = activate_name # we separate the name and the actual function here because comparing functions is tricky
        self.activation_function = np.tanh if activate_name == "tanh" else logistic_sigmoid
        self.output = None

    def __repr__(self) -> str:
        return str(self.output)

    def compute_output(self) -> float:
        z = np.dot(self.weights, self.inputs)
        y = self.activation_function(z)
        print(f"z = {round(z, 4)}, y = {y}")
        self.output = y
        return y
    
    def derivative(self) -> float:
        if self.activation_name == "tanh":
            return 1 - np.tanh(self.output) ** 2
        else:
            return self.output * ( 1 - self.output )
        
    def update_inputs(self, new_inputs) -> float:
        self.inputs = np.array(new_inputs)
        return self.compute_output()
    
    def update_weights(self, lr, error_term):
        num_weights = len(self.weights)
        for weight in range(num_weights):
            old_weight = self.weights[weight]
            delta = -lr * self.inputs[weight] * error_term
            self.weights[weight] += delta
            new_weight = self.weights[weight]
            print(f"old weight: {old_weight}, new weight: {round(new_weight, 4)}")
            


def main():
    inputs = [ 1.0, 1.0, 1.0 ]
    weights = [ 0.9, -0.6, -0.5 ]
    a = Neuron(inputs, weights)

if __name__ == "__main__":
    main()