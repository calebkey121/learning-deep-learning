import numpy as np

class Perceptron:
    def __init__(self, inputs, weights, bias) -> None:
        self.inputs = np.array(inputs)
        self.weights = np.array(weights)
        self.bias = bias
        self.output = self.compute_output()

    def compute_output(self):
        if len(self.weights) != len(self.inputs):
            raise ValueError("weights and biases must be of same length")
        z = np.dot(self.weights, self.inputs)
        z += self.bias
        return np.sign(z)


def main():
    a = Perceptron(inputs=[-1, -1], weights=[-0.6, -0.5], bias=0)
    print(a.output)

if __name__ == "__main__":
    main()