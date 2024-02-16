class Perceptron:
    def __init__(self, inputs, weights, bias) -> None:
        self.inputs = inputs
        self.weights = weights
        self.bias = bias
        self.output = self.compute_output()

    def compute_output(self):
        if len(self.weights) != len(self.inputs):
            raise ValueError("weights and biases must be of same length")
        z = 0.0
        for i in range(len(self.inputs)):
            z += self.inputs[i] * self.weights[i]
        z += self.bias
        if z >= 0:
            return 1
        else:
            return -1


def main():
    a = Perceptron(inputs=[-1, -1], weights=[-0.6, -0.5], bias=0)
    print(a.output)

if __name__ == "__main__":
    main()