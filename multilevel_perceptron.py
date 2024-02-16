from perceptron import Perceptron

class MultilevelPerceptron():
    def __init__(self, inputs, weights, biases) -> None:
        self.inputs = inputs
        self.weights = weights
        self.biases = biases
        #self.layers = [] # i don't think this is really needed
        self.output = self.compute_output()

    def compute_output(self):
        for i, x in enumerate(self.inputs):
            print(f"x{i + 1} = {x}", end=", ")
        print()
        inputs = self.inputs
        y = 0
        for layer in range(len(self.weights)):
            outputs = []
            #self.layers.append([])
            for neuron in range(len(self.weights[layer])):
                p = Perceptron(weights=self.weights[layer][neuron], inputs=inputs, bias=self.biases[layer][neuron])
                #self.layers[-1].append(p)
                outputs.append(p.output)
                print(f"y{y} = {p.output}")
                y += 1
            inputs = outputs
        return outputs
    
    def update_input(self, inputs):
        self.inputs = inputs
        self.output = self.compute_output()
            



def main():
    a = MultilevelPerceptron (
        [-1, -1], # input
        [ # weights
            [ # layer 1
                [ -0.6, -0.5 ], # first perceptron (first layer)
                [ 0.6, 0.6 ]    # second perceptron
            ],
            [ # layer 2
                [ 0.6, 0.6 ]    # first perceptron (second layer)
            ]
        ], 
        [ # biases
            [ 0.9, 0.2 ],
            [ -0.9 ]
        ]
    )
    assert a.output == [-1]
    a.update_input([1, -1])
    assert a.output == [1]
    a.update_input([-1, 1])
    assert a.output == [1]
    a.update_input([1, 1])
    assert a.output == [-1]

if __name__ == "__main__":
    main()