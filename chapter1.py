import random

# attempted, then followed book example
def compute_output(weights, inputs):
    if len(weights) != len(inputs):
        raise ValueError("weights and biases must be of same length")
    z = 0.0
    for i in range(len(inputs)):
        z += inputs[i] * weights[i]
    if z >= 0:
        return 1
    else:
        return -1
    
def show_learning(weights):
    assert len(weights) == 3
    print(f"w0 = {weights[0]:5.2f}, w1 = {weights[1]:5.2f}, w2 = {weights[2]:5.2f}")

def learning_algorithm(x_train, y_train, weights, learning_rate=0.1):
    show_learning(weights)

    learned = False
    indicies = [ 0, 1, 2, 3 ]
    while not learned:
        i = indicies[random.randint(0, len(indicies) - 1)]
        z = compute_output(weights, x_train[i])
        if z != y_train[i]:
            # adjust
            for j in range(len(weights)):
                adjust = learning_rate * x_train[i][j]
                if z > 0:
                    adjust = -adjust
                weights[j] += adjust
            show_learning(weights)
        else:
            indicies.remove(i)
            if len(indicies) == 0:
                learned = True

def main():
    random.seed(7)
    learning_rate = 0.1

    x_train = [ (1.0, -1.0, -1.0), 
                (1.0, -1.0, 1.0),
                (1.0, 1.0, -1.0),
                (1.0, 1.0, 1.0)
            ] # inputs
    y_train = [ 1.0, 1.0, 1.0, -1.0 ] # ground truth

    # weights = [ round(random.uniform(-1.0, 1.0), 2) for _ in range(3) ]
    weights = [ 0.2, -0.6, 0.25 ]

    learning_algorithm(x_train, y_train, weights, learning_rate)

    

if __name__ == "__main__":
    main()