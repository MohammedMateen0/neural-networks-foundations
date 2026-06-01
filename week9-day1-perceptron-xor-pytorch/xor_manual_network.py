import numpy as np
class Perceptron:
    def __init__(self,weight,bias):
        self.weight=weight
        self.bias=bias
    def step_activation(self,z):
        return 1 if z>0 else 0
    def predict(self,x):
        z=np.dot(x,self.weight)+self.bias
        return self.step_activation(z)

X = np.array([
    [0,0],
    [0,1],
    [1,0],
    [1,1]
])

and_neuron = Perceptron([1,1], -1)
or_neuron  = Perceptron([1,1], -0.5)

output_neuron = Perceptron([-1,2], -1.5)

for x in X:

    h1 = and_neuron.predict(x)
    h2 = or_neuron.predict(x)

    hidden = np.array([h1,h2])

    y = output_neuron.predict(hidden)

    print(f"{x} -> {y}")