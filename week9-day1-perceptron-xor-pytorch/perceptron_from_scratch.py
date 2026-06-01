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
    
p=Perceptron([0.5,0.8],-1.5)

x = np.array([2, 1])

prediction = p.predict(x)

print("Input:", x)
print("Prediction:", prediction)