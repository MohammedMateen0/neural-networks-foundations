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

y = np.array([0,1,1,0])  
p=Perceptron([1,1],-1.5)
p1=Perceptron([1,1],-0.5)
for x in X:
    pred=p.predict(x)
    pred1=p1.predict(x)
    print(f'''Input:{x},AND={pred},OR={pred1}''')

