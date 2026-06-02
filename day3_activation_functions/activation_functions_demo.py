import torch
import torch.nn as nn
X=torch.tensor([
    -5.,
    -2.,
    -1.,
    0.,
    1.,
    2.,
    5.
])

sigmoid=nn.Sigmoid()
tanh=nn.Tanh()
relu=nn.ReLU()
leaky_relu=nn.LeakyReLU(0.01)

print(f'''input:
{X}
Sigmoid:
{sigmoid(X)}
Tanh:
{tanh(X)}
ReLU:
{relu(X)}
Leaky ReLU:
{leaky_relu(X)}''')