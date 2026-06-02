import torch
from torch.utils.data import Dataset

class XORDataSet(Dataset):
    def __init__(self):
        self.X=torch.tensor([
            [0.,0.],
            [0.,1.],
            [1.,0.],
            [1.,1.]
        ])
        self.y=torch.tensor([
            [0.],
            [1.],
            [1.],
            [0.]
        ])
    def __len__(self):
        return len(self.X)
    def __getitem__(self, key):
        return self.X[key],self.y[key]

if __name__ =="__main__":
    
    dataset = XORDataSet()

    print(len(dataset))

    print(dataset[0])

    print(dataset[3])