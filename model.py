import math
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

class Phone_classify(nn.Module):
    def __init__(self, input_size, fc1_size, fc2_size, fc3_size, n_classes):
        super(Phone_classify, self).__init__()
        self.input_size = input_size
        self.fc1_size = fc1_size
        self.fc2_size = fc2_size
        self.fc3_size = fc3_size
        self.n_classes = n_classes
        self.front_fc = nn.Sequential(
            nn.Linear(self.input_size, self.fc1_size),
            nn.BatchNorm1d(self.fc1_size),
            nn.ReLU(True),
            nn.Linear(self.fc1_size, self.fc2_size),
            nn.BatchNorm1d(self.fc2_size),
            nn.ReLU(True),
            nn.Linear(self.fc2_size, self.fc3_size),
            nn.BatchNorm1d(self.fc3_size),
            nn.ReLU(True),
            nn.Linear(self.fc3_size, self.n_classes)
        )
        
    def forward(self,x):
        x = self.front_fc(x)

        return x

'''
inputs = torch.randn(4, 13).cuda()
label = torch.tensor([4, 15, 20, 100]).cuda()
model = Phone_classify(13, 39, 39, 78, 188)
model = model.cuda()
criter = nn.CrossEntropyLoss()
out = model(inputs)
print(out.size())
loss = criter(out, label)
print(loss)
'''


