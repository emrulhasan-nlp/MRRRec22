import torch
import torch.nn as nn
import torch.nn.functional as F

class Overall(nn.Module):
    def __init__(self,input_dim, output_dim=1, dropout=0.25):
        
        super(Overall, self).__init__()

        self.dropout=nn.Dropout(dropout)
        self.relu=nn.ReLU()

        self.linear1=nn.Linear(input_dim, 256)
        self.linear2=nn.Linear(256, 128)  
        self.linear3=nn.Linear(128, 64)
        self.linear4=nn.Linear(64, 32)
        # self.linear5=nn.Linear(32, 16)
        self.fc=nn.Linear(32,output_dim)

    def forward(self, x):
        
        # layer1=self.linear1(x)
        # layer2=self.dropout(self.linear2(layer1))
        # layer3=self.dropout(self.linear3(layer2))
        # layer4=self.relu(self.dropout(self.linear4(layer3)))

        x=self.relu(self.linear1(x))
        x=self.relu(self.linear2(x))
        x=self.relu(self.linear3(x))
        x=self.relu(self.linear4(x))
        # x=self.relu(self.linear5(x))


        overall_rating=self.fc(self.dropout(x))

        return overall_rating