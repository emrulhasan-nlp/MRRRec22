import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Criteria(nn.Module):
    def __init__(self,num_users, num_items, output_dim=5, dropout=0.5):
        
        super(Criteria, self).__init__()

        self.user_embedding=nn.Embedding(num_users, 128)
        self.user_embedding.weight.requires_grad=True

        self.item_embedding=nn.Embedding(num_items, 128)
        self.item_embedding.weight.requires_grad=True

        # Initialize all weights using random uniform distribution from [-0.01, 0.01]
        self.user_embedding.weight.data.uniform_(-0.01, 0.01)
        self.item_embedding.weight.data.uniform_(-0.01, 0.01)

        self.dropout=nn.Dropout(dropout)
        self.relu=nn.ReLU()

        self.linear1=nn.Linear(256, 256)
        self.linear2=nn.Linear(256, 128)  
        self.linear3=nn.Linear(128, 64)
        self.linear4=nn.Linear(64, 32)
        # self.linear5=nn.Linear(32, 16)

        self.fc=nn.Linear(32, output_dim)

    def forward(self, users, items):
        
        features=torch.cat([self.user_embedding(users), self.item_embedding(items)], dim=1)
        x=self.relu(self.linear1(features))
        x=self.relu(self.linear2(x))
        
        x=self.relu(self.linear3(x))
        
        x=self.relu(self.linear4(x))
        # x=self.relu(self.linear5(x))
    
        criteria_ratings=self.fc(self.dropout(x))

        return criteria_ratings