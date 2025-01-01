import torch
import torch.nn as nn

class CommonNN(nn.Module):
    def __init__(self,feature_count):
        super(CommonNN,self).__init__()

        self.fc1 = nn.Linear(feature_count,64)
        self.fc2 = nn.Linear(64,128)
        self.fc3 = nn.Linear(128,32)

    def forward(self,x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)
    

class RecommenderNN(nn.Module):
    def __init__(self,user_feature_no,movie_feature_no):
        super(RecommenderNN,self).__init__()
        self.UserNN = CommonNN(user_feature_no)
        self.MovieNN = CommonNN(movie_feature_no)
        
    def forward(self,user_train,movie_train):
        return torch.sum(self.UserNN(user_train) * self.MovieNN(movie_train),dim=1).reshape(-1,1)