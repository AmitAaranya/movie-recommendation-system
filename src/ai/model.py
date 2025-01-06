import torch
import torch.nn as nn
import os

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
        out =  torch.sum(self.UserNN(user_train) * self.MovieNN(movie_train),dim=1).reshape(-1,1)
        return torch.sigmoid(out)
    


class LoadModel:
    def __init__(self,model_data_dir, user_feature:int, movie_feature:int):
        self.user_feature = user_feature
        self.movie_feature = movie_feature
        self.rating_model = self.__rating_model(os.path.join(model_data_dir,"rating_v0.pth"))
        self.user_model = self.__user_model(os.path.join(model_data_dir,"user_v0.pth"))
        self.movie_model = self.__movie_model(os.path.join(model_data_dir,"movie_v0.pth"))
        
    def __rating_model(self,model_path):
        model = RecommenderNN(self.user_feature, self.movie_feature)
        model.load_state_dict(torch.load(model_path))
        model.eval()
        return model
    
    def __user_model(self,model_path):
        model = CommonNN(self.user_feature)
        model.load_state_dict(torch.load(model_path))
        model.eval()
        return model
    
    def __movie_model(self,model_path):
        model = CommonNN(self.movie_feature)
        model.load_state_dict(torch.load(model_path))
        model.eval()
        return model