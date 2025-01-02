from .model import LoadModel
from .transform import LoadScalar

import torch

class Ai():
    def __init__(self, model_data_dir:str, user_feature:int, movie_feature:int):
        self.user_array_len = user_feature
        self.movie_array_len = movie_feature
        self.__model = LoadModel(model_data_dir,user_feature,movie_feature)
        self.__scalar = LoadScalar(model_data_dir)


    def predict_rating(self,user_array,movie_array):
        user_ = torch.tensor(self.__scalar.user_transform(user_array),dtype=torch.float32)
        movie_ = torch.tensor(self.__scalar.movie_transform(movie_array),dtype=torch.float32)
        assert user_.shape[1] == self.user_array_len
        assert movie_.shape[1] == self.movie_array_len
        
        with torch.no_grad():
            return self.__model.rating_model(user_,movie_).item()

    def get_user_vector(self,user_array):
        user_ = torch.tensor(self.__scalar.user_transform(user_array),dtype=torch.float32)
        assert user_.shape[1] == self.user_array_len

        with torch.no_grad():
            return self.__model.user_model(user_).tolist()
        
    def get_movie_vector(self,movie_array):
        movie_ = torch.tensor(self.__scalar.movie_transform(movie_array),dtype=torch.float32)
        assert movie_.shape[1] == self.movie_array_len

        with torch.no_grad():
            return self.__model.movie_model(movie_).tolist()