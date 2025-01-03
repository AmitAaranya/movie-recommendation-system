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
            return self.__model.movie_model(movie_)
        
    def get_all_movie_vector(self,movies):
        self.all_movie_vector_dict = {}
        for movie in movies:
            self.all_movie_vector_dict[movie.Id] = self.get_movie_vector(movie.to_array())
        return self.all_movie_vector_dict
    
    def get_cosine_simillarity_score(self,movie_id):
        import torch
        import torch.nn.functional as F

        movie_vector = self.all_movie_vector_dict[movie_id]
        
        similarity_scores = []
        for other_movie_id, other_movie_vector in self.all_movie_vector_dict.items():
            if other_movie_id != movie_id:
                cosine_sim = F.cosine_similarity(movie_vector, other_movie_vector)
                similarity_scores.append((other_movie_id, cosine_sim.item()))
        
        similarity_scores.sort(key=lambda x: x[1], reverse=True)
        
        return similarity_scores



        
