import joblib
import os
import numpy as np

class LoadScalar:
    def __init__(self,model_data_dir):
        self.movie_scaler = joblib.load(os.path.join(model_data_dir,'movie_scaler.pkl'))
        self.user_scaler = joblib.load(os.path.join(model_data_dir,'user_scaler.pkl'))

    def movie_transform(self,data):
        try:
            data = np.array(data,dtype=np.float32)
            data[0] = self.movie_scaler.transform(np.array(data[0]).reshape(-1,1)).flatten()[0]
            return data.reshape(1,-1)
        except ValueError as ve:
            raise ValueError(f"Error: {ve}")
        except Exception as e:
            return {"message": "something wrong", "error": str(e)} 
    
    def user_transform(self,data):
        try:
            data = np.array(data,dtype=np.float32).reshape(1,-1)
            return self.user_scaler.transform(data)
        except ValueError as ve:
            raise ValueError(f"Error: {ve}")
        except Exception as e:
            return {"message": "something wrong", "error": str(e)} 

    def rating_transform(self,data):
        try:
            return np.array(data).reshape(1,-1)/5
        except ValueError as ve:
            raise ValueError(f"Error: {ve}")
        except Exception as e:
            return {"message": "something wrong", "error": str(e)} 