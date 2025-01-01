from src.db.model import RatingDb

class RatingOps():
    def __init__(self,session):
        self.session = session

    def add(self,user_id, movie_id, rating):
        try:
            new_rating = RatingDb(UserId=user_id,MovieId=movie_id,Rating=rating)
            self.session.add(new_rating)
            return new_rating.Id  
        except TypeError as te:
            return {"message": "Error initializing movie", "error": str(te)}
        except ValueError as ve:
            return {"message": "Invalid argument value", "error": str(ve)}
        
    
    


