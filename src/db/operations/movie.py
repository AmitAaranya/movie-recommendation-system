from src.db.model import MovieDb
from src.error import MovieNotFoundError

class MovieOps():
    def __init__(self,session):
        self.session = session

    def add(self,Name, Year, **genres):
        new_movie = MovieDb(Name=Name,Year=Year,**genres)
        try:
            self.session.add(new_movie)
            return new_movie
        except TypeError as te:
            return {"message": "Error initializing movie", "error": str(te)}
        except ValueError as ve:
            return {"message": "Invalid argument value", "error": str(ve)}
        
    def get_all(self):
        movies = MovieDb.query.all()
        return movies
    
    def get(self,Id):
        movie = MovieDb.query.get(Id)
        if movie is None:
            MovieNotFoundError(Id)
        return movie
    
    def get_genres(self,Id):
        movie = self.get(Id)
        genres_fields = [column.name for column in MovieDb.__table__.columns if column.name not in ['Id',"Name",'Year']]

        return [genre for genre in genres_fields if getattr(movie, genre) == 1]
    
       