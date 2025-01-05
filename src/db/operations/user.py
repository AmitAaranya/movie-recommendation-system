import bcrypt
from sqlalchemy.orm import aliased
from sqlalchemy import exists
from src.db.model import MovieDb
from src.db.model import RatingDb
from src.db.model import UserDb
from .rating import RatingOps
from .movie import MovieOps
from src.error import UserNotFoundError

class UserOps():

    def __init__(self,session):
        self.session = session

    def add(self,name, email,password):
        existing_user = self.session.query(UserDb).filter_by(Email=email).first()
        if existing_user:
            raise ValueError("Email already in use")
        new_user = UserDb(Name=name, Email=email, Password=password)
        self.session.add(new_user)
        return new_user
    
    def delete_user(self, email):
        user = self.session.query(UserDb).filter_by(Email=email).first()
        if not user:
            raise ValueError(f"User with email {email} not found")
        
        self.session.delete(user)
        return user
    
    def get(self, email):
        user = self.session.query(UserDb).filter_by(Email=email).first()
        if not user:
            raise UserNotFoundError(email)
        return user
    
    def get_by_id(self,Id):
        user = self.session.query(UserDb).filter_by(Id=Id).first()
        if not user:
            raise UserNotFoundError(Id)
        return user
        
    def modify(self,email,**kwrgs):
        user = self.session.query(UserDb).filter_by(Email=email).first()
        if not user:
            raise ValueError(f"User with email {email} not found")
        
        fields = [column.name for column in UserDb.__table__.columns \
                  if column.name not in ['Id',"Name","Email",'Password']]
        for field in fields:
            setattr(user, field, kwrgs.get(field, getattr(user, field)))
        return user
    
    def get_rated_movies(self,user_id):
        rated_movie = self.session.query(MovieDb.Id,MovieDb.Name,MovieDb.Year,RatingDb.Rating).join(RatingDb).filter(RatingDb.UserId == user_id).all()
        return [{"Id": movie[0],"Name": movie[1],"Year": movie[2],"Rating":movie[3]} for movie in rated_movie]
    
    def get_non_rated_movies(self,user_id):
        rating_alias = aliased(RatingDb)

        # Query for movies that are not rated by the specific user
        non_rated_movies = self.session.query(MovieDb.Id,MovieDb.Name,MovieDb.Year) \
            .filter(~exists().where(rating_alias.MovieId == MovieDb.Id).where(rating_alias.UserId == user_id)) \
            .all()
        
        return [{"Id": movie[0],"Name": movie[1],"Year": movie[2]} for movie in non_rated_movies]

    def rate_movie(self,email,movie_id,rating):
        user = self.get(email)
        movie_genre = MovieOps(self.session).get_genres(movie_id)
        user_ratings = user.ratings
        user_dict = user.to_dict()
        updated_rating_dict = {}
        for genre in movie_genre:
            if user_ratings:
                rate_count = len([a for a in user_ratings if a.movie.to_dict()[genre] == 1])
                old_avg_rating = user_dict[genre]
                new_avg_rating = (old_avg_rating*rate_count + rating)/(rate_count+1)
                updated_rating_dict[genre] = new_avg_rating
            else:
                updated_rating_dict[genre] = rating
        RatingOps(self.session).add(user.Id,movie_id,rating)
        return self.modify(email=email,**updated_rating_dict)


    def authenticate(self,user:UserDb,password):
        return bcrypt.checkpw(password.encode('utf-8'), user.Password.encode('utf-8'))
