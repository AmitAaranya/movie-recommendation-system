from datetime import datetime
from sqlalchemy import String
from sqlalchemy.orm import validates
import bcrypt
from .setup import db

class BaseModel():
    def to_dict(self):
        return {column.name: getattr(self, column.name) for column in self.__table__.columns}


class Movie(db.Model,BaseModel):
    __tablename__ = 'movies'

    Id = db.Column(db.Integer, primary_key=True,autoincrement=True)
    Name = db.Column(String(255), nullable=False)
    Year = db.Column(db.Integer, nullable=False)
    Adventure = db.Column(db.Integer, default=0, nullable=False)
    Animation = db.Column(db.Integer, default=0, nullable=False)
    Children = db.Column(db.Integer, default=0, nullable=False)
    Comedy = db.Column(db.Integer, default=0, nullable=False)
    Fantasy = db.Column(db.Integer, default=0, nullable=False)
    Romance = db.Column(db.Integer, default=0, nullable=False)
    Drama = db.Column(db.Integer, default=0, nullable=False)
    Action = db.Column(db.Integer, default=0, nullable=False)
    Crime = db.Column(db.Integer, default=0, nullable=False)
    Thriller = db.Column(db.Integer, default=0, nullable=False)
    Horror = db.Column(db.Integer, default=0, nullable=False)
    Mystery = db.Column(db.Integer, default=0, nullable=False)
    SciFi = db.Column(db.Integer, default=0, nullable=False)
    Documentary = db.Column(db.Integer, default=0, nullable=False)

    def __init__(self,Name, Year, Adventure=0, Animation=0, Children=0, Comedy=0, Fantasy=0, Romance=0, Drama=0, Action=0, Crime=0, Thriller=0, Horror=0, Mystery=0, SciFi=0, Documentary=0):
        self.Name = Name
        self.Year = Year
        self.Adventure = Adventure
        self.Animation = Animation
        self.Children = Children
        self.Comedy = Comedy
        self.Fantasy = Fantasy
        self.Romance = Romance
        self.Drama = Drama
        self.Action = Action
        self.Crime = Crime
        self.Thriller = Thriller
        self.Horror = Horror
        self.Mystery = Mystery
        self.SciFi = SciFi
        self.Documentary = Documentary

    @validates('Year')
    def validate_year(self, key, value):
        if not (1500 <= value <= datetime.now().year):  # Use current year if needed
            raise ValueError(f"Year must be between 1500 and {datetime.now().year}")
        return value

    @validates('Adventure', 'Animation', 'Children', 'Comedy', 'Fantasy', 'Romance', 
               'Drama', 'Action', 'Crime', 'Thriller', 'Horror', 'Mystery', 'SciFi', 'Documentary')
    def validate_binary_values(self, key, value):
        if value not in [0, 1]:
            raise ValueError(f"{key} must be either 0 or 1")
        return value



class User(db.Model,BaseModel):
    __tablename__ = 'users'

    Id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    Name = db.Column(String(255), nullable=False)
    Email = db.Column(String(255), unique=True, nullable=False) 
    Password = db.Column(String(255), nullable=False)
    Adventure = db.Column(db.Float, nullable=False)
    Animation = db.Column(db.Float, nullable=False)
    Children = db.Column(db.Float, nullable=False)
    Comedy = db.Column(db.Float, nullable=False)
    Fantasy = db.Column(db.Float, nullable=False)
    Romance = db.Column(db.Float, nullable=False)
    Drama = db.Column(db.Float, nullable=False)
    Action = db.Column(db.Float, nullable=False)
    Crime = db.Column(db.Float, nullable=False)
    Thriller = db.Column(db.Float, nullable=False)
    Horror = db.Column(db.Float, nullable=False)
    Mystery = db.Column(db.Float, nullable=False)
    SciFi = db.Column(db.Float, nullable=False)
    Documentary = db.Column(db.Float, nullable=False)

    def __init__(self,Name, Email, Password, Adventure=0, Animation=0, Children=0, Comedy=0, Fantasy=0, Romance=0, Drama=0, Action=0, Crime=0, Thriller=0, Horror=0, Mystery=0, SciFi=0, Documentary=0):
        self.Name = Name
        self.Email = Email
        self.Password = bcrypt.hashpw(Password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
        self.Adventure = Adventure
        self.Animation = Animation
        self.Children = Children
        self.Comedy = Comedy
        self.Fantasy = Fantasy
        self.Romance = Romance
        self.Drama = Drama
        self.Action = Action
        self.Crime = Crime
        self.Thriller = Thriller
        self.Horror = Horror
        self.Mystery = Mystery
        self.SciFi = SciFi
        self.Documentary = Documentary

    @validates('Adventure', 'Animation', 'Children', 'Comedy', 'Fantasy', 'Romance', 
               'Drama', 'Action', 'Crime', 'Thriller', 'Horror', 'Mystery', 'SciFi', 'Documentary')
    def validate_ratings(self, key, value):
        if not (0 <= value <= 5):
            raise ValueError(f"Invalid rating for {key}: {value}. It must be between 0 and 5.")
        return value
    
    @validates('Email')  # Validate email format
    def validate_email(self, key, value):
        import re
        if not re.match(r"[^@]+@[^@]+\.[^@]+", value):
            raise ValueError(f"Invalid email format: {value}")
        return value
    


