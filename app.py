import os
from flask import Flask, jsonify, request
from sqlalchemy.exc import SQLAlchemyError

from src.db.operations.user import UserOps
from src.db.operations.movie import MovieOps
from src.db.setup import db

from src.ai.task import Ai
from src.error import MovieNotFoundError, UserNotFoundError

app = Flask(__name__)

# SQLAlchemy
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///movies.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db.init_app(app)
with app.app_context():
    db.create_all()

# AI
AI = Ai(model_data_dir=os.path.join("data","model"),user_feature=14,movie_feature=15)
@app.route("/")
def home():
    return "Hello World"

@app.route("/ai/rate",methods=['POST'])
def movie_embed():
    data = request.get_json()
    movie = MovieOps(db.session).get(data['movieId'])
    user = UserOps(db.session).get(data['Email'])

    return jsonify(AI.predict_rating(user.to_array(),movie.to_array()))

@app.route('/movies', methods=['POST'])
def add_movie():
    data = request.get_json()
    movie = MovieOps(db.session).add(**data)
    try:
        db.session.commit()
        return jsonify({"message": "Movie added successfully", "Id": movie.Id}), 201
    except Exception as e:
        db.session.rollback()
        return jsonify({"message": "Error adding movie", "error": str(e)}), 500
    

@app.route('/movies', methods=['GET'])
def get_all_movies():
    try:
        movies = MovieOps(db.session).get_all()
        db.session.commit()
        if not movies:
            return jsonify({"message": "No movies found"}), 404
        movie_list = [movie.to_dict() for movie in movies]
        return jsonify(movie_list)
    except SQLAlchemyError as e:
        db.session.rollback()
        return jsonify({"message": "Error retrieving all movies", "error": str(e)}), 500

@app.route('/movies/<int:id>', methods=['GET'])
def get_movie_by_id(id):
    try:
        movie = MovieOps(db.session).get(id)
        db.session.commit()
        return jsonify(movie.to_dict())
    except SQLAlchemyError as e:
        db.session.rollback()
        return jsonify({"message": "Error retrieving movie", "error": str(e)}), 500

@app.route('/register', methods=['POST'])
def register_user():
    data = request.get_json()
    new_user = UserOps(db.session).add(data.get('Name'),data.get('Email'),data.get('Password'))
    try:
        db.session.commit()
        return jsonify({"message": "User Created successfully", "Id": new_user.Email}), 201
    except Exception as e:
        db.session.rollback()
        return jsonify({"message": "Error While Creating User", "error": str(e)}), 500
    
@app.route('/user', methods=['GET'])
def get_user():
    email = request.args.get('Email')  
    if not email:
        return jsonify({"error": "Email is required"}), 400
    
    user = UserOps(db.session).get(request.args.get('Email'))
    db.session.commit()
    if not user:
        return jsonify({"error": "User not found"}), 404
    user_data = user.to_dict()
    return jsonify(user_data)

@app.route("/rate",methods = ['POST'])
def rate_movies():
    data = request.get_json()
    user = UserOps(db.session).rate_movie(data['Email'],data['MovieId'],data["Rating"])
    try:
        db.session.commit()
        return jsonify(user.to_dict()), 201
    except Exception as e:
        db.session.rollback()
        return jsonify({"message": "Rate a movie", "error": str(e)}), 500


@app.errorhandler(Exception)
def handle_not_found_error(error):
    if isinstance(error, MovieNotFoundError):
        return jsonify({"error": error.message}), 404
    elif isinstance(error, UserNotFoundError):
        return jsonify({"error": error.message}), 404
    else:

        return jsonify({"error": f"{error}"}), 500
    
if __name__ == "__main__":
    # app.run(debug=True)
    app.run()
