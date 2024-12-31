from flask import Flask, jsonify, request
from sqlalchemy.exc import SQLAlchemyError
from src.db.setup import db
from src.db.model import Movie, User
app = Flask(__name__)

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///movies.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize SQLAlchemy
db.init_app(app)

with app.app_context():
    db.create_all()

@app.route("/")
def home():
    return "Hello World"

@app.route('/movies', methods=['POST'])
def add_movie():
    data = request.get_json()
    new_movie = Movie(**data)
    try:
        db.session.add(new_movie)
        db.session.commit()
        return jsonify({"message": "Movie added successfully", "Id": new_movie.Id}), 201
    except Exception as e:
        db.session.rollback()
        return jsonify({"message": "Error adding movie", "error": str(e)}), 500
    

@app.route('/movies', methods=['GET'])
def get_all_movies():
    try:
        movies = Movie.query.all()
        if not movies:
            return jsonify({"message": "No movies found"}), 404
        movie_list = [movie.to_dict() for movie in movies]
        return jsonify(movie_list)
    except SQLAlchemyError as e:
        db.session.rollback()
        return jsonify({"message": "Error retrieving movies", "error": str(e)}), 500

@app.route('/movies/<int:id>', methods=['GET'])
def get_movie_by_id(id):
    try:
        movie = Movie.query.get(id)
        if movie is None:
            return jsonify({"message": f"Movie with ID {id} not found"}), 404
        return jsonify(movie.to_dict())
    except SQLAlchemyError as e:
        db.session.rollback()
        return jsonify({"message": "Error retrieving movie", "error": str(e)}), 500

@app.route('/register', methods=['POST'])
def register_user():
    data = request.get_json()
    new_user = User(data.get('Name'),data.get('Email'),data.get('Password'))
    try:
        db.session.add(new_user)
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
    
    user = User.query.filter_by(Email=email).first()
    if not user:
        return jsonify({"error": "User not found"}), 404
    
    user_data = user.to_dict()
    return jsonify(user_data)


if __name__ == "__main__":
    app.run(debug=True)
