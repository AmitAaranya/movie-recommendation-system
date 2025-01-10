import os
from flask import Flask, jsonify, redirect, render_template, request, url_for
from sqlalchemy.exc import SQLAlchemyError

from src.db.operations.user import UserOps
from src.db.operations.movie import MovieOps
from src.db.setup import db

from src.ai.task import Ai
from src.error import MovieNotFoundError, UserNotFoundError

app = Flask(__name__,template_folder=os.path.join('src','templates')\
            ,static_folder=os.path.join('src','static'))

# SQLAlchemy
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///movies.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db.init_app(app)
with app.app_context():
    db.create_all()

# Flask-login
from flask_login import LoginManager, current_user, login_required, login_user, logout_user
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'
app.secret_key = os.urandom(24)

@login_manager.user_loader
def load_user(user_id):
    return UserOps(db.session).get_by_id(user_id)

# AI
AI = Ai(model_data_dir=os.path.join("data","model"),user_feature=14,movie_feature=15)
# AI.get_all_movie_vector(MovieOps(db.session).get_all())

@app.route('/')
@login_required
def home():
    user_id = current_user.Id
    try:
        user_ops = UserOps(db.session)
        rated_movie = user_ops.get_rated_movies(user_id)
        non_rated_movie = user_ops.get_non_rated_movies(user_id)
        user = user_ops.get_by_id(user_id)
        predicted_rating_movie = []
        for movie in non_rated_movie:
            predicted_rating_movie.append({"Id": movie.Id,
                                "Name": movie.Name,
                                "Year": movie.Year,
                                "Rating":round(AI.predict_rating(user.to_array(),movie.to_array())*2)/2})
        predicted_rating_movie.sort(key= lambda x: x['Rating'],reverse=True)
        db.session.commit()
    except SQLAlchemyError as e:
        db.session.rollback()
        return jsonify({"error": "Error retrieving all movies", "error": str(e)}), 500
    return render_template('home.html', rated_movies=rated_movie,recommended_movies=predicted_rating_movie)

@app.route("/ai/rate",methods=['POST'])
def movie_embed():
    data = request.get_json()
    movie = MovieOps(db.session).get(data['movieId'])
    user = UserOps(db.session).get(data['Email'])
    return jsonify(AI.predict_rating(user.to_array(),movie.to_array()))

@app.route('/movie/add', methods=['GET','POST'])
def add_movie():
    if request.method == 'GET':
        return render_template('movie_add.html')
    data = request.get_json()
    movie_ops = MovieOps(db.session)
    for movie in data:
        genres = {genre: 1 for genre in movie['genres']}
        movie_ops.add(Name=movie['name'],Year=int(movie['year']),**genres)
    try:
        db.session.commit()
        return redirect(url_for('home'))
    except Exception as e:
        db.session.rollback()
        return jsonify({"error": "Error adding movie", "error": str(e)}), 500
    

@app.route('/movies', methods=['GET'])
def get_all_movies():
    try:
        movies = MovieOps(db.session).get_all()
        db.session.commit()
        if not movies:
            return jsonify({"error": "No movies found"}), 404
        movie_list = [movie.to_dict() for movie in movies]
        return jsonify(movie_list)
    except SQLAlchemyError as e:
        db.session.rollback()
        return jsonify({"error": "Error retrieving all movies", "error": str(e)}), 500

@app.route('/movies/<int:id>', methods=['GET'])
def get_movie_by_id(id):
    try:
        movie = MovieOps(db.session).get(id)
        db.session.commit()
        return jsonify(movie.to_dict())
    except SQLAlchemyError as e:
        db.session.rollback()
        return jsonify({"error": "Error retrieving movie", "error": str(e)}), 500
    
@app.route('/movie/simillar', methods=['GET'])
def get_movies():
    movies = MovieOps(db.session).get_all()
    db.session.commit()
    if not movies:
        return jsonify({"error": "No movies found"}), 404
    movie_list = [{'Id': movie.Id, "Name": movie.Name,"Year": movie.Year} for movie in movies]
    return render_template('simillar_movie.html', movies=movie_list)

@app.route('/movie/simillar/<int:id>', methods=['GET'])
def get_simillar_movies(id):
    movies = MovieOps(db.session).get_all()
    db.session.commit()
    AI.get_all_movie_vector(MovieOps(db.session).get_all())
    score = AI.get_cosine_simillarity_score(int(id))
    movie_dict = {movie.Id: {"Name": movie.Name,"Year": movie.Year} for movie in movies}
    res_movie_list = [{'Id': a[0], "Name": movie_dict[a[0]]['Name'],\
                       "Year": movie_dict[a[0]]['Year'],'score': a[1]} for a in score]
    return jsonify({'related_movies': res_movie_list})
    
@app.route('/register',methods = ["GET"])
def add_user():
    try:
        movies = MovieOps(db.session).get_all()
        db.session.commit()
        if not movies:
            return jsonify({"error": "No movies found"}), 404
        movie_list = [{'Id': movie.Id, "Name": movie.Name,"Year": movie.Year} for movie in movies]
    except SQLAlchemyError as e:
        db.session.rollback()
        return jsonify({"error": "Error retrieving all movies", "error": str(e)}), 500
    return render_template('register.html', movies=movie_list)

@app.route('/register', methods=['POST'])
def register_user():
    data = request.get_json()
    user_ops = UserOps(db.session)
    new_user = user_ops.add(data.get('Name'),data.get('Email'),data.get('Password'))
    for movie_id,rating in data['Ratings'].items():
        user_ops.rate_movie(data.get('Email'),movie_id,float(rating))
    try:
        db.session.commit()
        return jsonify({"success": "User Created successfully", "Id": new_user.Email}), 201
    except Exception as e:
        db.session.rollback()
        return jsonify({"error": "Error While Creating User", "error": str(e)}), 500
    
@app.route('/login', methods=['GET'])
def login():
    return render_template('login.html')

@app.route('/login', methods=['POST'])
def login_post():
    data = request.get_json()
    user_ops = UserOps(db.session)
    user = user_ops.get(data.get('Email'))
    if user_ops.authenticate(user,data.get('Password')):
        login_user(user)
        return jsonify({"success": "Log In"}), 201
    return jsonify({"error": "something wrong"}), 500
    
@app.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))
    
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

@app.route('/user/about',methods=['GET'])
@login_required
def about():
    user_details = UserOps(db.session).get_by_id(current_user.Id).to_dict()
    del user_details['Id']
    del user_details['Password']
    Name = user_details.pop("Name")
    Email = user_details.pop("Email")
    user = {"Name": Name,
            "Email": Email,
            "Ratings":user_details
            }
    return render_template('about.html', user = user)
    

@app.route("/movie/rate",methods = ['POST'])
@login_required
def rate_movies():
    data = request.get_json()
    user_ops = UserOps(db.session)
    for movie in data['movies']:
        user_ops.rate_movie(current_user.Email,movie['MovieId'],movie["Rating"])
    try:
        db.session.commit()
        return jsonify({'success': "Movie Ratings Saved"}), 201
    except Exception as e:
        db.session.rollback()
        return jsonify({"error": str(e)}), 500

@app.errorhandler(Exception)
def handle_not_found_error(error):
    if isinstance(error, MovieNotFoundError):
        return jsonify({"error": error.message}), 404
    elif isinstance(error, UserNotFoundError):
        return jsonify({"error": error.message}), 404
    else:

        return jsonify({"error": f"{error}"}), 500
    
if __name__ == "__main__":
    app.run(debug=True)
    # app.run()
