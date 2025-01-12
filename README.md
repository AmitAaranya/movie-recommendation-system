# Movie-Recommendation-System
[![EC2 Deployment](https://github.com/AmitAaranya/movie-recommendation-system/actions/workflows/ec2-deploy.yml/badge.svg?branch=main)](https://github.com/AmitAaranya/movie-recommendation-system/actions/workflows/ec2-deploy.yml)
[![site](https://img.shields.io/badge/Docker-hub-0088CC?logo=docker)](https://hub.docker.com/r/amitaaranya/movie-recommendation-system/tags)
[![site](https://img.shields.io/badge/Website-%F0%9F%8C%8D%20home-%23333)](http://3.111.58.65/)

The Movie Recommendation System is a web application designed to provide personalized movie suggestions based on user preferences and past reviews. Utilizing reinforcement machine learning techniques such as content-based filtering, and collaborative filtering, the system predicts movie ratings and recommends similar films. It also features user authentication, a movie rating system, and tailored recommendations.

## Technologies Used
- Backend: `Flask` (`Python`)
- Frontend: `Flask` (`HTML`, `CSS`, `JavaScript`)
- Database: `SQLite` (`SQLAlchemy`)
- ML Tools: `PyTorch`, `Scikit-learn`, `Pandas`, `NumPy`
- ML Techniques: `Reinforcement Learning`, `Content-Based Filtering`, `Collaborative Filtering`

## Features
1. **MovieMatch**: Content-Based filtering, Allows users to find similar movies by selecting any movie from the list, enhancing their discovery of content based on preferences.
2. **Recommended Movies for Users**:Implements collaborative filtering techniques to predict a user's rating for a movie they have not yet rated, using past review data and user similarities.
3. **Login and Registration**:  Enables users to create accounts and log in, ensuring personalized movie recommendations and ratings.
4. **User Rating System**: User Rating System: Users can rate movies, contributing to the accuracy of the recommendation system.


## Machine Leanring Implementation
1. **Exploratory Data Analysis (EDA)**: Analyzed the movie dataset to understand patterns, distributions, and trends in ratings and genres.
2. **Feature Engineering**: Processed raw data to create meaningful features like average rating by user for each genre, improving the performance of the recommendation model.
3. **Reinforcement learning based model**: Used content filtering based reinforcement learning to dynamically adjust recommendations based on user interactions and preferences.
4. **User / Movie Model**: Created separate models for users and movies, enabling the prediction of movie ratings, finding similar movies, and identifying related users.
```py
class CommonNN(nn.Module):
    def __init__(self,feature_count):
        super(CommonNN,self).__init__()

        self.fc1 = nn.Linear(feature_count,64)
        self.fc2 = nn.Linear(64,128)
        self.fc3 = nn.Linear(128,32)

    def forward(self,x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)
    
class RecommenderNN(nn.Module):
    def __init__(self,user_feature_no,movie_feature_no):
        super(RecommenderNN,self).__init__()
        self.UserNN = CommonNN(user_feature_no)
        self.MovieNN = CommonNN(movie_feature_no)
        
    def forward(self,user_train,movie_train):
        out =  torch.sum(self.UserNN(user_train) * self.MovieNN(movie_train),dim=1).reshape(-1,1)
        return torch.sigmoid(out)
```

## Deployment
1. **Flask Web Application**: Built a Flask-based backend that serves the recommendation engine and handles user interactions.
2. **CI/CD with GitHub Actions**: Automated testing and deployment using GitHub Actions, ensuring continuous delivery and streamlined updates.
3. **Docker**: Containerized the application to ensure consistent execution across different environments and simplify deployment.
4. **Deployed on AWS EC2**: Hosted the application on an EC2 instance through docker.


## Installation

### Prerequisites
- Python 3.x
- Flask
- Dependencies mentioned in `requirements.txt`
```sh
Flask==3.1.0
Flask-Login==0.6.3
SQLAlchemy==2.0.36
Flask-SQLAlchemy==3.1.1
bcrypt==4.2.1
torch==2.5.1
joblib==1.4.2
numpy==2.0.2
scikit-learn==1.6.0
```

### Steps to Run the Application
1. Clone the repository:
```sh
git clone https://github.com/AmitAaranya/movie-recommendation-system.git
```
2. Navigate to the project directory:
```sh
cd movie-recommendation-system
```
3. Install required dependencies:
```sh
pip install -r requirements.txt
```
4. Run the application:
```sh
python app.py
```
5. Open your browser and navigate to `http://127.0.0.1:8000/` to start using the application.


## References
1. [MovieLens Latest Datasets](https://grouplens.org/datasets/movielens/latest/)
2. [PyTorch documentation](https://pytorch.org/docs/stable/index.html)
3. [Coursera: Unsupervised Learning, Recommenders, Reinforcement Learning](https://www.coursera.org/learn/unsupervised-learning-recommenders-reinforcement-learning)