import numpy as np
import pandas as pd
import re

def load_user_data():
    # users.dat
    users_label = ['UserID', 'Gender', 'Age', 'OccupationID', 'Zip-code']
    users = pd.read_table('./ml-1m/users.dat',
                        sep='::',
                        header=None,
                        names=users_label,
                        engine = 'python')
    users=users.filter(regex='Gender|Age|OccupationID')
    # user_value=users.values

    #embedding
    gender_map={'M':1,'F':0}
    users['Gender'] = users['Gender'].map(gender_map)

    age_map={1: 1, 35: 4, 45: 5, 50: 6, 18: 2, 56: 7, 25: 3}

    users['Age']=users['Age'].map(age_map)

    ratings_title = ['UserID', 'MovieID', 'ratings', 'timestamps']
    ratings = pd.read_table('./ml-1m/ratings.dat', sep='::', header=None, names=ratings_title, engine='python')
    ratings = ratings.filter(regex='UserID|MovieID|ratings')

    return users,ratings


def load_movie_data():
    movie_label={'MovieID','Title','Genres'}
    movie=pd.read_table('./ml-1m/movies.dat',
                        sep='::',
                        header=None,
                        names=movie_label,
                        engine='python')
    movies = []
    random = np.random.randint(1, 200, 10)


    for i in range(0, 10):
        movies.append(movie.iloc[random[i], :])

    return movie,movies

#movie=load_movie_data()

