import numpy as np
import pandas as pd
import ast
import nltk
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
movies = pd.read_csv("tmdb_5000_movies.csv")
credits = pd.read_csv("tmdb_5000_credits.csv")

print(movies.head(1))
print(credits.head(1))

movies = movies.merge(credits , on='title')
print(movies)
print(movies.head(1))

# genres , id , keyword , title , over views , release date , cast , crew
movies = movies[['movie_id','title','overview','genres','keywords','cast','crew']]
print(movies.head().shape)

# Missing data
print(movies.isnull().sum())
print(movies.dropna(inplace=True))
print(movies.isnull().sum())

# duplicate data
print(movies.duplicated().sum())
print(movies.iloc[0].genres)


def convert(obj):
    L = []
    for i in ast.literal_eval(obj):
        L.append(i['name'])
    return L
    
movies['genres'] = movies['genres'].apply(convert)
print(movies['genres'].head())
print(movies.head())

movies['keywords']= movies['keywords'].apply(convert)
print(movies['keywords'].head())
print(movies.head())

def convert3(obj):
    L = []
    counter = 0
    for i in ast.literal_eval(obj):
        if counter != 3 :
           L.append(i['name'])
           counter += 1
        else:
           break
    return L

movies['cast'] = movies['cast'].apply(convert3)
print(movies['cast'].head())
print(movies.head())

def fetch_director(obj):
    L = []
    for i in ast.literal_eval(obj):
        if i['job'] == 'Director':
           L.append(i['name'])
           break
    return L

movies['crew'] = movies['crew'].apply(fetch_director)
print(movies['crew'].head())
print(movies.head())

movies['overview']=movies['overview'].apply(lambda x :x.split())
print(movies['overview'].head())
print(movies.head())

movies['genres'] = movies['genres'].apply(lambda x:[i.replace(" ","") for i in x])
movies['keywords'] = movies['keywords'].apply(lambda x:[i.replace(" ","") for i in x])
movies['cast'] = movies['cast'].apply(lambda x:[i.replace(" ","") for i in x])
movies['crew'] = movies['crew'].apply(lambda x:[i.replace(" ","") for i in x])

print(movies['genres'].head())
print(movies.head())
print(movies['cast'].head())
print(movies.head())
print(movies['crew'].head())
print(movies.head())
print(movies['keywords'].head())
print(movies.head())

movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']
print(movies['tags'].head())
print(movies.head())

new_df = movies[['movie_id' , 'title' , 'tags']]
print(new_df)
new_df['tags'] = new_df['tags'].apply(lambda x:" ".join(x))
print(new_df['tags'].head())
print(new_df.head())


new_df['tags'] = new_df['tags'].apply(lambda x:x.lower())
print(new_df['tags'].head())
print(new_df.head())

cv = CountVectorizer(max_features=5000 , stop_words="english")
vectors = cv.fit_transform(new_df['tags']).toarray()
print(vectors)
print((cv.get_feature_names_out()))

import nltk
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()
def stem(text):
    y = []
    for i in text.split():
        y.append(ps.stem(i))
    return " ".join(y)

new_df['tags'] = new_df['tags'].apply(stem)
print(new_df['tags'].head())
print(new_df.head())

from sklearn.metrics.pairwise import cosine_similarity
similarity = cosine_similarity(vectors)
print(similarity)
print(sorted(list(enumerate(similarity[0])),reverse=True , key = lambda x:x[1])[1:6])

def recommend(movie):
    movie_index = new_df[new_df['title'] == movie].index[0]
    distances = similarity[movie_index]
    movies_list = sorted(list(enumerate(distances)),reverse = True , key =lambda x:x[1])[1:6]
    for i in movies_list:
        print(new_df.iloc[i[0]].title)

print(recommend('Batman Begins'))
print(new_df.iloc[1216])
print(new_df['title'].values)

import pickle
print(pickle.dump(new_df , open('movies.pkl','wb')))
print(pickle.dump(new_df.to_dict(),open('movie_dict.pkl','wb')))
print(pickle.dump(similarity,open('similarity.pkl','wb')))
