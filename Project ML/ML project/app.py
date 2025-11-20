import streamlit as st
import pickle
import pandas as pd
import requests

# Fetch poster from TMDB
def fetch_poster(movie_id):
    try:
        # secrets.toml
        tmdb_api_key = "8265bd1679663a7ea12ac168da84d2e8"

        url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={st.secrets['tmdb_api_key']}&language=en-US"
        response = requests.get(url, timeout=3)
        response.raise_for_status()
        data = response.json()
        poster_path = data.get('poster_path')
        return f"https://image.tmdb.org/t/p/w500{poster_path}" if poster_path else "https://via.placeholder.com/500x750?text=No+Image"
    except requests.exceptions.Timeout:
        return "https://via.placeholder.com/500x750?text=Timeout"
    except Exception as e:
        st.warning(f"Error fetching poster: {e}")
        return "https://via.placeholder.com/500x750?text=Error"

# Recommend movies
def recommend(movie):
    movie = movie.strip()
    movie_index_list = movies[movies['title'].str.lower() == movie.lower()].index.tolist()
    if not movie_index_list:
        st.error("Movie not found in database!")
        return [], []
    index = movie_index_list[0]

    distances = sorted(list(enumerate(similarity[index])), reverse=True, key=lambda x: x[1])
    names, posters = [], []

    for i in distances[1:6]:  # top 5 recommendations
        movie_id = movies.iloc[i[0]].movie_id
        posters.append(fetch_poster(movie_id))
        names.append(movies.iloc[i[0]].title)

    return names, posters

# Load data
movies_dict = pickle.load(open('movie_dict.pkl', 'rb'))
movies = pd.DataFrame(movies_dict)
similarity = pickle.load(open('similarity.pkl', 'rb'))

# Streamlit UI
st.title('🎥 Movie Recommender System')

selected_movie_name = st.selectbox(
    'Type or select a movie from the dropdown',
    movies['title'].values
)

if st.button('Show Recommendation'):
    names, posters = recommend(selected_movie_name)
    if names and posters:
        cols = st.columns(len(names))
        for col, name, poster in zip(cols, names, posters):
            with col:
                st.text(name)
                st.image(poster)
