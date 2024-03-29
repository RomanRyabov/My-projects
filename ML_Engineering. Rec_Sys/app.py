import warnings
warnings.filterwarnings("ignore")


import streamlit as st
import numpy as np
import pandas as pd
import lightfm as lf
import nmslib
import pickle
import scipy.sparse as sparse
import plotly.express as px

@st.cache
def read_files(folder_name='data'):
    """
    Функция для чтения файлов.
    Возвращает два DataFrame с рейтингами и характеристиками книг.
    """
    ratings = pd.read_csv(folder_name + '/ratings.csv')
    books = pd.read_csv(folder_name + '/books.csv')
    return ratings, books

def make_mappers(books):
    """
    Функция для создания отображения id в title и authors.
    Возвращает два словаря:
    * Ключи первого словаря — идентификаторы книг, значения — их названия.
    * Ключи второго словаря — идентификаторы книг, значения — их авторы.
    """
    name_mapper = dict(zip(books.book_id, books.title))
    author_mapper = dict(zip(books.book_id, books.authors))

    return name_mapper, author_mapper

def load_embeddings(file_name='item_embeddings.pkl'):
    """
    Функция для загрузки векторных представлений.
    Возвращает прочитанные эмбеддинги книг и индекс (граф) для поиска похожих книг.
    """
    with open(file_name, 'rb') as f:
        item_embeddings = pickle.load(f)

    nms_idx = nmslib.init(method='hnsw', space='cosinesimil')
    nms_idx.addDataPointBatch(item_embeddings)
    nms_idx.createIndex(print_progress=True)
    return item_embeddings, nms_idx

def nearest_books_nms(book_id, index, n=10):
    """
    Функция для поиска ближайших соседей, возвращает построенный индекс.
    Возвращает n наиболее похожих книг и расстояние до них.
    """
    nn = index.knnQuery(item_embeddings[book_id], k=n)
    return nn

def get_recomendation_df(ids, distances, name_mapper, author_mapper):
    """
    Функция для составления таблицы из рекомендованных книг.
    Возвращает DataFrame со столбцами:
    * book_name — название книги;
    * book_author — автор книги;
    * distance — значение метрики расстояния до книги.
    """
    names = []
    authors = []
    
    for idx in ids:
        names.append(name_mapper[idx])
        authors.append(author_mapper[idx])
    
    recomendation_df = pd.DataFrame({'book_name': names, 'book_author': authors, 'distance': distances})
    return recomendation_df


ratings, books = read_files(folder_name='data')
name_mapper, author_mapper = make_mappers(books)
item_embeddings, nms_idx = load_embeddings()

st.title("Recommendation System Of Books")

st.markdown("""Welcome to the web page of the book recommendation app!
This application is a prototype of a recommendation system based on a machine learning model.

To use the application, you need:
1. Enter the approximate name of the book you like
2. Select its exact name in the pop-up list of books
3. Specify the number of books you need to recommend

After that, the application will give you a list of books most similar to the book you specified""")

title = st.text_input('Please enter book name', '')
title = title.lower()

output = books[books['title'].apply(lambda x: x.lower().find(title)) >= 0]


option = st.selectbox("Select the book you need", output['title'].values)


if option:
    
    st.markdown('You selected: "{}"'.format(option))

    count_recomendation = st.number_input(
        label="Specify the number of recommendations you need",
        value=10
    )  

    ids, distances = nearest_books_nms(val_index, nms_idx, count_recomendation+1)
    
    ids, distances = ids[1:], distances[1:]

    st.markdown('Most simmilar books are: ')

    df = get_recomendation_df(ids, distances, name_mapper, author_mapper)
    
    st.dataframe(df[['book_name', 'book_author']])
    
    
    fig = px.bar(
        data_frame=df,
        x='book_name',
        y='distance',
        hover_data=['book_author'],
        title='Cosine distance to the nearest books'
    )

    st.write(plot_fig)
