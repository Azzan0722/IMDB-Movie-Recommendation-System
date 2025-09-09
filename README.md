🎬 IMDB Movie Recommendation System Using Storylines
📌 Project Overview



This project focuses on building a content-based movie recommendation system using IMDb 2024 movie data.
We scrape IMDb using Selenium to collect movie names and storylines, preprocess them using NLP techniques, and recommend similar movies based on Cosine Similarity over TF-IDF representations.

The system is deployed with an interactive Streamlit app, where users can input a storyline and receive top 5 recommended movies.



🚀 Skills Gained

Selenium (Web Scraping)

Python & Pandas (Data Analysis & Cleaning)

NLP (Text Preprocessing, Tokenization, TF-IDF)

Machine Learning (Cosine Similarity, Content-Based Recommendation)

Data Visualization (Matplotlib, Seaborn, Streamlit)

Streamlit (User Interface Development)



🎯 Domain

Entertainment / Data Analytics / Recommender Systems



❓ Problem Statement

Most recommendation systems focus on ratings, actors, or genres.
This project builds a storyline-based movie recommender, where the plot summary is used as the main feature for similarity comparison.



👉 Users enter a storyline or select a movie, and the system recommends the 5 most similar movies.



💼 Business Use Cases

Personalized Recommendations – Suggest movies based on preferred storylines.

Entertainment Applications – Enhance streaming platforms with storyline-based filtering.

Movie Discovery – Help users find hidden gems with similar plots.




🔎 Approach Breakdown
1️⃣ Data Scraping

Source: IMDb 2024 movie list

Method: Selenium

Columns: Movie Name, Storyline

Output: movies_2024.csv

2️⃣ Data Preprocessing

Remove stopwords, punctuation, and special characters

Tokenize storylines

Vectorize using TF-IDF

3️⃣ Similarity & Recommendation

Compute Cosine Similarity on TF-IDF matrix

Rank similarity scores

Return Top 5 recommended movies

4️⃣ Streamlit Interface

User inputs a storyline or selects a movie

System shows top 5 recommended movies

Display: Movie Names + Storylines

📊 System Workflow

Scrape IMDb movies using Selenium

Store dataset in CSV

Preprocess with NLP

Apply TF-IDF + Cosine Similarity

Build recommender system

Deploy interactive Streamlit app
