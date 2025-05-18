import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import pandas as pd
from recommender.recommend import recommend_books
from io import BytesIO
import base64

st.set_page_config(page_title="Emotion-based Book Recommender", layout="wide", page_icon="📚")

mode = st.sidebar.selectbox("Choose Mode", ["Dark", "Light"])
if mode == "Dark":
    st.markdown("""
        <style>
            body { background-color: #111827; color: white; }
            .stCard { background-color: #1f2937; border-radius: 12px; padding: 20px; margin-bottom: 15px; box-shadow: 0 4px 8px rgba(0,0,0,0.3); transition: all 0.3s ease; }
            .stCard:hover { transform: scale(1.02); }
        </style>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
        <style>
            body { background-color: #f9fafb; color: black; }
            .stCard { background-color: #ffffff; border-radius: 12px; padding: 20px; margin-bottom: 15px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); transition: all 0.3s ease; }
            .stCard:hover { transform: scale(1.02); }
        </style>
    """, unsafe_allow_html=True)

st.sidebar.header("🔪 Customize Your Recommendation")
emotion = st.sidebar.selectbox("Select your mood/emotion:", ["joy", "sadness", "anger", "fear", "surprise"])
top_n = st.sidebar.slider("How many book recommendations?", 1, 20, 5)

if st.sidebar.button("Recommend 📚"):
    books = recommend_books(emotion, top_n)

    st.success(f"🎉 Top {top_n} book recommendations for mood: '{emotion.capitalize()}'")

    search_term = st.text_input("Search in recommendations")
    if search_term:
        books = books[books['title'].str.contains(search_term, case=False)]

    def convert_df(df):
        return df.to_csv(index=False).encode('utf-8')

    csv = convert_df(books)
    st.download_button(
        label="📥 Download Recommendations as CSV",
        data=csv,
        file_name='book_recommendations.csv',
        mime='text/csv',
    )

    for _, row in books.iterrows():
        with st.container():
            st.markdown(f"""
            <div class='stCard'>
                <h3>📚 {row['title']}</h3>
                <p><strong>Author:</strong> {row['authors']}</p>
                <p><strong>Average Rating:</strong> ⭐ {row['average_rating']}</p>
            </div>
            """, unsafe_allow_html=True)

else:
    st.info("📝 Start by selecting an emotion and clicking Recommend!")
