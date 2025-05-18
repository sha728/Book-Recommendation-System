import pandas as pd
from model.emotion_predictor import EmotionPredictor

def recommend_books(user_emotion, top_n=5):
    df = pd.read_csv('data/books.csv')
    predictor = EmotionPredictor()
    
    if 'predicted_emotion' not in df.columns:
        df['predicted_emotion'] = predictor.predict_batch(df['description'].tolist())
        df.to_csv('data/books.csv', index=False) 

    recommended = df[df['predicted_emotion'] == user_emotion]
    recommended = recommended.sort_values(by='average_rating', ascending=False).head(top_n)
    return recommended[['title', 'authors', 'average_rating']]

if __name__ == "__main__":
    emotion = input("Enter your mood/emotion (e.g., joy, sadness, anger): ")
    books = recommend_books(emotion)
    print(books)
