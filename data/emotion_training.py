import pandas as pd
from transformers import pipeline

df = pd.read_csv('data/books.csv')

sample_df = df.sample(n=30, random_state=42).reset_index(drop=True)

classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", return_all_scores=True)

def predict_emotion(text):
    if pd.isna(text) or text.strip() == "":
        return "unknown"
    result = classifier(text[:512])[0]
    top_emotion = max(result, key=lambda x: x['score'])['label']
    return top_emotion

sample_df['true_emotion'] = sample_df['description'].apply(predict_emotion)

sample_df[['description', 'true_emotion']].to_csv('data/emotions_test.csv', index=False)

print("Generated realistic emotions_test.csv ✅")
