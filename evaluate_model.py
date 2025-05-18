import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from model.emotion_predictor import EmotionPredictor

test_df = pd.read_csv('model/emotion_test_generated.csv')

#Extracting columns
texts = test_df['description'].tolist()
labels = test_df['true_emotion'].tolist()

predictor = EmotionPredictor()

predictions = predictor.predict_batch(texts)

#Evaluation metrics
accuracy = accuracy_score(labels, predictions)
precision = precision_score(labels, predictions, average='weighted', zero_division=0)
recall = recall_score(labels, predictions, average='weighted', zero_division=0)
f1 = f1_score(labels, predictions, average='weighted', zero_division=0)


print("=== Evaluation Results ===")
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")
print("\n=== Classification Report ===")
print(classification_report(labels, predictions, zero_division=0))


os.makedirs('results', exist_ok=True)

#Bar chart
metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
scores = [accuracy, precision, recall, f1]

plt.figure(figsize=(8, 5))
plt.bar(metrics, scores, color=['skyblue', 'lightgreen', 'salmon', 'plum'])
plt.ylim(0, 1)
plt.title('Model Evaluation Metrics')
plt.ylabel('Score')
plt.savefig('results/evaluation_metrics_new.png')  # Save bar chart
plt.close()

#Confusion Matrix
cm = confusion_matrix(labels, predictions, labels=list(set(labels)))
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=list(set(labels)), yticklabels=list(set(labels)))
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.savefig('results/confusion_matrix_new.png')  # Save confusion matrix
plt.close()

print("Evaluation complete. Results saved in the 'results/' folder.")
