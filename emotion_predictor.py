from transformers import pipeline

class EmotionPredictor:
    def __init__(self, batch_size=64):
        self.classifier = pipeline(
            "text-classification",
            model="j-hartmann/emotion-english-distilroberta-base",
            top_k=None,  
            device=-1  
        )
        self.batch_size = batch_size

    def predict(self, text):
        if not isinstance(text, str) or text.strip() == "":
            return "unknown"
        result = self.classifier(text[:512])[0]
        top_emotion = max(result, key=lambda x: x['score'])['label']
        return top_emotion

    def predict_batch(self, texts):
        cleaned_texts = []
        for text in texts:
            if not isinstance(text, str) or text.strip() == "":
                cleaned_texts.append("unknown")
            else:
                cleaned_texts.append(text[:512])  

        results = []
        for i in range(0, len(cleaned_texts), self.batch_size):
            batch = cleaned_texts[i:i+self.batch_size]
            preds = self.classifier(batch)
            for pred, text in zip(preds, batch):
                if text == "unknown":
                    results.append("unknown")
                else:
                    top_emotion = max(pred, key=lambda x: x['score'])['label']
                    results.append(top_emotion)
        return results
