import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import SGDClassifier
import joblib
import os
import time

os.makedirs('models', exist_ok=True)

def train_sentiment_model(batch_size=None):
    start_time = time.time()

    def load_datasets():
        all_data = []

        try:
            user_df = pd.read_csv('data/usercontent_dataset.csv')
            all_data.append(user_df)
        except Exception as e:
            print(f"Error loading usercontent dataset: {e}")
            raise FileNotFoundError("usercontent_dataset.csv not found!")

        try:
            emotion_df = pd.read_csv('data/emotion_dataset.csv')
            emotion_df = emotion_df.rename(columns={
                col: 'text' if 'text' in col.lower() else col 
                for col in emotion_df.columns
            })
            emotion_df = emotion_df.rename(columns={
                col: 'sentiment' if any(term in col.lower() for term in ['sentiment', 'emotion', 'label']) 
                else col for col in emotion_df.columns
            })
            all_data.append(emotion_df)
        except Exception as e:
            print(f"Unable to load emotion dataset: {str(e)}")

        combined_data = pd.concat(all_data, ignore_index=True)
        return combined_data

    try:
        data = load_datasets()
    except FileNotFoundError as e:
        print(str(e))
        return

    if batch_size:
        data = data.sample(n=min(batch_size, len(data)), random_state=42)

    print(f"Training with dataset of {len(data)} samples")

    def map_sentiment_to_emotion(sentiment, text):
        if sentiment == 'positive':
            if '!' in text or any(word in text.lower() for word in ['excited', 'amazing', 'wonderful']):
                return 'excited'
            elif any(word in text.lower() for word in ['content', 'satisfied', 'pleased']):
                return 'content'
            else:
                return 'happy'
        elif sentiment == 'neutral':
            return 'neutral'
        else:
            if any(word in text.lower() for word in ['angry', 'mad', 'furious']):
                return 'angry'
            elif any(word in text.lower() for word in ['afraid', 'scared', 'anxious']):
                return 'fearful'
            else:
                return 'sad'

    if len(data) > 50000:
        chunk_size = 10000
        emotions = []
        for i in range(0, len(data), chunk_size):
            chunk = data.iloc[i:i+chunk_size]
            chunk_emotions = chunk.apply(lambda row: map_sentiment_to_emotion(row['sentiment'], row['text']), axis=1)
            emotions.extend(chunk_emotions)
            print(f"Processed {min(i+chunk_size, len(data))}/{len(data)} samples")
        data['emotion'] = emotions
    else:
        data['emotion'] = data.apply(lambda row: map_sentiment_to_emotion(row['sentiment'], row['text']), axis=1)

    print("Splitting data into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        data['text'], data['emotion'], test_size=0.2, random_state=42, stratify=data['emotion']
    )

    del data
    import gc
    gc.collect()

    param_grid = {
        'tfidf__max_features': [15000, 20000],
        'tfidf__ngram_range': [(1, 2)],
        'clf__alpha': [0.0001, 0.001],
        'clf__penalty': ['l2']
    }

    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer()),
        ('clf', SGDClassifier(max_iter=1000, tol=1e-3))
    ])

    n_iter = 5 if len(X_train) > 100000 else 10

    random_search = RandomizedSearchCV(
        pipeline, param_grid, n_iter=n_iter, cv=3, scoring='f1_weighted', 
        verbose=2, n_jobs=-1, random_state=42
    )

    print("Training model...")
    random_search.fit(X_train, y_train)
    sentiment_model = random_search.best_estimator_

    print(f"Best parameters: {random_search.best_params_}")
    print("Evaluating model on test data...")
    y_pred = sentiment_model.predict(X_test)
    print(classification_report(y_test, y_pred))

    model_filename = 'models/sentiment_model.pkl'
    print("Saving model...")
    joblib.dump(sentiment_model, model_filename)

    elapsed_time = time.time() - start_time
    print(f"Training completed in {elapsed_time:.2f}s")

    return sentiment_model

if __name__ == "__main__":
    train_sentiment_model()
