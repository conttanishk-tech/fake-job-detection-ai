"""Train the same main prediction pipeline used in the supplied Colab project.
Run: python train_model.py
"""
from pathlib import Path
import re
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

ROOT = Path(__file__).resolve().parent
DATASET = ROOT / 'fake_job_postings.csv'
MODEL_FILE = ROOT / 'job_model.joblib'
VECTORIZER_FILE = ROOT / 'tfidf_vectorizer.joblib'


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'<[^>]*>', ' ', text)
    text = re.sub(r'http\S+|www\S+', ' ', text)
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def main():
    if not DATASET.exists():
        raise SystemExit(
            'Dataset missing! Put fake_job_postings.csv in this folder, then rerun python train_model.py'
        )
    df = pd.read_csv(DATASET)
    required = ['title', 'description', 'company_profile', 'fraudulent']
    missing = [x for x in required if x not in df.columns]
    if missing:
        raise SystemExit('Missing dataset columns: ' + ', '.join(missing))

    data = df[required].copy()
    for col in ['title', 'description', 'company_profile']:
        data[col] = data[col].fillna('')
    data = data.drop_duplicates().reset_index(drop=True)
    data['text'] = data['title'] + ' ' + data['description'] + ' ' + data['company_profile']
    data['clean_text'] = data['text'].apply(clean_text)

    X, y = data['clean_text'], data['fraudulent'].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    tfidf = TfidfVectorizer(stop_words='english', max_features=5000, ngram_range=(1, 2))
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)

    model = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
    model.fit(X_train_tfidf, y_train)
    pred = model.predict(X_test_tfidf)
    print('\nModel evaluation on held-out test data:')
    print(f'Accuracy:  {accuracy_score(y_test, pred):.4f}')
    print(f'Precision: {precision_score(y_test, pred, zero_division=0):.4f}')
    print(f'Recall:    {recall_score(y_test, pred, zero_division=0):.4f}')
    print(f'F1-score:  {f1_score(y_test, pred, zero_division=0):.4f}')

    joblib.dump(model, MODEL_FILE)
    joblib.dump(tfidf, VECTORIZER_FILE)
    print('\nSaved job_model.joblib and tfidf_vectorizer.joblib')
    print('Next run: python -m streamlit run app.py')


if __name__ == '__main__':
    main()
