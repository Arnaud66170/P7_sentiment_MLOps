import os
import re
import pickle
import emoji
import joblib
import numpy as np
import pandas as pd
import spacy
import nltk
nltk.download('vader_lexicon')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.utils import resample
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from joblib import Parallel, delayed
import fasttext
import tensorflow_hub as hub
from transformers import DistilBertTokenizerFast
from datasets import Dataset, load_from_disk
from utils import mlflow_run_safety

nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
nltk.download('stopwords')
nltk.download('wordnet')
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
analyzer = SentimentIntensityAnalyzer()


@mlflow_run_safety(experiment_name="P7_sentiment_analysis")
def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'@[A-Za-z0-9_]+', '', text)
    text = emoji.replace_emoji(text, replace='')
    text = re.sub(r'[^a-zA-Z ]', '', text)
    text = ' '.join([word for word in text.split() if len(word) > 1])
    return text


@mlflow_run_safety(experiment_name="P7_sentiment_analysis")
def lemmatize_text(text):
    doc = nlp(text)
    return ' '.join([token.lemma_ for token in doc if not token.is_stop])


@mlflow_run_safety(experiment_name="P7_sentiment_analysis")
def preprocess_batch(batch_df):
    batch_df['text'] = batch_df['text'].astype(str).apply(clean_text)
    batch_df['text'] = batch_df['text'].apply(lemmatize_text)
    return batch_df


@mlflow_run_safety(experiment_name="P7_sentiment_analysis")
def preprocess_tweets_parallel(df, filename = "../models_saved/cleaned_tweets.pkl", n_jobs = -1, batch_size = 50000):
    if os.path.exists(filename):
        print(f"✅ Chargement des tweets nettoyés depuis {filename}")
        return pd.read_pickle(filename)
    print("🚀 Nettoyage des tweets en cours...")
    batches = [df.iloc[i:i + batch_size] for i in range(0, len(df), batch_size)]
    cleaned_batches = Parallel(n_jobs=n_jobs)(delayed(preprocess_batch)(batch) for batch in batches)
    df_cleaned = pd.concat(cleaned_batches, ignore_index = True)
    df_cleaned.to_pickle(filename)
    print(f"✅ Tweets nettoyés sauvegardés dans {filename}")
    return df_cleaned


@mlflow_run_safety(experiment_name="P7_sentiment_analysis")
def compute_vader_scores(df, text_column = "text", save_path = "../models_saved/vader_scores.pkl"):
    if os.path.exists(save_path):
        print(f"✅ Scores VADER chargés depuis {save_path}...")
        return joblib.load(save_path)
    print("🔄 Calcul des scores VADER...")
    scores = df[text_column].astype(str).apply(lambda x: analyzer.polarity_scores(x)["compound"])
    joblib.dump(scores, save_path)
    print(f"✅ Scores VADER sauvegardés dans {save_path}.")
    return scores


@mlflow_run_safety(experiment_name="P7_sentiment_analysis")
def save_tweets_for_fasttext(X_text_full, filename = "../models_saved/tweets_fasttext.txt"):
    with open(filename, 'w', encoding = 'utf-8') as f:
        for tweet in X_text_full:
            f.write(tweet + '\n')
    print(f"✅ Fichier {filename} créé avec succès.")



# Vectorisation des tweets
@mlflow_run_safety(experiment_name="P7_sentiment_analysis")
def vectorize_tweets(X_text_full, X_text_reduced, y_full, y_reduced):
    print("🚀 Vectorisation BoW et TF-IDF en cours...")
    count_vectorizer = CountVectorizer()
    X_bow = count_vectorizer.fit_transform(X_text_full)

    tfidf_vectorizer = TfidfVectorizer(max_features=2000)
    X_tfidf = tfidf_vectorizer.fit_transform(X_text_full)

    # === FASTTEXT ===
    fasttext_file = "../models_saved/tweets_fasttext.txt"
    if not os.path.exists(fasttext_file):
        save_tweets_for_fasttext(X_text_full)

    print("🚀 Entraînement du modèle FastText...")
    fasttext_model = fasttext.train_unsupervised(fasttext_file, model='skipgram', dim=300)
    X_fasttext = np.array([fasttext_model.get_sentence_vector(text) for text in X_text_full])

    # === USE ===
    print("🚀 Chargement Universal Sentence Encoder...")
    use_model = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
    X_use = np.array([use_model([text]).numpy().flatten() for text in X_text_reduced])

    print("✅ Vectorisation terminée.")
    return X_bow, X_tfidf, X_fasttext, X_use, y_reduced


# Fonction de vectorisation des tweets (BoW, TF-IDF, FastText, USE) + sauvegarde des labels USE
@mlflow_run_safety(experiment_name="P7_sentiment_analysis")
def vectorize_and_save(X_text_full, X_text_reduced, y_full, y_reduced,
                       bow_file="../models_saved/X_bow.pkl", 
                       tfidf_file="../models_saved/X_tfidf.pkl", 
                       fasttext_file="../models_saved/X_fasttext.pkl", 
                       use_file="../models_saved/X_use.pkl", 
                       labels_file="../models_saved/y_use.pkl"):
    """
    Fonction pour vectoriser les tweets (BoW, TF-IDF, FastText, USE) + sauvegarder les labels USE
    """
    if all(os.path.exists(f) for f in [bow_file, tfidf_file, fasttext_file, use_file, labels_file]):
        print("📂 Chargement des matrices vectorisées existantes...")
        X_bow = joblib.load(bow_file)
        X_tfidf = joblib.load(tfidf_file)
        X_fasttext = joblib.load(fasttext_file)
        X_use = joblib.load(use_file)
        y_use = joblib.load(labels_file)
    else:
        print("🚀 Vectorisation en cours...")
        X_bow, X_tfidf, X_fasttext, X_use, y_use = vectorize_tweets(X_text_full, X_text_reduced, y_full, y_reduced)

        joblib.dump(X_bow, bow_file)
        joblib.dump(X_tfidf, tfidf_file)
        joblib.dump(X_fasttext, fasttext_file)
        joblib.dump(X_use, use_file)
        joblib.dump(y_use, labels_file)

    return X_bow, X_tfidf, X_fasttext, X_use, y_use


@mlflow_run_safety(experiment_name="P7_sentiment_analysis")
def balance_dataset(df):
    negatives = df[df['label'] == 0]
    positives = df[df['label'] == 1]
    positives_resampled = resample(positives, replace = True, n_samples = len(negatives), random_state = 70)
    balanced_df = pd.concat([negatives, positives_resampled]).sample(frac = 1, random_state = 70).reset_index(drop = True)
    return balanced_df


# Tokenization DistilBERT
@mlflow_run_safety(experiment_name="P7_sentiment_analysis")
def tokenize_distilbert_dataset(df, tokenizer_path = 'distilbert-base-uncased', save_path = "../models_saved/tokenized_distilbert_dataset"):
    tokenizer = DistilBertTokenizerFast.from_pretrained(tokenizer_path)
    hf_dataset = Dataset.from_pandas(df[['text', 'label']])

    def tokenize_function(examples):
        return tokenizer(examples['text'], truncation = True, padding = 'max_length', max_length = 128)

    if os.path.exists(save_path):
        print(f"✅ Tokenized dataset déjà existant. Chargement depuis {save_path}...")
        tokenized_dataset = load_from_disk(save_path)
    else:
        print("🔄 Tokenisation des tweets...")
        tokenized_dataset = hf_dataset.map(tokenize_function, batched=True)
        tokenized_dataset.save_to_disk(save_path)
        print(f"✅ Dataset tokenizé sauvegardé dans {save_path}")

    return tokenized_dataset


# Préparation du dataset DistilBERT (échantillon)
@mlflow_run_safety(experiment_name="P7_sentiment_analysis")
def prepare_distilbert_dataset(df, sample_size = 100000, dataset_path="../models_saved/distilbert_dataset.pkl"):
    if os.path.exists(dataset_path):
        print("✅ Dataset DistilBERT existant. Chargement...")
        dataset = pd.read_pickle(dataset_path)
    else:
        print("🔄 Préparation du dataset DistilBERT...")
        dataset = df.sample(n = sample_size, random_state = 70).reset_index(drop = True)
        pd.to_pickle(dataset, dataset_path)
        print(f"✅ Dataset sauvegardé sous {dataset_path}")
    return dataset