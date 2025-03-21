# import re
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# import os
# import gc
# import psutil
# import time
# import pickle

# import spacy
# import nltk
# from nltk import pos_tag
# from nltk.corpus import stopwords, wordnet
# from nltk.stem import WordNetLemmatizer
# import importlib
# import emoji
# from requirements import *

# import fasttext

# import tensorflow_hub as hub
# import tensorflow as tf
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
# from tensorflow.keras.callbacks import EarlyStopping
# from tensorflow.keras.preprocessing.sequence import pad_sequences

# from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
# from sklearn.linear_model import LogisticRegression
# from sklearn.neural_network import MLPClassifier
# from sklearn.preprocessing import LabelEncoder
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.utils import resample
# from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
# from sklearn.decomposition import TruncatedSVD
# from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
# from sklearn.metrics.pairwise import cosine_similarity

# from transformers import BertTokenizer, BertForSequenceClassification, AdamW, Trainer, TrainingArguments, EarlyStoppingCallback, DistilBertTokenizer, DistilBertForSequenceClassification
# import torch
# from torch.utils.data import DataLoader, TensorDataset

# import joblib
# from joblib import Parallel, delayed
# from datasets import Dataset

# import lightgbm as lgb
# from lightgbm import LGBMClassifier, early_stopping, log_evaluation

# from scipy.sparse import vstack, csr_matrix

from requirements import *


nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])  # Désactiver les fonctionnalités inutiles

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('omw-1.4')

# Définition après téléchargement
stop_words = set(stopwords.words('english'))  
lemmatizer = WordNetLemmatizer()


# Affiche le temps écoulé, l'utilisation CPU et RAM en temps réel ou fin d'exécution.
def suivi_temps_ressources(start_time, model_name, phase = "Fin"):
    end_time = time.time()
    elapsed = end_time - start_time
    cpu_usage = psutil.cpu_percent(interval=1)
    ram_usage = psutil.virtual_memory().percent
    print(f"\n⏱️ [{model_name}] - {phase} : {elapsed:.2f} sec | CPU: {cpu_usage}% | RAM: {ram_usage}%")


# Sauvegarde le modèle si le fichier n'existe pas déjà.
def save_model_if_not_exists(model, filename):
    if os.path.exists(filename):
        print(f"✅ Modèle déjà sauvegardé : {filename}. Chargement du modèle existant...")
        model = joblib.load(filename)
        return model
    else:
        joblib.dump(model, filename)
        print(f"✅ Modèle sauvegardé : {filename}")
        return model


# Analyse du dataset
def exploration_data(df):
    print("\nRésumé du dataset :")
    print(df.info())
    print("\nValeurs manquantes :")
    print(df.isnull().sum())
    print("\nDistribution des classes :")
    print(df['label'].value_counts(normalize=True))



# Affiche la structure d'un dossier et de ses sous-dossiers jusqu'à un certain niveau.
def afficher_structure_dossier(chemin_base, niveau=0, max_niveaux=3):
    """
    Paramètres :
    - chemin_base (str) : Chemin du dossier de base.
    - niveau (int) : Niveau actuel (pour la récursion).
    - max_niveaux (int) : Profondeur maximale d'affichage.

    Impact : Utile pour montrer l'organisation du projet.
    """
    if niveau > max_niveaux:
        return  # Stoppe si le niveau maximal est atteint

    indent = "│   " * (niveau - 1) + ("├── " if niveau > 0 else "")
    
    try:
        elements = sorted(os.listdir(chemin_base))
    except PermissionError:
        print(indent + "🔒 [Accès refusé]")
        return
    
    for i, element in enumerate(elements):
        chemin_complet = os.path.join(chemin_base, element)
        is_last = (i == len(elements) - 1)
        prefix = "└── " if is_last else "├── "
        
        print(indent + prefix + element)
        
        # Si c'est un dossier, appel récursif
        if os.path.isdir(chemin_complet):
            afficher_structure_dossier(chemin_complet, niveau + 1, max_niveaux)


# Prend en entrée un texte et retourne le score "compound" de VADER
analyzer = SentimentIntensityAnalyzer()
def compute_vader_scores(df, text_column='text', save_path='vader_scores.pkl'):
    """
    Calcule les scores VADER pour une colonne texte et sauvegarde les résultats pour éviter les recalculs.

    Args:
        df (pd.DataFrame): Le DataFrame contenant les tweets.
        text_column (str): Nom de la colonne contenant les textes.
        save_path (str): Chemin du fichier de sauvegarde des scores.

    Returns:
        pd.Series: Les scores compound VADER.
    """
    if os.path.exists(save_path):
        print(f"✅ Scores VADER déjà calculés. Chargement depuis {save_path}...")
        vader_scores = joblib.load(save_path)
    else:
        print("🔄 Calcul des scores VADER en cours...")
        vader_scores = df[text_column].astype(str).apply(lambda x: analyzer.polarity_scores(x)['compound'])
        joblib.dump(vader_scores, save_path)
        print(f"✅ Scores VADER sauvegardés dans {save_path}.")
    
    return vader_scores



# Association de la bonne étiquette grammaticale à un mot pour une meilleure lemmatisation.
def get_wordnet_pos(word):
    tag = pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ, "V": wordnet.VERB, "N": wordnet.NOUN, "R": wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)              # Par défaut, on considère un nom si non trouvé



# Nettoyage de base du texte.
def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)  # Supprimer les URLs
    text = re.sub(r'@[A-Za-z0-9_]+', '', text)  # Supprimer les mentions
    text = emoji.replace_emoji(text, replace='')  # Supprimer les emojis
    text = re.sub(r'[^a-zA-Z ]', '', text)  # Supprimer les caractères non alphabétiques
    text = ' '.join([word for word in text.split() if len(word) > 1])  # Suppression des lettres isolées
    return text



# Lemmatisation rapide avec spaCy
def lemmatize_text(text):
    doc = nlp(text)
    return ' '.join([token.lemma_ for token in doc if not token.is_stop])


# Appliquer le nettoyage et la lemmatisation en batch
def preprocess_batch(batch_df):
    batch_df['text'] = batch_df['text'].astype(str).apply(clean_text)  # Nettoyage rapide
    batch_df['text'] = batch_df['text'].apply(lemmatize_text)  # Lemmatisation rapide
    return batch_df



# Nettoie les tweets en parallèle avec batch processing et sauvegarde les résultats.
def preprocess_tweets_parallel(df, filename="cleaned_tweets.pkl", n_jobs=-1, batch_size=50000):
    
    if os.path.exists(filename):
        print(f"✅ Chargement des tweets nettoyés depuis {filename}")
        return pd.read_pickle(filename)

    print("🚀 Nettoyage des tweets en cours...")

    # Découper en batches pour optimiser la mémoire
    batches = [df.iloc[i:i + batch_size] for i in range(0, len(df), batch_size)]

    # Nettoyage en parallèle
    cleaned_batches = Parallel(n_jobs=n_jobs, backend="multiprocessing")(
        delayed(preprocess_batch)(batch) for batch in batches
    )

    # Reconstruction du DataFrame final
    df_cleaned = pd.concat(cleaned_batches, ignore_index=True)

    # Sauvegarde du DataFrame nettoyé
    df_cleaned.to_pickle(filename)
    print(f"✅ Tweets nettoyés sauvegardés dans {filename}")

    return df_cleaned


# Sauvegarde les tweets dans un fichier texte pour FastText
def save_tweets_for_fasttext(X_text_full, filename="tweets_fasttext.txt"):
    with open(filename, 'w', encoding='utf-8') as f:
        for tweet in X_text_full:
            f.write(tweet + '\n')
    print(f"✅ Fichier {filename} créé avec succès.")




# Vérifie si les matrices vectorisées existent déjà. Si oui, les charge, sinon les génère et les sauvegarde
def vectorize_and_save(X_text_full, X_text_reduced, bow_file="X_bow.pkl", tfidf_file="X_tfidf.pkl", fasttext_file="X_fasttext.pkl", use_file="X_use.pkl"):
    # Vérifier si les fichiers existent déjà
    if all(os.path.exists(f) for f in [bow_file, tfidf_file, fasttext_file, use_file]):
        print("📂 Chargement des matrices vectorisées existantes...")
        X_bow = load_matrix(bow_file)
        X_tfidf = load_matrix(tfidf_file)
        X_fasttext = load_matrix(fasttext_file)
        X_use = load_matrix(use_file)
    else:
        print("🚀 Vectorisation en cours...")
        X_bow, X_tfidf, X_fasttext, X_use = vectorize_tweets(X_text_full, X_text_reduced)

        # Sauvegarder les matrices
        save_matrix(X_bow, bow_file)
        save_matrix(X_tfidf, tfidf_file)
        save_matrix(X_fasttext, fasttext_file)
        save_matrix(X_use, use_file)

    return X_bow, X_tfidf, X_fasttext, X_use


# Vectorise les tweets en BoW, TF-IDF, FastText et Universal Sentence Encoder
def vectorize_tweets(X_text_full, X_text_reduced):
    print("🚀 Vectorisation BoW et TF-IDF en cours...")
    count_vectorizer = CountVectorizer()
    X_bow = count_vectorizer.fit_transform(X_text_full)

    tfidf_vectorizer = TfidfVectorizer(max_features=2000)
    X_tfidf = tfidf_vectorizer.fit_transform(X_text_full)

    # 🔍 Vérifier et créer le fichier pour FastText si nécessaire
    fasttext_file = "tweets_fasttext.txt"
    if not os.path.exists(fasttext_file):
        print("📂 Création du fichier tweets_fasttext.txt pour FastText...")
        save_tweets_for_fasttext(X_text_full)

    # 🚀 Entraînement FastText
    try:
        print("🚀 Entraînement du modèle FastText en cours...")
        fasttext_model = fasttext.train_unsupervised(fasttext_file, model='skipgram', dim=300)
        X_fasttext = np.array([fasttext_model.get_sentence_vector(text) for text in X_text_full])
    except Exception as e:
        print(f"❌ Erreur lors de l'entraînement FastText : {e}")
        X_fasttext = None  # Permet de ne pas interrompre la pipeline

    # 🚀 Chargement de Universal Sentence Encoder (USE)
    try:
        print("🚀 Chargement du modèle Universal Sentence Encoder...")
        use_model = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
        X_use = np.array([use_model([text]).numpy().flatten() for text in X_text_reduced])
    except Exception as e:
        print(f"❌ Erreur lors du chargement Universal Sentence Encoder : {e}")
        X_use = None  # Évite une interruption du script

    print("✅ Vectorisation terminée.")
    return X_bow, X_tfidf, X_fasttext, X_use

# ✅ S'assurer que les labels sont bien formatés pour FastText
def format_label(label):
    return f"__label__{label}"


# Gestion du déséquilibre des labels
def balance_dataset(df):
    negatives = df[df['label'] == 0]
    positives = df[df['label'] == 1]
    positives_resampled = resample(positives, replace=True, n_samples=len(negatives), random_state=42)
    balanced_df = pd.concat([negatives, positives_resampled]).sample(frac=1, random_state=42).reset_index(drop=True)
    return balanced_df


# Analyse des longueurs des tweets
def compute_tweet_lengths(df):
    df['length'] = df['text'].apply(len)
    return df


# Visualisation des longueurs des tweets
def plot_tweet_length_distribution(df):
    import matplotlib.pyplot as plt
    import seaborn as sns
    plt.figure(figsize=(8, 4))
    sns.histplot(df['length'], bins=50, kde=True)
    plt.title("Distribution des longueurs des tweets")
    plt.xlabel("Nombre de caractères")
    plt.ylabel("Fréquence")
    plt.show()


# Sauvegarde une matrice avec pickle
def save_matrix(matrix, filename):
    with open(filename, 'wb') as f:
        pickle.dump(matrix, f)
    print(f"✅ {filename} sauvegardé avec succès.")


# Charge une matrice sauvegardée avec pickle
def load_matrix(filename):
    with open(filename, 'rb') as f:
        print(f"✅ Chargement de {filename}...")
        return pickle.load(f)


# train_logistic_regression_with_cv (Optimisée CPU + Sauvegarde)
def train_logistic_regression_with_cv(X, y, model_path = "log_reg_model.pkl"):
    if os.path.exists(model_path):
        print("✅ Modèle Régression Logistique déjà existant. Chargement...")
        return joblib.load(model_path)

    start = time.time()
    param_grid = {
        "C": [0.01, 0.1, 1, 10, 100],
        "penalty": ["l1", "l2"],
        "solver": ["liblinear"]
    }
    log_reg = LogisticRegression(max_iter = 200)
    grid_search = GridSearchCV(log_reg, param_grid, cv = 5, scoring = "accuracy", verbose = 1, n_jobs = -1)
    print("🔄 Entraînement Régression Logistique avec GridSearchCV (optimisé multicœur)...")
    grid_search.fit(X, y)
    best_log_reg = grid_search.best_estimator_
    print(f"✅ Meilleur modèle : {best_log_reg}")
    joblib.dump(best_log_reg, model_path)
    print(f"✅ Modèle sauvegardé sous {model_path}")
    suivi_temps_ressources(start, "Régression Logistique")
    return best_log_reg


# Affiche une matrice de confusion pour un modèle donné.
def plot_confusion_matrix(y_true, y_pred, model_name):
    plt.figure(figsize = (6,6))
    sns.heatmap(confusion_matrix(y_true, y_pred), annot = True, fmt = "d", cmap = "Blues")
    plt.xlabel("Prédictions")
    plt.ylabel("Réel")
    plt.title(f"Matrice de confusion - {model_name}")
    plt.show()


# Affiche les tweets mal classifiés.
def display_misclassified_tweets(X_test_text, y_true, y_pred, model_name, max_display = 10):
    misclassified_idx = np.where(y_pred != y_true)[0]
    print(f"\n🔍 Tweets mal classifiés par {model_name} :")
    count = 0
    for idx in misclassified_idx[:max_display]:
        print(f"❌ {X_test_text.iloc[idx]} \n   Réel: {y_true[idx]} - Prédit: {y_pred[idx]}\n")
        count += 1
    if count == 0:
        print("✅ Aucune erreur détectée !")



# train_random_forest (Optimisée CPU + Sauvegarde)
def train_random_forest(X_train, y_train, model_path = "rf_model.pkl"):
    if os.path.exists(model_path):
        print("✅ Modèle RandomForest déjà existant. Chargement...")
        return joblib.load(model_path)

    start = time.time()
    rf = RandomForestClassifier(n_estimators = 100, max_depth = 10, n_jobs = -1)
    rf.fit(X_train, y_train)
    joblib.dump(rf, model_path)
    print(f"✅ Modèle RandomForest sauvegardé sous {model_path}")
    suivi_temps_ressources(start, "RandomForest")
    return rf


# train_lightgbm (Optimisée CPU + Sauvegarde)
def train_lightgbm(X_train, y_train, X_val, y_val, model_path="lgbm_model.txt"):
    if os.path.exists(model_path):
        print("✅ Modèle LightGBM déjà existant. Chargement...")
        lgbm = lgb.Booster(model_file=model_path)
        return lgbm

    start = time.time()

    # 🚀 Conversion des labels pour éviter le type 'object'
    y_train = pd.Series(y_train).astype(int)
    y_val = pd.Series(y_val).astype(int)

    lgb_train = lgb.Dataset(X_train, label=y_train)
    lgb_eval = lgb.Dataset(X_val, label=y_val, reference=lgb_train)

    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'num_leaves': 31,
        'learning_rate': 0.1,
        'max_depth': 5,
        'n_jobs': -1,
        'verbose': -1
    }

    print("🚀 Entraînement LightGBM en cours...")
    lgbm_model = lgb.train(
        params,
        lgb_train,
        valid_sets=[lgb_eval],
        num_boost_round=100,
        callbacks=[
            early_stopping(stopping_rounds=10),
            log_evaluation(period=10)
        ]
    )

    lgbm_model.save_model(model_path)
    print(f"✅ Modèle LightGBM sauvegardé sous {model_path}")
    suivi_temps_ressources(start, "LightGBM")
    return lgbm_model




def train_neural_network_with_cv(X, y, model_path="mlp_model.pkl"):
    if os.path.exists(model_path):
        print("✅ Modèle MLP déjà existant. Chargement...")
        return joblib.load(model_path)

    start = time.time()

    # SUPPRIMÉ : Conversion en dense (cause la mémoire saturée)
    # X = X.toarray().astype(np.float16)

    y = LabelEncoder().fit_transform(y)

    param_grid = {
        'hidden_layer_sizes': [(10,), (20,), (50,)],
        'activation': ['relu', 'tanh'],
        'solver': ['adam'],
        'alpha': [0.0001, 0.001, 0.01],
        'learning_rate': ['constant', 'adaptive']
    }
    mlp = MLPClassifier(max_iter=300, tol=1e-4, verbose=True)

    random_search = RandomizedSearchCV(
        mlp, param_distributions=param_grid, n_iter=5, cv=3,
        scoring="accuracy", verbose=2, n_jobs=1,  # TEMPORAIREMENT n_jobs=1 (peuvent être remis à -1 après tests)
        random_state=70
    )
    print("🔄 Entraînement MLP avec RandomizedSearchCV... (Format sparse optimisé, RAM allégée)")
    random_search.fit(X, y)

    best_mlp = random_search.best_estimator_
    print(f"✅ Meilleur modèle MLP : {best_mlp}")

    joblib.dump(best_mlp, model_path)
    print(f"✅ Modèle sauvegardé sous {model_path}")

    # Appelle ta fonction de suivi (si elle existe chez toi)
    suivi_temps_ressources(start, "MLP")

    return best_mlp


# cosine_similarity_heatmap - Calcule et affiche la matrice de similarité cosinus pour un échantillon donné d'embeddings
def cosine_similarity_heatmap(embeddings, sample_size = 1000, save_path = "cosine_similarity.npy"):
    if os.path.exists(save_path):
        print("✅ Matrice de similarité déjà calculée. Chargement...")
        similarity_matrix = np.load(save_path)
    else:
        print("🔄 Calcul de la matrice de similarité cosinus...")
        start = time.time()
        similarity_matrix = cosine_similarity(embeddings[:sample_size])
        np.save(save_path, similarity_matrix)
        suivi_temps_ressources(start, f"Similarité Cosinus ({sample_size} tweets)")
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(similarity_matrix, cmap = 'coolwarm')
    plt.title(f"Matrice de Similarité Cosinus sur {sample_size} Embeddings USE")
    plt.xlabel("Tweets")
    plt.ylabel("Tweets")
    plt.show()
    return similarity_matrix


# Entraînement LSTM + sauvegarde
def train_lstm_model(X_embeddings, y_labels, model_path="lstm_model.h5"):
    # Normalisation & conversion
    X = np.array(X_embeddings)
    y = np.array(y_labels).astype(int)  # S’assurer que y est bien en int

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=70)

    # Reshape pour LSTM
    X_train_reshaped = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
    X_test_reshaped = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

    # === Modèle déjà existant ===
    if os.path.exists(model_path):
        print(f"✅ Modèle LSTM déjà existant. Chargement...")
        model = tf.keras.models.load_model(model_path)
        return model, (X_test_reshaped, y_test), None

    # === Nouveau modèle ===
    start = time.time()

    model = Sequential()
    model.add(Bidirectional(LSTM(128), input_shape=(1, X.shape[1])))
    model.add(Dropout(0.3))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

    print("🚀 Entraînement du modèle LSTM en cours...")
    # model.fit(X_train_reshaped, y_train, validation_data=(X_test_reshaped, y_test),
    #           epochs=10, batch_size=256, callbacks=[early_stop], verbose=1)
    history = model.fit(X_train_reshaped, y_train, validation_data=(X_test_reshaped, y_test),
              epochs=10, batch_size=256, callbacks=[early_stop], verbose=1)

    # Sauvegarde
    model.save(model_path)
    print(f"✅ Modèle LSTM sauvegardé sous {model_path}")

    # Évaluation finale
    loss, acc = model.evaluate(X_test_reshaped, y_test)
    print(f"🎯 Performance LSTM - Accuracy: {acc:.4f} | Loss: {loss:.4f}")

    suivi_temps_ressources(start, "LSTM FastText")

    # return model, (X_test_reshaped, y_test)
    return model, (X_test_reshaped, y_test), history


# Prédiction & rapport
def evaluate_lstm_model(model, X_test, y_test):
    y_pred = (model.predict(X_test) > 0.5).astype(int).flatten()
    print("\n📊 Rapport de classification :")
    print(classification_report(y_test, y_pred))

    plt.figure(figsize = (6,6))
    sns.heatmap(confusion_matrix(y_test, y_pred), annot = True, fmt = "d", cmap = "Blues")
    plt.xlabel("Prédictions")
    plt.ylabel("Réel")
    plt.title("Matrice de confusion - LSTM (FastText)")
    plt.show()


def load_finetuned_distilbert(model_save_path="distilbert_model"):
    """
    Charge un modèle DistilBERT fine-tuné depuis le disque.

    Args:
        model_save_path (str): Chemin du dossier contenant le modèle sauvegardé.

    Returns:
        model: Modèle DistilBERT fine-tuné prêt à l'emploi.
    """
    if not os.path.exists(model_save_path):
        print(f"❌ Aucun modèle DistilBERT trouvé à l'emplacement {model_save_path}.")
        return None

    print(f"✅ Chargement du modèle DistilBERT depuis {model_save_path}...")
    model = DistilBertForSequenceClassification.from_pretrained(model_save_path)
    return model

def load_tokenized_dataset(save_path="tokenized_distilbert_dataset"):
    """
    Charge un dataset tokenizé DistilBERT depuis le disque.
    
    Args:
        save_path (str): Chemin du dossier contenant le dataset tokenizé.

    Returns:
        Dataset: Le dataset tokenizé prêt à l'emploi.
    """
    if not os.path.exists(save_path):
        print(f"❌ Aucun dataset tokenizé trouvé à l'emplacement {save_path}.")
        return None

    print(f"✅ Chargement du dataset tokenizé depuis {save_path}...")
    tokenized_dataset = load_from_disk(save_path)
    return tokenized_dataset


# Préparation du dataset pour DistilBERT (100k tweets)
def prepare_distilbert_dataset(tweets, sample_size = 100000, dataset_path = "distilbert_dataset.pkl"):
    if os.path.exists(dataset_path):
        print("✅ Dataset DistilBERT déjà préparé. Chargement...")
        dataset = pd.read_pickle(dataset_path)
    else:
        print("🔄 Préparation du dataset DistilBERT...")
        dataset = tweets.sample(n = sample_size, random_state = 70).reset_index(drop = True)
        pd.to_pickle(dataset, dataset_path)
        print(f"✅ Dataset échantillonné et sauvegardé sous {dataset_path}")
    return dataset


# Tokenisation avec DistilBERT tokenizer
def tokenize_distilbert_dataset(df, tokenizer_path = 'distilbert-base-uncased', save_path = "tokenized_distilbert_dataset"):
    tokenizer = DistilBertTokenizerFast.from_pretrained(tokenizer_path)
    hf_dataset = Dataset.from_pandas(df[['text', 'label']])

    def tokenize_function(examples):
        return tokenizer(examples['text'], truncation = True, padding = 'max_length', max_length = 128)

    if os.path.exists(save_path):
        print(f"✅ Tokenized dataset déjà existant. Chargement depuis {save_path}...")
        tokenized_dataset = load_from_disk(save_path)
    else:
        print("🔄 Tokenisation des tweets...")
        tokenized_dataset = hf_dataset.map(tokenize_function, batched = True)
        tokenized_dataset.save_to_disk(save_path)
        print(f"✅ Dataset tokenizé et sauvegardé dans {save_path}")

    return tokenized_dataset


# Entraînement DistilBERT + Sauvegarde du modèle
def train_distilbert_model(tokenized_dataset, model_save_path="distilbert_model"):
    from datasets import ClassLabel

    if os.path.exists(model_save_path):
        print(f"✅ Modèle DistilBERT déjà fine-tuné. Chargement depuis {model_save_path}...")
        model = DistilBertForSequenceClassification.from_pretrained(model_save_path)
        return model, None, None  # <-- correction ici

    print("🚀 Fine-tuning DistilBERT...")
    model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)

    if tokenized_dataset.features['label'].__class__.__name__ != 'ClassLabel':
        num_classes = len(set(tokenized_dataset['label']))
        print(f"🔄 Conversion de la colonne label en ClassLabel avec {num_classes} classes...")
        tokenized_dataset = tokenized_dataset.cast_column('label', ClassLabel(num_classes=num_classes))

    # Split
    dataset_split = tokenized_dataset.train_test_split(test_size=0.2, stratify_by_column='label')
    train_dataset = dataset_split['train']
    test_dataset = dataset_split['test']

    training_args = TrainingArguments(
        output_dir="distilbert_output",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        logging_dir="logs",
        logging_steps=50
    )

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = logits.argmax(axis=-1)
        acc = accuracy_score(labels, predictions)
        f1 = f1_score(labels, predictions)
        return {"accuracy": acc, "f1": f1}

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics
    )

    trainer.train()
    model.save_pretrained(model_save_path)
    print(f"✅ Modèle DistilBERT sauvegardé sous {model_save_path}")

    return model, trainer, test_dataset


# Evaluation DistilBert
def evaluate_distilbert_model(model, tokenized_dataset, results_path="distilbert_eval_results.pkl"):
    import joblib

    # Vérifier si les résultats existent déjà
    if os.path.exists(results_path):
        print(f"✅ Résultats d'évaluation déjà disponibles. Chargement depuis {results_path}...")
        eval_results = joblib.load(results_path)
        
        # Affichage des résultats sauvegardés
        print("\n📊 Rapport de classification :")
        print(eval_results['classification_report'])
        
        sns.heatmap(eval_results['confusion_matrix'], annot=True, fmt="d", cmap="Blues")
        plt.xlabel("Prédictions")
        plt.ylabel("Réel")
        plt.title("Matrice de confusion - DistilBERT (chargée)")
        plt.show()
        
        return eval_results['accuracy'], eval_results['f1']

    print("📊 Évaluation DistilBERT...")

    # ✅ Vérifier si la colonne label est bien ClassLabel
    if tokenized_dataset.features['label'].__class__.__name__ != 'ClassLabel':
        num_classes = len(set(tokenized_dataset['label']))
        print(f"🔄 Conversion de la colonne label en ClassLabel avec {num_classes} classes pour évaluation...")
        tokenized_dataset = tokenized_dataset.cast_column('label', ClassLabel(num_classes=num_classes))

    # Split du dataset (stratifié)
    dataset_split = tokenized_dataset.train_test_split(test_size=0.2, stratify_by_column='label')
    test_dataset = dataset_split['test']

    # Configuration Trainer pour prédiction
    trainer = Trainer(model=model)
    preds = trainer.predict(test_dataset)
    y_pred = preds.predictions.argmax(axis=-1)
    y_true = test_dataset['label']

    # Rapport & Matrice
    report = classification_report(y_true, y_pred)
    matrix = confusion_matrix(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    print("\n📊 Rapport de classification :")
    print(report)

    plt.figure(figsize=(6, 6))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Prédictions")
    plt.ylabel("Réel")
    plt.title("Matrice de confusion - DistilBERT")
    plt.show()

    # Sauvegarde des résultats
    eval_results = {
        'accuracy': acc,
        'f1': f1,
        'classification_report': report,
        'confusion_matrix': matrix
    }
    joblib.dump(eval_results, results_path)
    print(f"✅ Résultats sauvegardés sous {results_path}")

    return acc, f1



from sklearn.metrics import accuracy_score, f1_score, classification_report

def get_all_model_scores(tweets, 
                         X_tfidf_train, X_tfidf_test, 
                         X_ft_train, X_ft_test, 
                         X_use_train, X_use_test, 
                         y_train, y_test, 
                         lstm_model, X_ft_test_reshaped, y_ft_test, 
                         distilbert_model, tokenized_dataset):

    all_scores = {}

    # 🔥 Forcer conversion des labels
    y_train_int = y_train.astype(int)
    y_test_int = y_test.astype(int)
    y_ft_test_int = y_ft_test.astype(int)

    # 1️⃣ TF-IDF + Logistic Regression
    log_reg_model = joblib.load("log_reg_model.pkl")
    y_pred_tfidf = log_reg_model.predict(X_tfidf_test)
    all_scores['tfidf_acc'] = round(accuracy_score(y_test_int, y_pred_tfidf), 4)
    all_scores['tfidf_f1'] = round(f1_score(y_test_int, y_pred_tfidf), 4)

    # 2️⃣ FastText + Logistic Regression
    log_reg_ft = LogisticRegression(max_iter=200)
    log_reg_ft.fit(X_ft_train, y_train_int)
    y_pred_ft_logreg = log_reg_ft.predict(X_ft_test)
    all_scores['fasttext_logreg_acc'] = round(accuracy_score(y_test_int, y_pred_ft_logreg), 4)
    all_scores['fasttext_logreg_f1'] = round(f1_score(y_test_int, y_pred_ft_logreg), 4)

    # 3️⃣ FastText + RandomForest
    rf_model = joblib.load("rf_model.pkl")
    # Vérifier si RandomForest a été entraîné sur str → convertissons les prédictions
    y_pred_rf = rf_model.predict(X_ft_test)
    if isinstance(y_pred_rf[0], str):
        y_pred_rf = y_pred_rf.astype(int)
    all_scores['fasttext_rf_acc'] = round(accuracy_score(y_test_int, y_pred_rf), 4)
    all_scores['fasttext_rf_f1'] = round(f1_score(y_test_int, y_pred_rf), 4)

    # 4️⃣ FastText + LSTM
    y_pred_lstm = (lstm_model.predict(X_ft_test_reshaped) > 0.5).astype(int).flatten()
    all_scores['fasttext_lstm_acc'] = round(accuracy_score(y_ft_test_int, y_pred_lstm), 4)
    all_scores['fasttext_lstm_f1'] = round(f1_score(y_ft_test_int, y_pred_lstm), 4)

    # 5️⃣ USE + LightGBM
    lgbm_model = lgb.Booster(model_file="lgbm_model.txt")
    y_pred_lgb = (lgbm_model.predict(X_use_test) > 0.5).astype(int)
    all_scores['use_lgbm_acc'] = round(accuracy_score(y_test_int, y_pred_lgb), 4)
    all_scores['use_lgbm_f1'] = round(f1_score(y_test_int, y_pred_lgb), 4)

    # 6️⃣ DistilBERT
    acc_distilbert, f1_distilbert = evaluate_distilbert_model(distilbert_model, tokenized_dataset)
    all_scores['distilbert_acc'] = round(acc_distilbert, 4)
    all_scores['distilbert_f1'] = round(f1_distilbert, 4)

    return all_scores




# Comparaison finale des modèles
# def update_comparison_table(distilbert_acc, distilbert_f1):
#     comparison_results = {
#         'Modèle': ['TF-IDF + LogReg', 'FastText + LogReg', 'FastText + RandomForest', 'FastText + LSTM', 'USE + LightGBM', 'DistilBERT fine-tuné'],
#         'Accuracy': ["à remplir", "à remplir", "à remplir", "à remplir", "à remplir", distilbert_acc],
#         'F1-score': ["à remplir", "à remplir", "à remplir", "à remplir", "à remplir", distilbert_f1],
#         'Temps entraînement': ["", "", "", "", "", "à mesurer"],
#         'Ressources': ["", "", "", "", "", "CPU"]
#     }

#     df_results = pd.DataFrame(comparison_results)
#     print("\n📊 Comparaison des modèles :")
#     print(df_results)
#     return df_results


def update_comparison_table(all_scores):
    comparison_results = {
        'Modèle': ['TF-IDF + LogReg', 'FastText + LogReg', 'FastText + RandomForest', 
                   'FastText + LSTM', 'USE + LightGBM', 'DistilBERT fine-tuné'],
        'Accuracy': [all_scores.get('tfidf_acc', "à remplir"), 
                     all_scores.get('fasttext_logreg_acc', "à remplir"), 
                     all_scores.get('fasttext_rf_acc', "à remplir"), 
                     all_scores.get('fasttext_lstm_acc', "à remplir"), 
                     all_scores.get('use_lgbm_acc', "à remplir"), 
                     all_scores.get('distilbert_acc', "à remplir")],
        'F1-score': [all_scores.get('tfidf_f1', "à remplir"), 
                     all_scores.get('fasttext_logreg_f1', "à remplir"), 
                     all_scores.get('fasttext_rf_f1', "à remplir"), 
                     all_scores.get('fasttext_lstm_f1', "à remplir"), 
                     all_scores.get('use_lgbm_f1', "à remplir"), 
                     all_scores.get('distilbert_f1', "à remplir")],
        'Temps entraînement': ["à remplir"]*6,
        'Ressources': ["CPU"]*6
    }

    df_results = pd.DataFrame(comparison_results)
    print("\n📊 Comparaison des modèles :")
    display(df_results)
    return df_results