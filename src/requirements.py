import sys

# ====================================
# INSTALLATION AUTOMATIQUE (au besoin)
# ====================================

def check_and_install(package, import_name=None):

    try:
        __import__(import_name or package)
    except ImportError:
        import subprocess
        print(f"⚠️ Le package '{package}' n'est pas installé. Installation en cours...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# Liste des packages à vérifier/installer
packages = [
    "pandas",
    "numpy",
    "openpyxl",
    "matplotlib",
    "seaborn",
    "plotly",
    "nltk",
    "vaderSentiment",
    "spacy",
    "transformers",
    "sentencepiece",
    "wordcloud",
    "gensim",
    "tensorflow",
    "torch",
    # "scikit-learn",
    "mlflow",
    "optuna",
    "flask",
    "fastapi",
    "uvicorn",
    "streamlit",
    # "azure-monitor-query",
    # "azure-mgmt-monitor",
    "requests",
    "tqdm",
    "pytest"
]

# Certains packages ont un nom d'import différent
import_names = {
    "vaderSentiment": "vaderSentiment.vaderSentiment"
}

# Vérification et installation
for package in packages:
    check_and_install(package, import_names.get(package, package))

# ====================================
# IMPORTATION DES LIBRAIRIES
# ====================================

# Standard Libraries
import os
import gc
import re
import time
import json
import glob
import pickle
import multiprocessing
from collections import Counter

# Data Manipulation
import pandas as pd
import numpy as np
import openpyxl

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from wordcloud import WordCloud

# System & Performance
import psutil
from tqdm import tqdm

# Text Processing & NLP
import nltk
from nltk import pos_tag
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import spacy
try:
    spacy.load("en_core_web_sm")
except OSError:
    print("🔄 Téléchargement du modèle spaCy en_core_web_sm...")
    import subprocess
    subprocess.check_call(["python", "-m", "spacy", "download", "en_core_web_sm"])
import emoji
import sentencepiece

# Machine Learning & Preprocessing
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import TruncatedSVD
from sklearn.utils import resample
from sklearn.metrics import (accuracy_score, confusion_matrix, classification_report,
                             log_loss, f1_score)
from sklearn.metrics.pairwise import cosine_similarity

# Sampling
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

# Deep Learning & Embeddings
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow import keras
import tensorflow_hub as hub

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import gensim
import fasttext
import h5py

# Transformers & NLP Models
from transformers import (pipeline, BertTokenizer, BertForSequenceClassification,
                          DistilBertTokenizer, DistilBertForSequenceClassification,
                          DistilBertTokenizerFast, Trainer, TrainingArguments,
                          EarlyStoppingCallback, AdamW, TFBertForSequenceClassification)

# Parallel Processing & Utilities
import joblib
from joblib import Parallel, delayed
import importlib

# Dataset Management
from datasets import Dataset, load_from_disk, ClassLabel

# Gradient Boosting
import lightgbm as lgb
from lightgbm import LGBMClassifier, early_stopping, log_evaluation

# Experiment Tracking & Optimization
import mlflow
import mlflow.sklearn
import optuna

# API & Web
from flask import Flask, request, jsonify
from fastapi import FastAPI
import uvicorn
import streamlit as st

# Testing & Requests
import pytest
import requests

# Optional Cloud Monitoring (commented out)
# from azure.monitor.query import LogsQueryClient
# from azure.mgmt.monitor import MonitorManagementClient
# from azure.identity import DefaultAzureCredential



# ====================================
# MESSAGES DE CONFIRMATION
# ====================================
print("\n✅ Toutes les librairies sont présentes et prêtes à être utilisées !\n")
