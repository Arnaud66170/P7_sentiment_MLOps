from requirements import *
from src.utils import suivi_temps_ressources

# Logistic Regression

def train_logistic_regression_with_cv(X, y, model_path="models_saved/log_reg_model.pkl"):
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
    print("🔄 Entraînement Régression Logistique...")
    grid_search.fit(X, y)
    best_log_reg = grid_search.best_estimator_
    joblib.dump(best_log_reg, model_path)
    print(f"✅ Modèle sauvegardé sous {model_path}")
    suivi_temps_ressources(start, "Régression Logistique")
    return best_log_reg


# Random Forest
def train_random_forest(X_train, y_train, model_path = "models_saved/rf_model.pkl"):
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


# LightGBM
def train_lightgbm(X_train, y_train, X_val, y_val, model_path = "models_saved/lgbm_model.txt"):
    if os.path.exists(model_path):
        print("✅ Modèle LightGBM existant. Chargement...")
        return lgb.Booster(model_file = model_path)
    start = time.time()
    lgb_train = lgb.Dataset(X_train, label = y_train)
    lgb_eval = lgb.Dataset(X_val, label = y_val, reference = lgb_train)
    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'num_leaves': 31,
        'learning_rate': 0.1,
        'max_depth': 5,
        'n_jobs': -1,
        'verbose': -1
    }
    print("🚀 Entraînement LightGBM...")
    lgbm_model = lgb.train(
        params, lgb_train, valid_sets = [lgb_eval], num_boost_round = 100,
        callbacks=[early_stopping(stopping_rounds = 10), log_evaluation(period = 10)]
    )
    lgbm_model.save_model(model_path)
    suivi_temps_ressources(start, "LightGBM")
    return lgbm_model



# FastText Supervised Training
def train_fasttext_supervised(file_path = "models_saved/tweets_fasttext.txt", model_path = "models_saved/fasttext_model.ftz"):
    if os.path.exists(model_path):
        print("✅ Modèle FastText supervisé existant. Chargement...")
        return fasttext.load_model(model_path)

    print("🚀 Entraînement FastText supervisé...")
    model = fasttext.train_supervised(file_path, epoch = 10, wordNgrams = 2, dim = 300)
    model.save_model(model_path)
    print(f"✅ Modèle FastText sauvegardé sous {model_path}")
    return model


# LSTM Training sur FastText
def train_lstm_model(X_embeddings, y_labels, model_path = "models_saved/lstm_model.h5"):
    X = np.array(X_embeddings)
    y = np.array(y_labels).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, stratify = y, random_state = 70)

    X_train_reshaped = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
    X_test_reshaped = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

    if os.path.exists(model_path):
        print(f"✅ Modèle LSTM déjà existant. Chargement...")
        model = tf.keras.models.load_model(model_path)
        return model, (X_test_reshaped, y_test), None

    start = time.time()
    model = Sequential()
    model.add(Bidirectional(LSTM(128), input_shape=(1, X.shape[1])))
    model.add(Dropout(0.3))
    model.add(Dense(64, activation = 'relu'))
    model.add(Dense(1, activation = 'sigmoid'))

    model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    early_stop = EarlyStopping(monitor = 'val_loss', patience = 3, restore_best_weights = True)

    print("🚀 Entraînement LSTM...")
    history = model.fit(X_train_reshaped, y_train, validation_data = (X_test_reshaped, y_test),
                        epochs = 10, batch_size = 256, callbacks = [early_stop], verbose = 1)

    model.save(model_path)
    print(f"✅ Modèle LSTM sauvegardé sous {model_path}")
    suivi_temps_ressources(start, "LSTM")

    return model, (X_test_reshaped, y_test), history


# DistilBERT Fine-tuning
def train_distilbert_model(tokenized_dataset, model_save_path = "models_saved/distilbert_model"):
    from datasets import ClassLabel

    if os.path.exists(model_save_path):
        print(f"✅ Modèle DistilBERT déjà fine-tuné. Chargement depuis {model_save_path}...")
        model = DistilBertForSequenceClassification.from_pretrained(model_save_path)
        return model, None, None

    print("🚀 Fine-tuning DistilBERT...")
    model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels = 2)

    if tokenized_dataset.features['label'].__class__.__name__ != 'ClassLabel':
        num_classes = len(set(tokenized_dataset['label']))
        tokenized_dataset = tokenized_dataset.cast_column('label', ClassLabel(num_classes = num_classes))

    dataset_split = tokenized_dataset.train_test_split(test_size = 0.2, stratify_by_column = 'label')
    train_dataset = dataset_split['train']
    test_dataset = dataset_split['test']

    training_args = TrainingArguments(
        output_dir="models_saved/distilbert_output",
        evaluation_strategy = "epoch",
        save_strategy = "epoch",
        learning_rate = 2e-5,
        per_device_train_batch_size = 16,
        per_device_eval_batch_size = 16,
        num_train_epochs = 3,
        weight_decay = 0.01,
        save_total_limit = 1,
        load_best_model_at_end = True,
        metric_for_best_model = "accuracy",
        logging_dir = "models_saved/logs",
        logging_steps = 50
    )

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = logits.argmax(axis = -1)
        acc = accuracy_score(labels, predictions)
        f1 = f1_score(labels, predictions)
        return {"accuracy": acc, "f1": f1}

    trainer = Trainer(
        model = model,
        args = training_args,
        train_dataset = train_dataset,
        eval_dataset = test_dataset,
        compute_metrics = compute_metrics
    )

    trainer.train()
    model.save_pretrained(model_save_path)
    print(f"✅ Modèle DistilBERT sauvegardé sous {model_save_path}")
    return model, trainer, test_dataset