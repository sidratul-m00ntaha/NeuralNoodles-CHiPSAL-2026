"""
text_sentence_transformer.py
-----------------------------
Semantic text branch: multilingual SentenceTransformer embeddings
+ Logistic Regression, 5-fold stratified OOF probability generation.

Model used: paraphrase-multilingual-MiniLM-L12-v2
  - Supports 50+ languages including Nepali
  - 384-dimensional dense embeddings
  - ~117M parameters (lightweight, runs on CPU if needed)
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sentence_transformers import SentenceTransformer


MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def encode_texts(texts: list, model_name: str = MODEL_NAME,
                 batch_size: int = 32) -> np.ndarray:
    """
    Encode a list of strings to dense sentence embeddings.

    Parameters
    ----------
    texts      : list of str
    model_name : HuggingFace model identifier
    batch_size : encoding batch size

    Returns
    -------
    embeddings : np.ndarray of shape (n_samples, 384)
    """
    print(f"  [SentenceTF] Loading model: {model_name}")
    model = SentenceTransformer(model_name)
    embeddings = model.encode(
        texts, batch_size=batch_size, show_progress_bar=True
    )
    print(f"  [SentenceTF] Encoded {len(texts)} texts → {embeddings.shape}")
    return embeddings


def run_st_oof(
    train_embeddings: np.ndarray,
    test_embeddings: np.ndarray,
    train_labels,
    n_splits: int = 5,
    random_state: int = 42,
):
    """
    Logistic Regression with 5-fold stratified CV over SentenceTransformer
    embeddings. Returns OOF and averaged test probability matrices.

    Parameters
    ----------
    train_embeddings : np.ndarray (n_train, 384)
    test_embeddings  : np.ndarray (n_test, 384)
    train_labels     : array-like of integer labels
    n_splits         : number of CV folds
    random_state     : reproducibility seed

    Returns
    -------
    oof_probs  : np.ndarray (n_train, 3)
    test_probs : np.ndarray (n_test, 3)
    """
    labels    = np.array(train_labels)
    n_train   = len(train_embeddings)
    n_test    = len(test_embeddings)
    n_classes = 3

    oof_probs  = np.zeros((n_train, n_classes))
    test_probs = np.zeros((n_test, n_classes))

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True,
                          random_state=random_state)

    for fold, (train_idx, val_idx) in enumerate(
        skf.split(train_embeddings, labels)
    ):
        print(f"  [SentenceTF] Fold {fold + 1}/{n_splits}")

        X_tr, y_tr = train_embeddings[train_idx], labels[train_idx]
        X_val      = train_embeddings[val_idx]

        clf = LogisticRegression(max_iter=1000, class_weight="balanced")
        clf.fit(X_tr, y_tr)

        oof_probs[val_idx] += clf.predict_proba(X_val)
        test_probs          += clf.predict_proba(test_embeddings) / n_splits

    print("  [SentenceTF] OOF generation complete.")
    return oof_probs, test_probs