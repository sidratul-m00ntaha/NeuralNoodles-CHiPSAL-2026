"""
text_tfidf.py
-------------
TF-IDF lexical text branch: word n-grams + character n-grams,
Logistic Regression classifier, 5-fold stratified OOF probability generation.
"""

import re
import unicodedata
import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold


# ── Text cleaning ─────────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """
    Light normalisation suitable for Nepali/multilingual meme text.
    - Unicode NFKC normalisation
    - Remove URLs
    - Collapse whitespace
    """
    text = unicodedata.normalize("NFKC", str(text))
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ── OOF generation ────────────────────────────────────────────────────────────

def run_tfidf_oof(
    train_df,
    test_df,
    text_col: str = "clean_text",
    label_col: str = "label",
    n_splits: int = 5,
    random_state: int = 42,
    word_max_features: int = 3000,
    char_max_features: int = 2000,
):
    """
    Fit TF-IDF (word + char) + Logistic Regression using 5-fold stratified CV.
    Returns OOF probability matrix for train and averaged test probabilities.

    Parameters
    ----------
    train_df          : pd.DataFrame — must have text_col and label_col
    test_df           : pd.DataFrame — must have text_col
    text_col          : column name for cleaned text
    label_col         : column name for integer labels
    n_splits          : number of CV folds
    random_state      : reproducibility seed
    word_max_features : vocabulary size for word n-gram TF-IDF
    char_max_features : vocabulary size for char n-gram TF-IDF

    Returns
    -------
    oof_probs   : np.ndarray of shape (n_train, 3)
    test_probs  : np.ndarray of shape (n_test, 3)
    """
    n_train = len(train_df)
    n_test  = len(test_df)
    n_classes = 3

    oof_probs  = np.zeros((n_train, n_classes))
    test_probs = np.zeros((n_test, n_classes))

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True,
                          random_state=random_state)

    for fold, (train_idx, val_idx) in enumerate(
        skf.split(train_df, train_df[label_col])
    ):
        print(f"  [TF-IDF] Fold {fold + 1}/{n_splits}")

        X_train_text = train_df.iloc[train_idx][text_col]
        y_train      = train_df.iloc[train_idx][label_col]
        X_val_text   = train_df.iloc[val_idx][text_col]

        # Fit vectorisers on train fold only
        word_vec = TfidfVectorizer(
            ngram_range=(1, 2), max_features=word_max_features
        )
        char_vec = TfidfVectorizer(
            analyzer="char", ngram_range=(3, 5),
            max_features=char_max_features
        )
        word_vec.fit(X_train_text)
        char_vec.fit(X_train_text)

        X_train = hstack([word_vec.transform(X_train_text),
                          char_vec.transform(X_train_text)])
        X_val   = hstack([word_vec.transform(X_val_text),
                          char_vec.transform(X_val_text)])
        X_test  = hstack([word_vec.transform(test_df[text_col]),
                          char_vec.transform(test_df[text_col])])

        clf = LogisticRegression(max_iter=1000, class_weight="balanced")
        clf.fit(X_train, y_train)

        oof_probs[val_idx] += clf.predict_proba(X_val)
        test_probs          += clf.predict_proba(X_test) / n_splits

    print("  [TF-IDF] OOF generation complete.")
    return oof_probs, test_probs