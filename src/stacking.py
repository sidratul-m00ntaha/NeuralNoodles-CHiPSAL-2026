import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, classification_report

CLASS_NAMES = ["Negative", "Neutral", "Positive"]

def build_meta_features(text_probs, image_probs, st_probs):
    """Concatenate OOF probability arrays from all 3 modalities."""
    return np.hstack([text_probs, image_probs, st_probs])

def train_meta_model(meta_features, labels):
    """Train Logistic Regression meta-learner on stacked OOF probs."""
    meta_model = LogisticRegression(max_iter=1000, class_weight='balanced')
    meta_model.fit(meta_features, labels)
    return meta_model

def evaluate(y_true, y_pred):
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    report = classification_report(y_true, y_pred, target_names=CLASS_NAMES)
    print(f"Macro-F1: {macro_f1:.4f}")
    print(report)
    return macro_f1