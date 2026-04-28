# NeuralNoodles @ CHiPSAL 2026

> **Late-Fusion Multimodal Stacking for Nepali Meme Sentiment Classification**

[![Python](https://img.shields.io/badge/Python-3.12-blue)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.9-orange)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Leaderboard](https://img.shields.io/badge/CHiPSAL%202026-6th%20Place-gold)](https://chipsal2026.example.org)

---

## Overview

This repository contains the official implementation of the **NeuralNoodles** system submitted to the **CHiPSAL 2026 Shared Task** on Nepali meme sentiment classification. Our paper was accepted at the workshop.

We frame the task as **three-class sentiment classification** (Negative / Neutral / Positive) over multimodal Nepali memes — combining textual and visual signals through a **late-fusion stacking ensemble**.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        INPUT MEME                               │
│              Text (Nepali)          Image                        │
└────────────────┬────────────────────────┬───────────────────────┘
                 │                        │
       ┌─────────▼──────────┐    ┌───────▼────────────┐
       │  TEXT BRANCH        │    │  IMAGE BRANCH       │
       │                     │    │                     │
       │ ┌─────────────────┐ │    │  EfficientNet-B0    │
       │ │ TF-IDF (word +  │ │    │  (ImageNet pre-     │
       │ │ char n-grams)   │ │    │   trained, fine-    │
       │ │ + Logistic Reg  │ │    │   tuned 4 epochs)   │
       │ └─────────────────┘ │    │                     │
       │ ┌─────────────────┐ │    │  5-fold Stratified  │
       │ │ Sentence Trans- │ │    │  CV → OOF probs     │
       │ │ former (MiniLM) │ │    │                     │
       │ │ + Logistic Reg  │ │    └───────┬─────────────┘
       │ └─────────────────┘ │            │
       │                     │            │
       │  5-fold → OOF probs │            │
       └─────────┬───────────┘            │
                 │                        │
                 └──────────┬─────────────┘
                            │
              ┌─────────────▼──────────────┐
              │   STACKING META-LEARNER     │
              │   (Logistic Regression on   │
              │    concatenated OOF probs   │
              │    from all 3 modalities)   │
              └─────────────┬──────────────┘
                            │
              ┌─────────────▼──────────────┐
              │     FINAL PREDICTION        │
              │  Negative / Neutral /       │
              │  Positive                   │
              └─────────────────────────────┘
```

---

## Results

### Model Comparison (OOF Macro-F1)

| Model                      | Macro-F1   |
| -------------------------- | ---------- |
| TF-IDF Logistic Regression | 0.3838     |
| SentenceTransformer + LR   | 0.3774     |
| EfficientNet-B0 (Image)    | 0.3788     |
| **Final Stacked Model**    | **0.4234** |

### Per-Class Performance (Final Stacked Model)

| Class    | Precision | Recall | F1-Score |
| -------- | --------- | ------ | -------- |
| Negative | 0.4447    | 0.5777 | 0.5026   |
| Neutral  | 0.5375    | 0.3636 | 0.4338   |
| Positive | 0.3054    | 0.3684 | 0.3339   |

### Ablation Study — Modality Combinations

| Combination                              | Macro-F1   |
| ---------------------------------------- | ---------- |
| TF-IDF + Image                           | 0.3878     |
| SentenceTransformer + Image              | 0.3897     |
| TF-IDF + SentenceTransformer             | 0.3715     |
| **TF-IDF + SentenceTransformer + Image** | **0.4234** |

### Cross-Fold Stability

| Fold           | Macro-F1            |
| -------------- | ------------------- |
| Fold 1         | 0.4193              |
| Fold 2         | 0.4210              |
| Fold 3         | 0.4548              |
| Fold 4         | 0.4078              |
| Fold 5         | 0.4072              |
| **Mean ± Std** | **0.4220 ± 0.0180** |

---

## Repository Structure

```
NeuralNoodles-CHiPSAL-2026/
├── README.md
├── requirements.txt
├── .gitignore
├── LICENSE
│
├── data/
│   ├── README_data.md          # Dataset description and download instructions
│   ├── sample_train.csv        # First 20 rows (safe to commit)
│   └── sample_test.csv         # First 20 rows of test set
│
├── notebooks/
│   └── neuralnoodles_chipsal2026.ipynb   # Full pipeline (Kaggle-ready)
│
├── src/
│   ├── __init__.py
│   ├── dataset.py              # ImageDataset, transforms, DataLoader
│   ├── text_tfidf.py           # TF-IDF + LogReg, 5-fold OOF
│   ├── text_sentence_transformer.py  # SentenceTransformer + LR, 5-fold OOF
│   ├── image_efficientnet.py   # EfficientNet-B0 fine-tuning, 5-fold OOF
│   ├── stacking.py             # Meta-learner: build features, train, evaluate
│   └── predict.py              # Generate final submission CSV
│
├── outputs/
│   ├── confusion_matrix.png
│   ├── methodology_diagram.png
│
└── paper/
    ├── NeuralNoodles_CHiPSAL2026.pdf    # Accepted paper
```

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/sidratul-m00ntaha/NeuralNoodles-CHiPSAL-2026.git
cd NeuralNoodles-CHiPSAL-2026
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Prepare data

Download the CHiPSAL 2026 dataset and place files as follows:

```
data/
├── train.csv               # columns: index, text, label
├── index_text_test.csv     # columns: index, text
├── train_images/           # meme images for training
└── test_images/            # meme images for test
```

See `data/README_data.md` for full instructions and label mapping.

### 4. Run the full pipeline

```bash
# Option A: Jupyter notebook (recommended)
jupyter notebook notebooks/neuralnoodles_chipsal2026.ipynb

# Option B: Run on Kaggle
# Upload the notebook directly — paths auto-detect Kaggle environment
```

### 5. Generate submission

```bash
python src/predict.py
# Output: final_submission.csv
```

---

## Dataset

- **Task:** CHiPSAL 2026 Shared Task — Nepali Meme Sentiment
- **Train:** 1,061 multimodal samples (text + image pairs)
- **Test:** 133 samples
- **Labels:** 0 = Negative, 1 = Neutral, 2 = Positive
- **Language:** Nepali (text; may include code-switching)

> **Note:** Raw images are excluded from this repository due to size and licensing. Follow `data/README_data.md` to obtain the dataset from the official task organisers.

---

## Key Design Decisions

- **Late fusion over early fusion:** Each modality is trained independently with 5-fold CV to produce out-of-fold (OOF) probability vectors. This avoids overfitting and enables reliable meta-learning.
- **OOF stacking:** The meta-learner (Logistic Regression) is trained on OOF predictions, not on the full training set predictions, making the stacking unbiased.
- **Multilingual SentenceTransformer:** `paraphrase-multilingual-MiniLM-L12-v2` handles Nepali text without task-specific fine-tuning.
- **Class-balanced training:** All classifiers use `class_weight='balanced'` to handle label imbalance.
- **EfficientNet-B0:** Lightweight but effective vision backbone; 4 epochs of fine-tuning is sufficient given the small dataset (1,061 samples).

---

## Citation

If you use this work, please cite:

```bibtex
@inproceedings{neuralnoodles2026chipsal,
  title     = {Late-Fusion Multimodal Stacking for Nepali Meme
               Sentiment Classification},
  author    = {[Your Name] and [Co-author Names]},
  booktitle = {Proceedings of the Workshop on Challenges in Processing
               South Asian Languages (CHiPSAL 2026)},
  year      = {2026}
}
```

---

## Team

**NeuralNoodles** — [Chittagong University of Engineering & Technology]

- [Sidratul Muntaha] — [u21041193@student.cuet.ac.bd]
- [Sabila Anzum] — [u2104062@student.cuet.ac.bd]
- [Arpita Mallik] — [u2004023@student.cuet.ac.bd]
- [Hasan Murad] — [hasanmurad@cuet.ac.bd]

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
