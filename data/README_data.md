# Dataset — CHiPSAL 2026 Shared Task

## Overview

The dataset is from the **CHiPSAL 2026 Shared Task** on Nepali Meme Sentiment Classification.
It consists of Nepali memes, each with a textual component and a corresponding image.

Raw images and full CSVs are **not included in this repository** due to size and licensing.

---

## Label Mapping

| Integer Label | Sentiment |
|---|---|
| 0 | Negative |
| 1 | Neutral  |
| 2 | Positive |

---

## File Format

### `train.csv`
| Column | Type | Description |
|---|---|---|
| `index` | string | Image filename (e.g. `1234.jpg`) |
| `text` | string | Nepali text extracted from the meme |
| `label` | int | Sentiment label (0 / 1 / 2) |

### `index_text_test.csv`
| Column | Type | Description |
|---|---|---|
| `index` | string | Image filename |
| `text` | string | Nepali meme text |

---

## Statistics

| Split | Samples |
|---|---|
| Train | 1,061 |
| Test  | 133   |

### Train class distribution
| Class | Count | % |
|---|---|---|
| Negative | ~370 | ~35% |
| Neutral  | ~370 | ~35% |
| Positive | ~321 | ~30% |

---

## How to Obtain the Data

1. Register for the CHiPSAL 2026 Shared Task at the official website.
2. Download `train.csv`, `index_text_test.csv`, `train_images.zip`, `test_images.zip`.
3. Extract and place them as:

```
data/
├── train.csv
├── index_text_test.csv
├── train_images/
│   ├── 1001.jpg
│   ├── 1002.jpg
│   └── ...
└── test_images/
    ├── 5485.jpg
    ├── 5096.jpg
    └── ...
```

---

## Sample Files

`sample_train.csv` and `sample_test.csv` in this folder contain the first 20 rows
of each split — safe to inspect without downloading the full dataset.