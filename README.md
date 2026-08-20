# Financial Stance Classification with RoBERTa

A deep learning project utilizing fine-tuned **RoBERTa** models to classify financial communications and central bank (e.g., FOMC) policy statements into three distinct stance categories: **Dovish**, **Hawkish**, and **Neutral**.

---

## Table of Contents
- [Project Overview](#project-overview)
- [Key Features](#key-features)
- [Model Performance](#model-performance)
- [Installation & Setup](#installation--setup)
- [Repository Structure](#repository-structure)
- [Usage Guide](#usage-guide)
  - [1. Model Training](#1-model-training)
  - [2. Running Batch Inference](#2-running-batch-inference)
- [Implementation Details & Troubleshooting](#implementation-details--troubleshooting)
- [License](#license)

---

## Project Overview

Central bank announcements, press conference speeches, and policy meeting minutes carry significant predictive value for financial markets and macroeconomic trends. Manually parsing large volumes of central bank text is time-consuming and subjective. 

This repository provides an end-to-end NLP pipeline that leverages Hugging Face Transformers to automatically extract and score monetary policy stances from raw financial text.

---

## Key Features

- **Domain-Specific Fine-Tuning**: Fine-tuned Transformer backbone tailored for financial sentiment and policy tone detection.
- **Evaluation Metric Optimization**: Multi-class metric tracking using **Macro F1-Score** to handle class distribution shifts effectively.
- **Automated Early Stopping**: Configured with early stopping based on evaluation Macro F1 to prevent model overfitting.
- **Batch Inference Pipeline**: Simple interface for predicting stances on unlabelled statement datasets and exporting structured output.

---

## Model Performance

The fine-tuned RoBERTa model was evaluated on a test set comprising 573 unseen statement excerpts.

### Overall Summary Metrics
- **Validation Accuracy**: ~`71.90%`
- **Validation Macro F1**: ~`0.712`
- **Optimal Epoch**: `Epoch 4` *(Early stopping triggered after Epoch 7)*

### Classification Report (Test Set)

| Stance Category | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| **Dovish** | `0.69` | `0.71` | **`0.70`** | 173 |
| **Hawkish** | `0.64` | `0.71` | **`0.68`** | 148 |
| **Neutral** | `0.80` | `0.73` | **`0.76`** | 252 |
| **Macro Average** | **`0.71`** | **`0.72`** | **`0.71`** | **573** |

---

## Installation & Setup

1. **Clone the Repository**
   ```bash
   git clone https://github.com/your-username/financial-stance-roberta.git
   cd financial-stance-roberta
   ```

2. **Create a Virtual Environment & Install Dependencies**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

### Requirements (`requirements.txt`)
```text
torch>=2.0.0
transformers>=4.40.0
datasets>=2.14.0
scikit-learn>=1.2.0
pandas>=2.0.0
numpy>=1.24.0
```

---

## Repository Structure

```text
.
├── data/
│   ├── raw/                      # Raw financial statements & FOMC corpus
│   └── fomc_predictions.csv      # Batch prediction outputs
├── models/
│   └── roberta_combined/         # Saved model weights & tokenizer files
├── notebooks/
│   └── stance_classification.ipynb # End-to-end training & evaluation notebook
├── src/
│   ├── train.py                  # Training loop execution script
│   └── predict.py                # Batch inference script
├── .gitignore
├── README.md
└── requirements.txt
```

---

## Usage Guide

### 1. Model Training

When using modern versions of Hugging Face `transformers`, ensure that `eval_strategy` is used instead of the deprecated `evaluation_strategy` argument in `TrainingArguments`:

```python
from transformers import TrainingArguments, Trainer, AutoModelForSequenceClassification, AutoTokenizer

model_name = "roberta-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)

training_args = TrainingArguments(
    output_dir="./models/roberta_combined",
    eval_strategy="epoch",          # Updated Hugging Face keyword
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=7,
    load_best_model_at_end=True,
    metric_for_best_model="eval_macro_f1",
    greater_is_better=True,
    logging_steps=50,
)
```

### 2. Running Batch Inference

To run inference on new statements using the trained RoBERTa checkpoint:

```python
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

# Load fine-tuned model and tokenizer
model_path = "./models/roberta_combined/"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

# Initialize pipeline
classifier = pipeline("text-classification", model=model, tokenizer=tokenizer, device=0)

# Load data and run batch predictions
df = pd.read_csv("data/raw/fomc_statements.csv")
results = classifier(df["text"].tolist(), batch_size=32)

df["predicted_stance"] = [res["label"] for res in results]
df["confidence_score"] = [res["score"] for res in results]

# Export results
df.to_csv("data/fomc_predictions.csv", index=False)
print("Inference completed successfully!")
```

---

## Implementation Details & Troubleshooting

- **Hugging Face Syntax Update**: If you encounter a `TypeError: __init__() got an unexpected keyword argument 'evaluation_strategy'`, update your `TrainingArguments` block to use `eval_strategy="epoch"`.
- **Overfitting & Regularization**: If training loss drops rapidly while validation loss plateaus after Epoch 4, consider setting `weight_decay=0.01` and tuning `hidden_dropout_prob=0.2`.

---
