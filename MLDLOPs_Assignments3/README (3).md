# Assignment 3: End-to-End Hugging Face Model Training & Docker Deployment

**Course:** ML-DL-Ops  
**Student:** Kushal | M25CSA016  
**Branch:** `Assignment-3`

---

## 📌 Overview

This project implements a complete, production-ready machine learning workflow for **book genre classification** using the Goodreads dataset. The pipeline covers fine-tuning a pre-trained transformer model, containerizing the workflow with Docker, publishing the trained model to Hugging Face Hub, and evaluating it end-to-end — all following MLOps best practices.

---

## 🔗 Important Links

| Resource | Link |
|---|---|
| 🤗 Hugging Face Model | [itskushal0403/Assignment-3\_ML-DL\_Ops-goodreads-distilbert-classifier](https://huggingface.co/itskushal0403/Assignment-3_ML-DL_Ops-goodreads-distilbert-classifier) |
| 💻 GitHub Repository | [MLOPs-Kushal-M25CSA016 / Assignment-3](https://github.com/itskushal0403/MLOPs-Kushal-M25CSA016/blob/Assignment-3/MLDLOPs_Assignments3) |

---

## 🧠 Model Details

| Property | Value |
|---|---|
| **Base Model** | `distilbert-base-cased` |
| **Architecture** | DistilBertForSequenceClassification |
| **Task** | Single-label Text Classification |
| **Dataset** | Goodreads Book Reviews |
| **Number of Labels** | 8 |
| **Model Size** | 65.8M parameters |
| **Tensor Type** | Float32 (Safetensors) |
| **Transformers Version** | 4.41.2 |
| **Max Sequence Length** | 512 tokens |
| **Vocab Size** | 28,996 |
| **Hidden Dimension** | 768 |
| **Transformer Layers** | 6 |
| **Attention Heads** | 12 |
| **FFN Hidden Dim** | 3,072 |

### 🏷️ Label Classes

| ID | Label |
|---|---|
| 0 | `children` |
| 1 | `comics_graphic` |
| 2 | `fantasy_paranormal` |
| 3 | `history_biography` |
| 4 | `mystery_thriller_crime` |
| 5 | `poetry` |
| 6 | `romance` |
| 7 | `young_adult` |

### Why DistilBERT?

DistilBERT was selected for the following reasons:

- **Efficiency:** It is ~40% smaller and ~60% faster than BERT-base while retaining ~97% of BERT's performance on most NLP benchmarks — making it ideal for fine-tuning in a resource-constrained environment.
- **Strong baseline for classification:** DistilBERT has been widely validated on text classification tasks and is well-suited for book review genre prediction.
- **Cased variant:** The `distilbert-base-cased` variant preserves capitalization signals (e.g., proper nouns in titles/authors), which is meaningful in literary text.
- **Transformer ecosystem compatibility:** Native support via Hugging Face `transformers` makes it easy to integrate with the Trainer API, tokenizer, and Hub push workflows.

---

## 📁 Project Structure

```
MLDLOPs_Assignments3/
│
├── data/                   # Data loading and preprocessing utilities
│   └── data.py
│
├── modules/                # Modular source scripts
│   ├── train.py            # Model training script (Trainer API)
│   ├── eval.py             # Evaluation script
│   └── utils.py            # Helper utilities
│
├── Dockerfile              # Docker image definition (training + evaluation)
├── Dockerfile.eval         # Production Docker image (eval-only, pulls from HF Hub)
├── requirements.txt        # Python dependencies
│
├── ML_DL_Ops_Ass_3-Fine-Tuning-Classification.ipynb  # Original notebook
│
└── evaluation_results/     # Saved evaluation metrics (JSON/CSV)
    └── results.json
```

---

## ⚙️ Setup & Installation

### Prerequisites

- Python 3.9+
- Docker Desktop (or Docker Engine on Linux)
- Git
- A Hugging Face account with an access token

---

## 🐳 Docker Usage

### Task 2 — Build the Training Container

```bash
# Clone the repository and switch to the assignment branch
git clone https://github.com/itskushal0403/MLOPs-Kushal-M25CSA016.git
cd MLOPs-Kushal-M25CSA016
git checkout Assignment-3
cd MLDLOPs_Assignments3

# Build the Docker image
docker build -t goodreads-distilbert:train .

# Run the container interactively
docker run --rm -it goodreads-distilbert:train bash

# Verify Python and libraries inside container
python --version
python -c "import transformers, torch, datasets; print('All dependencies OK')"
```

### Task 9 — Production Evaluation-Only Container

This image pulls the fine-tuned model directly from the Hugging Face Hub and runs evaluation automatically on startup.

```bash
# Build the evaluation-only Docker image
docker build -f Dockerfile.eval -t goodreads-distilbert:eval .

# Run — evaluation starts automatically
docker run --rm goodreads-distilbert:eval
```

---

## 🚀 Running the Pipeline Locally

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the model

```bash
python modules/train.py
```

### 3. Evaluate locally saved model

```bash
python modules/eval.py --model_path ./saved_model
```

### 4. Evaluate from Hugging Face Hub

```bash
python modules/eval.py --model_path itskushal0403/Assignment-3_ML-DL_Ops-goodreads-distilbert-classifier
```

---

## 📦 requirements.txt

```
transformers==4.41.2
datasets
torch
scikit-learn
accelerate
huggingface_hub
evaluate
numpy
pandas
```

---

## 🏋️ Training Summary (Task 5)

Training was performed using the Hugging Face **Trainer API** with the following configuration:

| Parameter | Value |
|---|---|
| **Optimizer** | AdamW |
| **Learning Rate Scheduler** | Linear with warmup |
| **Task Type** | Single-label sequence classification |
| **Evaluation Strategy** | Per epoch |
| **Metric** | Accuracy, F1 |
| **Logging** | Training loss logged per step |

The dataset was split into **train / validation / test** sets, and standard tokenization using `distilbert-base-cased`'s tokenizer was applied with padding and truncation to a max length of 512 tokens.

---

## 📊 Evaluation Results (Task 6 & 8)

Evaluation was run in two stages: against the **locally saved model** and against the **model loaded from Hugging Face Hub**.

| Metric | Local Model | HF Hub Model |
|---|---|---|
| **Accuracy** | _(see results.json)_ | _(see results.json)_ |
| **F1 Score** | _(see results.json)_ | _(see results.json)_ |
| **Loss** | _(see results.json)_ | _(see results.json)_ |

> Both evaluations produced consistent metrics, confirming that the model was pushed and loaded correctly without any degradation.

Evaluation results are saved to `evaluation_results/results.json`.

---

## 🤗 Pushing Model to Hugging Face (Task 7)

The fine-tuned model, tokenizer, and training config were pushed to the Hub using the following approach:

```python
from huggingface_hub import HfApi

model.push_to_hub("itskushal0403/Assignment-3_ML-DL_Ops-goodreads-distilbert-classifier")
tokenizer.push_to_hub("itskushal0403/Assignment-3_ML-DL_Ops-goodreads-distilbert-classifier")
trainer.push_to_hub()
```

The model is publicly accessible at:  
👉 https://huggingface.co/itskushal0403/Assignment-3_ML-DL_Ops-goodreads-distilbert-classifier

---

## 🔄 Loading Model from Hub (Task 8)

```python
from transformers import pipeline

classifier = pipeline(
    "text-classification",
    model="itskushal0403/Assignment-3_ML-DL_Ops-goodreads-distilbert-classifier"
)

result = classifier("A thrilling mystery set in Victorian England with unexpected twists.")
print(result)
# [{'label': 'mystery_thriller_crime', 'score': 0.91}]
```

---

## 📋 Assignment Task Checklist

| Task | Description | Status |
|---|---|---|
| Task 1 | Download shared notebook | ✅ |
| Task 2 | Create environment using Docker | ✅ |
| Task 3 | Convert notebook to Python scripts | ✅ |
| Task 4 | Load model from Hugging Face | ✅ |
| Task 5 | Train model using Trainer API | ✅ |
| Task 6 | Evaluate model | ✅ |
| Task 7 | Save model to Hugging Face profile | ✅ |
| Task 8 | Re-evaluate model from HF repo | ✅ |
| Task 9 | Create final Docker image (eval-only) | ✅ |
| Task 10 | Push everything to GitHub | ✅ |

---

## ⚠️ Challenges & Notes

- **Pickle warning on `training_args.bin`:** The Hugging Face Hub flags this file for containing pickle imports. This is standard behaviour for Hugging Face `TrainingArguments` serialization; the safetensors model weights themselves are fully safe.
- **Docker base image selection:** A PyTorch-based base image (`python:3.9-slim`) was chosen and dependencies installed manually to keep the image lightweight while still supporting GPU-optional training.
- **Dataset size vs. training time:** Training on the full Goodreads dataset can be time-intensive; a subset was used for rapid iteration before full training runs.

---

## 👤 Author

**Kushal**  
M25CSA016  
ML-DL-Ops Course, Assignment 3
