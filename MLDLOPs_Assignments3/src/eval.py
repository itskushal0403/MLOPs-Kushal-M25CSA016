import torch
from transformers import DistilBertForSequenceClassification, Trainer
from sklearn.metrics import accuracy_score, f1_score

from src.data import prepare_data


MODEL_REPO = "itskushal0403/Assignment-3_ML-DL_Ops-goodreads-distilbert-classifier"


class GoodreadsDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = logits.argmax(axis=-1)

    acc = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average="weighted")

    return {
        "accuracy": acc,
        "f1": f1,
    }


def evaluate():
    print("Loading dataset...")
    _, test_encodings, _, test_labels, _, _ = prepare_data()

    print("Loading model from Hugging Face Hub...")
    model = DistilBertForSequenceClassification.from_pretrained(MODEL_REPO)

    # Get correct label mapping from model config
    label2id = model.config.label2id
    id2label = model.config.id2label

    # Re-map test labels according to model mapping
    test_labels_mapped = [label2id[id2label[i]] for i in test_labels]

    test_dataset = GoodreadsDataset(test_encodings, test_labels_mapped)

    trainer = Trainer(
        model=model,
        compute_metrics=compute_metrics,
    )

    results = trainer.evaluate(test_dataset)

    print("\nEvaluation Results from Hugging Face model:")
    print(results)


if __name__ == "__main__":
    evaluate()
