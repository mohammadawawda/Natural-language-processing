import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import BCEWithLogitsLoss
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModel, AutoTokenizer

import data_loader
from sentiment_analysis import get_available_device, plot_metrics, binary_accuracy

BATCH_SIZE = 64
LEARNING_RATE = 1e-5
WEIGHT_DECAY = 0
N_EPOCHS = 2
MAX_LENGTH = 64


class DistilRobertaSentiment(nn.Module):
    def __init__(self):
        super(DistilRobertaSentiment, self).__init__()
        self.distil_roberta = AutoModel.from_pretrained("distilroberta-base")
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(self.distil_roberta.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.distil_roberta(input_ids, attention_mask=attention_mask)
        # the first token (<s>) works like CLS
        pooled_output = outputs.last_hidden_state[:, 0, :]
        return self.fc(self.dropout(pooled_output))


def prepare_transformer_data(sentences, tokenizer, shuffle):
    """
    tokenizes the whole sentences and puts them in a DataLoader with their labels
    shuffle: only for the train set, so the test predictions keep the order of the sentences
    """
    texts = [" ".join(sent.text) for sent in sentences]
    labels = [sent.sentiment_class for sent in sentences]
    encodings = tokenizer(texts, padding=True, truncation=True, max_length=MAX_LENGTH, return_tensors="pt")
    dataset = TensorDataset(encodings["input_ids"], encodings["attention_mask"], torch.tensor(labels))
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=shuffle)


def run_epoch(model, loader, criterion, device, optimizer=None):
    """
    one pass over the data, it trains only when an optimizer is given
    returns the mean loss, the accuracy and the predictions
    """
    if optimizer:
        model.train()
    else:
        model.eval()

    total_loss, total_correct, total = 0, 0, 0
    predictions = []

    with torch.set_grad_enabled(optimizer is not None):
        for input_ids, attention_mask, labels in loader:
            input_ids, attention_mask = input_ids.to(device), attention_mask.to(device)
            labels = labels.to(device).float()

            outputs = model(input_ids, attention_mask).squeeze(1)
            loss = criterion(outputs, labels)
            if optimizer:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            probabilities = torch.sigmoid(outputs)
            total_loss += loss.item() * labels.size(0)
            total_correct += binary_accuracy(probabilities, labels) * labels.size(0)
            total += labels.size(0)
            predictions.append((probabilities >= 0.5).long().cpu().numpy())

    return total_loss / total, total_correct / total, np.concatenate(predictions)


if __name__ == "__main__":
    device = get_available_device()
    print(f"Using device: {device}")

    dataset = data_loader.SentimentTreeBank("stanfordSentimentTreebank", split_words=True)
    train_sentences = dataset.get_train_set_phrases()
    val_sentences = dataset.get_validation_set()
    test_sentences = dataset.get_test_set()

    tokenizer = AutoTokenizer.from_pretrained("distilroberta-base")
    train_loader = prepare_transformer_data(train_sentences, tokenizer, shuffle=True)
    val_loader = prepare_transformer_data(val_sentences, tokenizer, shuffle=False)
    test_loader = prepare_transformer_data(test_sentences, tokenizer, shuffle=False)

    model = DistilRobertaSentiment().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    criterion = BCEWithLogitsLoss()

    train_losses, train_accuracies, val_losses, val_accuracies = [], [], [], []
    for epoch in range(N_EPOCHS):
        train_loss, train_accuracy, _ = run_epoch(model, train_loader, criterion, device, optimizer)
        val_loss, val_accuracy, _ = run_epoch(model, val_loader, criterion, device)

        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

        print(f"Epoch {epoch + 1}/{N_EPOCHS}")
        print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")
        print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")

    plot_metrics(N_EPOCHS, train_losses, val_losses, "Loss", "Transformer: Loss")
    plot_metrics(N_EPOCHS, train_accuracies, val_accuracies, "Accuracy", "Transformer: Accuracy")

    test_loss, test_accuracy, predictions = run_epoch(model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

    labels = np.array([sent.sentiment_class for sent in test_sentences])
    negated = data_loader.get_negated_polarity_examples(test_sentences)
    rare = data_loader.get_rare_words_examples(test_sentences, dataset)
    print(f"Negated Polarity Accuracy: {np.mean(predictions[negated] == labels[negated]):.4f}")
    print(f"Rare Words Accuracy: {np.mean(predictions[rare] == labels[rare]):.4f}")
