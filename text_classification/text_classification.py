import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import torch

# subset of categories that we will use
category_dict = {'comp.graphics': 'computer graphics',
                 'rec.sport.baseball': 'baseball',
                 'sci.electronics': 'science, electronics',
                 'talk.politics.guns': 'politics, guns'
                 }


def get_data(categories=None, portion=1.):
    """
    train and test data of the given categories, empty posts are removed
    portion: portion of the train data to use
    """
    from sklearn.datasets import fetch_20newsgroups
    data_train = fetch_20newsgroups(categories=categories, subset='train', remove=('headers', 'footers', 'quotes'),
                                    random_state=21)
    data_test = fetch_20newsgroups(categories=categories, subset='test', remove=('headers', 'footers', 'quotes'),
                                   random_state=21)

    train_len = int(portion*len(data_train.data))
    x_train = np.array(data_train.data[:train_len])
    y_train = data_train.target[:train_len]
    non_empty = x_train != ""
    x_train, y_train = x_train[non_empty].tolist(), y_train[non_empty].tolist()

    x_test = np.array(data_test.data)
    y_test = data_test.target
    non_empty = np.array(x_test) != ""
    x_test, y_test = x_test[non_empty].tolist(), y_test[non_empty].tolist()
    return x_train, y_train, x_test, y_test


def plot_epochs(values, title, ylabel):
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(values) + 1), values, label=ylabel)
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.legend()
    plt.show()


# single layer and one hidden layer models
def MLP_classification(portion=1., model=None):
    """
    trains a model on tfidf vectors for 20 epochs
    portion: portion of the train data to use
    model: the model to train, a single linear layer if not given
    returns the train loss and the validation accuracy of every epoch
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    import torch.optim as optim
    from torch.utils.data import DataLoader, Dataset

    class TfidfDataset(Dataset):
        def __init__(self, data, labels):
            self.data = torch.tensor(data, dtype=torch.float32)
            self.labels = torch.tensor(labels, dtype=torch.long)

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, idx):
            return self.data[idx], self.labels[idx]

    x_train, y_train, x_test, y_test = get_data(categories=category_dict.keys(), portion=portion)

    vectorizer = TfidfVectorizer(max_features=2000)
    train_tfidf = vectorizer.fit_transform(x_train).toarray()
    test_tfidf = vectorizer.transform(x_test).toarray()

    train_dataset = TfidfDataset(train_tfidf, y_train)
    test_dataset = TfidfDataset(test_tfidf, y_test)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    if model is None:
        model = nn.Sequential(
            nn.Linear(2000, 4)
        )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters is: {total_params}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    train_losses = []
    val_accuracies = []

    for epoch in range(20):
        model.train()
        total_loss = 0
        for data, labels in train_loader:
            data, labels = data.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        train_losses.append(total_loss / len(train_loader))

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data, labels in test_loader:
                data, labels = data.to(device), labels.to(device)
                outputs = model(data)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_accuracy = correct / total
        val_accuracies.append(val_accuracy)

        print(f"Epoch {epoch + 1}, Loss: {train_losses[-1]}, Accuracy: {val_accuracy}")

    plot_epochs(train_losses, f"Train Loss for Portion {portion}", "Loss")
    plot_epochs(val_accuracies, f"Validation Accuracy for Portion {portion}", "Accuracy")

    return train_losses, val_accuracies


# finetuned transformer
def transformer_classification(portion=1.):
    """
    finetunes distilroberta with a classification head for 3 epochs
    portion: portion of the train data to use
    returns the train loss and the validation accuracy of every epoch
    """
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from torch.utils.data import DataLoader, Dataset

    class TextDataset(Dataset):
        def __init__(self, encodings, labels):
            self.encodings = encodings
            self.labels = labels

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, idx):
            item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
            item['labels'] = torch.tensor(self.labels[idx])
            return item

    x_train, y_train, x_test, y_test = get_data(categories=category_dict.keys(), portion=portion)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained('distilroberta-base')
    model = AutoModelForSequenceClassification.from_pretrained('distilroberta-base', num_labels=len(category_dict))
    model.to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters is: {total_params}")

    train_dataset = TextDataset(tokenizer(x_train, truncation=True, padding=True), y_train)
    test_dataset = TextDataset(tokenizer(x_test, truncation=True, padding=True), y_test)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

    train_losses = []
    val_accuracies = []

    for epoch in range(3):
        model.train()
        total_loss = 0
        for batch in train_loader:
            batch = {key: val.to(device) for key, val in batch.items()}
            optimizer.zero_grad()
            # with the labels the model returns the cross entropy loss by itself
            loss = model(**batch).loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        train_losses.append(total_loss / len(train_loader))

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in test_loader:
                batch = {key: val.to(device) for key, val in batch.items()}
                predicted = model(**batch).logits.argmax(dim=1)
                total += batch['labels'].size(0)
                correct += (predicted == batch['labels']).sum().item()

        val_accuracy = correct / total
        val_accuracies.append(val_accuracy)

        print(f"Epoch {epoch + 1}, Loss: {train_losses[-1]}, Accuracy: {val_accuracy}")

    plot_epochs(train_losses, f"Transformer Train Loss for Portion {portion}", "Loss")
    plot_epochs(val_accuracies, f"Transformer Validation Accuracy for Portion {portion}", "Accuracy")

    return train_losses, val_accuracies


if __name__ == "__main__":
    if torch.cuda.is_available():
        print(f"CUDA is available. Device: {torch.cuda.get_device_name(0)}")
    else:
        print("CUDA is not available.")
    portions = [0.1, 0.2, 0.5, 1.]

    # single layer
    for portion in portions:
        print(f"MLP Classification with portion: {portion}")
        MLP_classification(portion=portion)

    # one hidden layer of 500
    for portion in portions:
        print(f"Multi-layer MLP Classification with portion: {portion}")
        model = nn.Sequential(
            nn.Linear(2000, 500),
            nn.ReLU(),
            nn.Linear(500, 4)
        )
        MLP_classification(portion=portion, model=model)

    # transformer, only on 10% and 20% of the data
    for portion in [0.1, 0.2]:
        print(f"Transformer Classification with portion: {portion}")
        transformer_classification(portion=portion)
