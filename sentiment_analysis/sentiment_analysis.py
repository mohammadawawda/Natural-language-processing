import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os

from torch.utils.data import DataLoader, Dataset
import data_loader
import pickle
import matplotlib.pyplot as plt

SEQ_LEN = 52
W2V_EMBEDDING_DIM = 300
BATCH_SIZE = 64
ONEHOT_AVERAGE = "onehot_average"
W2V_AVERAGE = "w2v_average"
W2V_SEQUENCE = "w2v_sequence"

TRAIN = "train"
VAL = "val"
TEST = "test"


def get_available_device():
    # the gpu if there is one
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def save_pickle(obj, path):
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def save_model(model, path, epoch, optimizer):
    """
    saves the model and the optimizer so training can go on later
    """
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()}, path)


def load(model, path, optimizer):
    """
    loads a model and an optimizer that were saved with save_model
    """
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    return model, optimizer, epoch


def load_word2vec():
    # all 3 million word2vec vectors, each of length 300
    import gensim.downloader as api
    wv_from_bin = api.load("word2vec-google-news-300")
    print("Loaded vocab size %i" % len(wv_from_bin.key_to_index))
    return wv_from_bin


def create_or_load_slim_w2v(words_list, cache_w2v=False):
    """
    word2vec only for the words of our dataset
    words_list: the words to keep
    cache_w2v: save the small dictionary so next time we don't load all of word2vec again
    """
    w2v_path = "w2v_dict.pkl"
    if not os.path.exists(w2v_path):
        full_w2v = load_word2vec()
        w2v_emb_dict = {k: full_w2v[k] for k in words_list if k in full_w2v}
        if cache_w2v:
            save_pickle(w2v_emb_dict, w2v_path)
    else:
        w2v_emb_dict = load_pickle(w2v_path)
    return w2v_emb_dict


def get_w2v_average(sent, word_to_vec, embedding_dim):
    """
    the average word2vec vector of the words in the sentence
    words without a vector are left out of the average, if none has a vector we return zeros
    """
    embeddings = [word_to_vec[word] for word in sent.text if word in word_to_vec]
    if len(embeddings) == 0:
        return np.zeros(embedding_dim)
    return np.mean(embeddings, axis=0)


def get_one_hot(size, ind):
    """
    vector of zeros of length size with 1 at ind
    """
    one_hot = np.zeros(size)
    one_hot[ind] = 1
    return one_hot


def average_one_hots(sent, word_to_ind):
    """
    the average one hot vector of the words in the sentence
    """
    vocab_size = len(word_to_ind)
    sentence_embeddings = []

    for word in sent.text:
        if word in word_to_ind:
            sentence_embeddings.append(get_one_hot(vocab_size, word_to_ind[word]))

    if len(sentence_embeddings) == 0:
        return np.zeros(vocab_size)
    return np.mean(sentence_embeddings, axis=0)


def get_word_to_ind(words_list):
    # every word gets its index in the list
    return {word: idx for idx, word in enumerate(words_list)}


def sentence_to_embedding(sent, word_to_vec, seq_len, embedding_dim=300):
    """
    the word2vec vectors of the words in the sentence, as an array of shape (seq_len, embedding_dim)
    unknown words get zeros, long sentences are cut and short ones are padded with zeros
    """
    embeddings = []
    for word in sent.text:
        embeddings.append(word_to_vec.get(word, np.zeros(embedding_dim)))

    if len(embeddings) > seq_len:
        embeddings = embeddings[:seq_len]
    else:
        embeddings.extend([np.zeros(embedding_dim)] * (seq_len - len(embeddings)))

    return np.array(embeddings)


class OnlineDataset(Dataset):
    """
    dataset that builds the input of every sentence when it is asked for
    """

    def __init__(self, sent_data, sent_func, sent_func_kwargs):
        """
        sent_data: list of sentences from SentimentTreeBank
        sent_func: turns a sentence into the input of the model
        sent_func_kwargs: the fixed arguments of sent_func
        """
        self.data = sent_data
        self.sent_func = sent_func
        self.sent_func_kwargs = sent_func_kwargs

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sent = self.data[idx]
        sent_emb = self.sent_func(sent, **self.sent_func_kwargs)
        sent_label = sent.sentiment_class
        return sent_emb, sent_label


class DataManager:
    """
    loads the data and gives the batches for training and evaluation
    """

    def __init__(self, data_type=ONEHOT_AVERAGE, use_sub_phrases=True, dataset_path="stanfordSentimentTreebank",
                 batch_size=50, embedding_dim=None):
        """
        data_type: ONEHOT_AVERAGE, W2V_AVERAGE or W2V_SEQUENCE
        use_sub_phrases: train on all the sub phrases too and not only the full sentences
        dataset_path: the folder of the dataset
        batch_size: number of examples in a batch
        embedding_dim: only for the word2vec types
        """
        self.sentiment_dataset = data_loader.SentimentTreeBank(dataset_path, split_words=True)

        self.sentences = {}
        if use_sub_phrases:
            self.sentences[TRAIN] = self.sentiment_dataset.get_train_set_phrases()
        else:
            self.sentences[TRAIN] = self.sentiment_dataset.get_train_set()

        self.sentences[VAL] = self.sentiment_dataset.get_validation_set()
        self.sentences[TEST] = self.sentiment_dataset.get_test_set()

        words_list = list(self.sentiment_dataset.get_word_counts().keys())
        if data_type == ONEHOT_AVERAGE:
            self.sent_func = average_one_hots
            self.sent_func_kwargs = {"word_to_ind": get_word_to_ind(words_list)}
        elif data_type == W2V_SEQUENCE:
            self.sent_func = sentence_to_embedding
            self.sent_func_kwargs = {
                "seq_len": SEQ_LEN,
                "word_to_vec": create_or_load_slim_w2v(words_list, cache_w2v=True),
                "embedding_dim": embedding_dim
            }
        elif data_type == W2V_AVERAGE:
            self.sent_func = get_w2v_average
            self.sent_func_kwargs = {
                "word_to_vec": create_or_load_slim_w2v(words_list, cache_w2v=True),
                "embedding_dim": embedding_dim
            }
        else:
            raise ValueError("Invalid data_type: {}".format(data_type))

        self.torch_datasets = {
            k: OnlineDataset(sentences, self.sent_func, self.sent_func_kwargs)
            for k, sentences in self.sentences.items()
        }

        # only the train set is shuffled so the test predictions keep the order of the sentences
        self.torch_iterators = {
            k: DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=(k == TRAIN),
                num_workers=os.cpu_count(),
                pin_memory=True
            )
            for k, dataset in self.torch_datasets.items()
        }

    def get_torch_iterator(self, data_subset=TRAIN):
        return self.torch_iterators[data_subset]

    def get_labels(self, data_subset=TRAIN):
        # the labels in the same order as the iterator
        return np.array([sent.sentiment_class for sent in self.sentences[data_subset]])

    def get_input_shape(self):
        # the shape of one input (without the label)
        return self.torch_datasets[TRAIN][0][0].shape


class LSTM(nn.Module):
    """
    bidirectional lstm on the word2vec vectors, the last hidden state of both directions goes to a linear layer
    """

    def __init__(self, embedding_dim, hidden_dim, n_layers, dropout):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            bidirectional=True
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(2 * hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, text):
        # text: (batch_size, seq_len, embedding_dim), returns the logits (batch_size, 1)
        lstm_out, _ = self.lstm(text)
        # the forward direction ends at the last word and the backward direction ends at the first word
        hidden_fw = lstm_out[:, -1, :self.hidden_dim]
        hidden_bw = lstm_out[:, 0, self.hidden_dim:]
        hidden = torch.cat((hidden_fw, hidden_bw), dim=1)
        hidden = self.dropout(hidden)
        return self.fc(hidden)

    def predict(self, text):
        # 0 or 1 for every sentence
        probabilities = self.sigmoid(self.forward(text))
        return (probabilities >= 0.5).long()


class LogLinear(nn.Module):
    """
    one linear layer, used for both the one hot and the word2vec models
    """

    def __init__(self, embedding_dim):
        super(LogLinear, self).__init__()
        self.fc = nn.Linear(embedding_dim, 1)

    def forward(self, x):
        # the logits, the sigmoid is done inside the loss
        return self.fc(x)

    def predict(self, x):
        # 0 or 1 for every sentence
        probabilities = torch.sigmoid(self.forward(x))
        return (probabilities >= 0.5).long()


def binary_accuracy(preds, y):
    """
    preds: probabilities (after the sigmoid, not rounded)
    y: the true labels, 0 or 1
    returns the part of the predictions that are right
    """
    rounded_preds = (preds >= 0.5).long()
    correct = (rounded_preds == y.long()).float().sum()
    return (correct / y.size(0)).item()


def train_epoch(model, data_iterator, optimizer, criterion, device):
    """
    one pass over the train data, returns the mean loss and accuracy
    """
    model.train()
    epoch_loss = 0
    epoch_accuracy = 0
    total_examples = 0

    for inputs, labels in data_iterator:
        inputs, labels = inputs.to(device).float(), labels.to(device).float()

        optimizer.zero_grad()
        # squeeze(1) and not squeeze() so a batch of one example keeps its shape
        outputs = model(inputs).squeeze(1)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item() * labels.size(0)
        epoch_accuracy += binary_accuracy(torch.sigmoid(outputs), labels) * labels.size(0)
        total_examples += labels.size(0)

    return epoch_loss / total_examples, epoch_accuracy / total_examples


def evaluate(model, data_iterator, criterion, device):
    """
    the mean loss and accuracy on the data, without changing the model
    """
    model.eval()
    epoch_loss = 0
    epoch_accuracy = 0
    total_examples = 0

    with torch.no_grad():
        for inputs, labels in data_iterator:
            inputs, labels = inputs.to(device).float(), labels.to(device).float()
            outputs = model(inputs).squeeze(1)
            loss = criterion(outputs, labels)

            epoch_loss += loss.item() * labels.size(0)
            epoch_accuracy += binary_accuracy(torch.sigmoid(outputs), labels) * labels.size(0)
            total_examples += labels.size(0)

    return epoch_loss / total_examples, epoch_accuracy / total_examples


def get_predictions_for_data(model, data_iter, device):
    """
    the predictions of the model (0 or 1) for all the examples, in the order of data_iter
    """
    model.eval()
    predictions = []

    with torch.no_grad():
        for inputs, labels in data_iter:
            predicted = model.predict(inputs.to(device).float()).squeeze(1)
            predictions.append(predicted.cpu().numpy())

    return np.concatenate(predictions, axis=0)


def train_model(model, data_manager, n_epochs, lr, weight_decay, device, model_name):
    """
    trains the model and returns the train and validation loss and accuracy of every epoch
    """
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.BCEWithLogitsLoss()

    train_losses, train_accuracies = [], []
    val_losses, val_accuracies = [], []

    for epoch in range(n_epochs):
        train_loss, train_accuracy = train_epoch(
            model, data_manager.get_torch_iterator(TRAIN), optimizer, criterion, device
        )
        val_loss, val_accuracy = evaluate(
            model, data_manager.get_torch_iterator(VAL), criterion, device
        )

        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

        print(f"Epoch {epoch + 1}/{n_epochs}")
        print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")
        print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")

    return train_losses, train_accuracies, val_losses, val_accuracies


def plot_metrics(epochs, train_vals, val_vals, ylabel, title):
    # train and validation on the same graph
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, epochs + 1), train_vals, label="Train")
    plt.plot(range(1, epochs + 1), val_vals, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid()
    plt.show()


def plot_results(results, name):
    train_losses, train_accuracies, val_losses, val_accuracies = results
    plot_metrics(len(train_losses), train_losses, val_losses, "Loss", f"{name}: Loss")
    plot_metrics(len(train_accuracies), train_accuracies, val_accuracies, "Accuracy", f"{name}: Accuracy")


def test_and_special_subsets(model, data_manager, device):
    """
    test loss and accuracy, and the accuracy on the negated polarity and rare words sentences
    """
    criterion = nn.BCEWithLogitsLoss()
    test_loss, test_accuracy = evaluate(model, data_manager.get_torch_iterator(TEST), criterion, device)
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

    test_sentences = data_manager.sentences[TEST]
    predictions = get_predictions_for_data(model, data_manager.get_torch_iterator(TEST), device)
    labels = data_manager.get_labels(TEST)

    negated = data_loader.get_negated_polarity_examples(test_sentences)
    rare = data_loader.get_rare_words_examples(test_sentences, data_manager.sentiment_dataset)
    print(f"Negated Polarity Accuracy: {np.mean(predictions[negated] == labels[negated]):.4f}")
    print(f"Rare Words Accuracy: {np.mean(predictions[rare] == labels[rare]):.4f}")


def train_log_linear_with_one_hot(device):
    data_manager = DataManager(data_type=ONEHOT_AVERAGE, batch_size=BATCH_SIZE)
    model = LogLinear(data_manager.get_input_shape()[0])
    results = train_model(model, data_manager, 20, 0.01, 0.001, device, "log_linear_one_hot")
    plot_results(results, "One Hot Log Linear Model")
    test_and_special_subsets(model, data_manager, device)


def train_log_linear_with_w2v(device):
    data_manager = DataManager(data_type=W2V_AVERAGE, batch_size=BATCH_SIZE, embedding_dim=W2V_EMBEDDING_DIM)
    model = LogLinear(W2V_EMBEDDING_DIM)
    results = train_model(model, data_manager, 20, 0.01, 0.001, device, "log_linear_w2v")
    plot_results(results, "Word2Vec Log Linear Model")
    test_and_special_subsets(model, data_manager, device)


def train_lstm_with_w2v(device):
    data_manager = DataManager(data_type=W2V_SEQUENCE, batch_size=BATCH_SIZE, embedding_dim=W2V_EMBEDDING_DIM)
    model = LSTM(W2V_EMBEDDING_DIM, hidden_dim=100, n_layers=1, dropout=0.5)
    results = train_model(model, data_manager, 4, 0.001, 0.0001, device, "lstm_w2v")
    plot_results(results, "Word2Vec LSTM Model")
    test_and_special_subsets(model, data_manager, device)


if __name__ == "__main__":
    device = get_available_device()
    print(f"Using device: {device}")

    print("\nTraining Log Linear Model with One Hot Encoding")
    train_log_linear_with_one_hot(device)

    print("\nTraining Log Linear Model with Word2Vec")
    train_log_linear_with_w2v(device)

    print("\nTraining LSTM Model with Word2Vec")
    train_lstm_with_w2v(device)
