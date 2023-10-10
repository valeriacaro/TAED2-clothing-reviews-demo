from models.model_dl import *
import time
import random
from sklearn.model_selection import train_test_split
import numpy as np


# HYPERPARAMETERS
hidden_size = 256
embedding_size = 128
batch_size = 512
token_size = 200000
epochs = 10
bidirectional = True
seed = 1111


def set_seed():

    """Set the random seed for reproducibility."""

    random.seed(seed)
    np.random.RandomState(seed)
    torch.manual_seed(seed)


def stemming(df, stem=True) -> tuple:

    """
        Preprocesses data and selects features and target based on the stemming flag.
        Args:
            df (DataFrame): The input DataFrame.
            stem (bool): A flag indicating whether stemming is applied.
        Returns:
            tuple: A tuple containing the feature variable (x) and the target variable (y).
    """

    if stem:
        x = df["Stemmed Review Text"].values
    else:
        x = df["Review Text"].values
    y = df["Top Product"].values
    return x, y


def create_word_vocab(x):

    """
        Create a vocabulary for word tokens.

        Args:
            x (list): List of input text data containing word tokens.

        Returns:
            tuple: A tuple containing the word vocabulary (vocab), pad token index (pad_index), and unknown token index (unk_index).
    """

    vocab = Dictionary()
    pad_token = '<pad>'
    unk_token = '<unk>'
    pad_index = vocab.add_token(pad_token)
    unk_index = vocab.add_token(unk_token)

    # Join the list of strings into a single string
    combined_text = ' '.join(str(item) for item in x)

    # Split the combined text into words
    words = combined_text.split()

    # Create a sorted tuple of unique words
    unique_sorted_words = sorted(set(words))

    # Add words to the custom vocabulary
    for word in unique_sorted_words:
        vocab.add_token(word)

    # words = set(' '.join(x).split())
    # for word in sorted(words):
    #     vocab.add_token(word)
    return vocab, pad_index, unk_index


def create_label_vocab(y):

    """
        Create a vocabulary for labels.

        Args:
            y (list): List of labels.

        Returns:
            Dictionary: The label vocabulary.
    """

    label_vocab = Dictionary()
    # Create a set of labels from the training labels (0 or 1)
    labels = set(y)
    for label in sorted(labels):
        label_vocab.add_token(label)
    return label_vocab


def convert_data_to_indices(x, y, vocab, label_vocab):

    """
        Convert data to token indices.

        Args:
            x (list): List of input text data containing word tokens.
            y (list): List of labels.
            vocab (Dictionary): Word vocabulary.
            label_vocab (Dictionary): Label vocabulary.

        Returns:
            tuple: A tuple containing the converted input data (x_idx) and labels (y_idx) as token indices.
    """

    x_idx = []

    for sentence in x:
        tokens = sentence.split()
        indexes = np.array([vocab.token2idx[word] for word in tokens])
        x_idx.append(indexes)

    y_idx = np.array([label_vocab.token2idx[label] for label in y])
    return x_idx, y_idx

def split_data(x, y):

    """
        Split data into training and validation sets.

        Args:
            x (list): List of input data.
            y (list): List of labels.

        Returns:
            tuple: A tuple containing training and validation data splits.
    """

    x_train, x_temp, y_train, y_temp = train_test_split(x, y, test_size=0.3, random_state=seed)
    x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, test_size=0.5, random_state=seed)

    return (x_train, y_train), (x_val, y_val), (x_test, y_test)


def batch_generator(data, batch_size, token_size):

    """
        Yield elements from data in chunks with a maximum of batch_size sequences and token_size tokens.

        Args:
            data (list): List of input sequences.
            batch_size (int): Maximum batch size.
            token_size (int): Maximum token size for each batch.

        Returns:
            generator: A generator that yields batches of data.
    """

    minibatch, sequences_so_far, tokens_so_far = [], 0, 0
    for ex in data:
        seq_len = len(ex[0])
        if seq_len > token_size:
            ex = (ex[0][:token_size], ex[1])
            seq_len = token_size
        minibatch.append(ex)
        sequences_so_far += 1
        tokens_so_far += seq_len
        if sequences_so_far == batch_size or tokens_so_far == token_size:
            yield minibatch
            minibatch, sequences_so_far, tokens_so_far = [], 0, 0
        elif sequences_so_far > batch_size or tokens_so_far > token_size:
            yield minibatch[:-1]
            minibatch, sequences_so_far, tokens_so_far = minibatch[-1:], 1, len(minibatch[-1][0])
    if minibatch:
        yield minibatch


def pool_generator(data, batch_size, token_size, shuffle=False):

    """
        Sort within buckets, then batch, then shuffle batches.

        Args:
            data (list): List of input sequences.
            batch_size (int): Maximum batch size.
            token_size (int): Maximum token size for each batch.
            shuffle (bool): Whether to shuffle the batches.

        Returns:
            generator: A generator that yields batches of data.
    """

    for p in batch_generator(data, batch_size * 100, token_size * 100):
        p_batch = batch_generator(sorted(p, key=lambda t: len(t[0]), reverse=True), batch_size, token_size)
        p_list = list(p_batch)
        if shuffle:
            for b in random.sample(p_list, len(p_list)):
                yield b
        else:
            for b in p_list:
                yield b

def get_model(ntokens, nlabels, pad_index):

    """
        Get the model and optimizer for training.

        Returns:
            tuple: A tuple containing the model and optimizer.
    """

    model = CharRNNClassifier(ntokens, embedding_size, hidden_size, nlabels, bidirectional=bidirectional,
                              pad_idx=pad_index)
    optimizer = torch.optim.Adam(model.parameters())
    return model, optimizer


def train(model, optimizer, data, batch_size, token_size, max_norm=1, log=False):

    """
        Train the model on the provided data.

        Args:
            model (torch.nn.Module): The model to be trained.
            optimizer (torch.optim.Optimizer): The optimizer for training.
            data (list): List of training data.
            batch_size (int): Maximum batch size.
            token_size (int): Maximum token size for each batch.
            max_norm (float): Maximum gradient norm for gradient clipping.
            log (bool): Whether to log training statistics.

        Returns:
            tuple: A tuple containing the training accuracy and loss.
    """

    model.train()
    total_loss = 0
    ncorrect = 0
    nsentences = 0
    ntokens = 0
    niterations = 0
    criterion = torch.nn.CrossEntropyLoss(reduction='sum')
    for batch in pool_generator(data, batch_size, token_size, shuffle=True):
        # Get input and target sequences from batch
        X = [torch.from_numpy(d[0]) for d in batch]
        X_lengths = [x.numel() for x in X]
        ntokens += sum(X_lengths)
        X_lengths = torch.tensor(X_lengths, dtype=torch.long)
        y = torch.tensor([d[1] for d in batch], dtype=torch.long)
        # Pad the input sequences to create a matrix
        X = torch.nn.utils.rnn.pad_sequence(X)
        model.zero_grad()
        output = model(X, X_lengths)
        loss = criterion(output, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),
                                       max_norm)  # Gradient clipping https://www.kaggle.com/c/wili4/discussion/231378
        optimizer.step()
        # Training statistics
        total_loss += loss.item()
        ncorrect += (torch.max(output, 1)[1] == y).sum().item()
        nsentences += y.numel()
        niterations += 1

    total_loss = total_loss / nsentences
    accuracy = 100 * ncorrect / nsentences
    if log:
        print(f'Train: wpb={ntokens // niterations}, bsz={nsentences // niterations}, num_updates={niterations}')
    return accuracy, total_loss


def validate(model, data, batch_size, token_size):
    """
        Validate the model on the provided data.

        Args:
            model (torch.nn.Module): The model to be validated.
            data (list): List of validation data.
            batch_size (int): Maximum batch size.
            token_size (int): Maximum token size for each batch.

        Returns:
            tuple: A tuple containing predictions and validation accuracy.
    """
    model.eval()
    # calculate accuracy on validation set
    ncorrect = 0
    nsentences = 0
    predictions = []
    with torch.no_grad():
        for batch in pool_generator(data, batch_size, token_size):
            # Get input and target sequences from batch
            X = [torch.from_numpy(d[0]) for d in batch]
            X_lengths = torch.tensor([x.numel() for x in X], dtype=torch.long)
            y = torch.tensor([d[1] for d in batch], dtype=torch.long)
            # Pad the input sequences to create a matrix
            X = torch.nn.utils.rnn.pad_sequence(X)
            answer = model(X, X_lengths)
            ncorrect += (torch.max(answer, 1)[1] == y).sum().item()
            nsentences += y.numel()
            # Add predictions to the list
            predictions.extend(torch.max(answer, 1)[1].tolist())

        dev_acc = 100 * ncorrect / nsentences

    return predictions, dev_acc


def model_training(train_data, val_data, ntokens, nlabels, pad_index):

    """
        Train the model on the training data.

        Args:
            train_data (list): List of training data.
            val_data (list): List of validation data.

        Returns:
            torch.nn.Module: The trained model.
    """

    train_accuracy = []
    valid_accuracy = []
    model, optimizer = get_model(ntokens, nlabels, pad_index)
    print(f'Training cross-validation model for {epochs} epochs')
    t0 = time.time()
    # Model training
    for epoch in range(1, epochs + 1):
        acc = train(model, optimizer, train_data, batch_size, token_size, log=epoch == 1)[0]
        train_accuracy.append(acc)
        print(f'| epoch {epoch:03d} | train accuracy={acc:.1f}% ({time.time() - t0:.0f}s)')
        acc = validate(model, val_data, batch_size, token_size)[1]
        valid_accuracy.append(acc)
        print(f'| epoch {epoch:03d} | valid accuracy={acc:.1f}%')
    return model

if __name__ == '__main__':

    pass