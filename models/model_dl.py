# IMPORTS
import argparse
import mlflow.pytorch
import random
import pandas as pd
import torch  # Deep learning framework
import time
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


# FUNCTIONS
def set_seed():
    """Set the random seed for reproducibility."""
    random.seed(seed)
    np.random.RandomState(seed)
    torch.manual_seed(seed)


def read_data(path_to_data:str) -> pd.DataFrame:
    """
    Reads data from csv and creates a DataFrame from it.

    Args:
       path_to_data: Path where data we want can be found

    Returns:
        DataFrame: The preprocessed data in a pandas DataFrame.
    """
    dataframe = pd.read_csv(path_to_data)
    return dataframe


class Dictionary(object):
    def __init__(self):
        self.token2idx = {}
        self.idx2token = []

    def add_token(self, token):
        """
            Add a token to the dictionary and return its index.

            Args:
                token (str): The token to be added to the dictionary.

            Returns:
                int: The index of the added token.
        """
        if token not in self.token2idx:
            self.idx2token.append(token)
            self.token2idx[token] = len(self.idx2token) - 1
        return self.token2idx[token]

    def __len__(self):
        """
            Get the length of the dictionary.

            Returns:
                int: The number of tokens in the dictionary.
        """
        return len(self.idx2token)


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
        x = df["Stemmed Review Text"]
    else:
        x = df["Review Text"]
    y = df["Top Product"]
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

    words = set(' '.join(x).split())
    for word in sorted(words):
        vocab.add_token(word)
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
    x_idx = [np.array([vocab.token2idx[word] for word in line.split()]) for line in x]
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
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.15, random_state=seed)
    return (x_train, y_train), (x_val, y_val)


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


class CharRNNClassifier(torch.nn.Module):

    def __init__(self, input_size, embedding_size, hidden_size, output_size, model="lstm", num_layers=4,
                 bidirectional=False, pad_idx=0):
        """
               Initialize the Character RNN Classifier model.

               Args:
                   input_size (int): The size of the input vocabulary.
                   embedding_size (int): The size of word embeddings.
                   hidden_size (int): The size of the hidden layers.
                   output_size (int): The size of the output (number of classes).
                   model (str): The RNN model type ("lstm" or "gru").
                   num_layers (int): The number of RNN layers.
                   bidirectional (bool): Whether to use bidirectional RNN.
                   pad_idx (int): The padding index for embeddings.

               Returns:
                   None
        """
        super().__init__()
        self.model = model.lower()
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        self.embed = torch.nn.Embedding(input_size, embedding_size, padding_idx=pad_idx)
        if self.model == "gru":
            self.rnn = torch.nn.GRU(embedding_size, hidden_size, num_layers, bidirectional=bidirectional)
        elif self.model == "lstm":
            self.rnn = torch.nn.LSTM(embedding_size, 2*hidden_size, num_layers, bidirectional=bidirectional)
        self.h2o = torch.nn.Linear(self.num_directions * hidden_size, output_size)
        self.dropout = torch.nn.Dropout(0.2, inplace=True)


    def forward(self, input, input_lengths):
        """
                Forward pass of the model.

                Args:
                    input (Tensor): Input sequences.
                    input_lengths (Tensor): Lengths of input sequences.

                Returns:
                    Tensor: Model output.
        """
        # T x B
        encoded = self.embed(input)
        # T x B x E
        packed = torch.nn.utils.rnn.pack_padded_sequence(encoded, input_lengths)
        # Packed T x B x E
        output, _ = self.rnn(packed)
        # Packed T x B x H
        padded, _ = torch.nn.utils.rnn.pad_packed_sequence(output, padding_value=float('-inf'))
        # T x B x H
        output, _ = padded.max(dim=0)
        # Dropout
        output = self.dropout(output)
        # B x H
        output = self.h2o(output.view(-1, self.num_directions * self.hidden_size))
        # B x O
        return output


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


def model_training(train_data, val_data):
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
    model, optimizer = get_model()
    print(f'Training cross-validation model for {epochs} epochs')
    t0 = time.time()
    # Model training
    for epoch in range(1, epochs + 1):
        acc = train(model, optimizer, train_data, batch_size, token_size, log=epoch == 1)[0]
        train_accuracy.append(acc)
        # Log training accuracy
        mlflow.log_metric("train_accuracy", acc)
        print(f'| epoch {epoch:03d} | train accuracy={acc:.1f}% ({time.time() - t0:.0f}s)')
        acc = validate(model, val_data, batch_size, token_size)[1]
        valid_accuracy.append(acc)
        # Log validation accuracy
        mlflow.log_metric("val_accuracy", acc)
        print(f'| epoch {epoch:03d} | valid accuracy={acc:.1f}%')
    return model


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


def get_model():
    """
        Get the model and optimizer for training.

        Returns:
            tuple: A tuple containing the model and optimizer.
    """
    model = CharRNNClassifier(ntokens, embedding_size, hidden_size, nlabels, bidirectional=bidirectional,
                              pad_idx=pad_index)
    optimizer = torch.optim.Adam(model.parameters())
    return model, optimizer

def prediction(model, test_data, y_test) -> tuple:
    """
    Make predictions using a trained model and calculate accuracy.

    Args:
        model (torch.nn.Module): The trained model.
        test_data (list): List of test data.
        y_test (numpy.ndarray): True labels for the test data.

    Returns:
        tuple: A tuple containing predicted labels and accuracy.
    """
    y_pred = validate(model, test_data, batch_size, token_size)[0]
    accuracy = accuracy_score(y_test, y_pred)
    return y_pred, accuracy

def classification_task(model, x_train_scaled, y_train, x_test_scaled, y_test, predic, model_name) -> pd.DataFrame:
    """
    Perform a classification task and return performance metrics as a DataFrame.

    Args:
        model (scikit-learn classifier): The trained classification model.
        x_train_scaled (numpy.ndarray): Scaled training features.
        y_train (numpy.ndarray): True labels for the training data.
        x_test_scaled (numpy.ndarray): Scaled test features.
        y_test (numpy.ndarray): True labels for the test data.
        predic (numpy.ndarray): Predicted labels for the test data.
        model_name (str): Name of the model for indexing in the DataFrame.

    Returns:
        DataFrame: A DataFrame containing performance metrics for the classification task.
    """
    perf_df = pd.DataFrame(
        {'Train_Score': model.score(x_train_scaled, y_train), "Test_Score": model.score(x_test_scaled, y_test),
         "Precision_Score": precision_score(y_test, predic), "Recall_Score": recall_score(y_test, predic),
         "F1_Score": f1_score(y_test, predic), "accuracy": accuracy_score(y_test, predic)}, index=[model_name])
    return perf_df

def eval_metrics(actual, pred):
    """
    Calculate evaluation metrics for regression tasks.

    Args:
        actual (numpy.ndarray): True target values.
        pred (numpy.ndarray): Predicted target values.

    Returns:
        tuple: A tuple containing RMSE (Root Mean Squared Error), MAE (Mean Absolute Error), and R-squared (R2) scores.
    """
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2

# MAIN
if __name__ == '__main__':
    # Init random seed to get reproducible results
    set_seed()

    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    # Initialize MLflow
    mlflow.set_tracking_uri("https://github.com/MLOps-essi-upc/clothing-reviews/tree/develop/models")
    mlflow.set_experiment("experiment_dl")

    # Read the preprocessed data
    path_data = "./data/raw/raw_data.csv"
    df = read_data(path_data)
    # Set this flag based on whether stemming is applied or not
    use_stemming = True
    x, y = stemming(df, use_stemming)

    word_vocab, pad_index, unk_index = create_word_vocab(x)
    label_vocab = create_label_vocab(y)

    # Convert the data to indices
    x_idx, y_idx = convert_data_to_indices(x, y, word_vocab, label_vocab)

    # Split the data into training (70%), validation (15%), and test (15%) sets
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = split_data(x_idx, y_idx)

    train_data = [(x, y) for x, y in zip(x_train, y_train)]
    val_data = [(x, y) for x, y in zip(x_val, y_val)]
    test_data = [(x, y) for x, y in zip(x_test, y_test)]

    criterion = torch.nn.CrossEntropyLoss(reduction='sum')

    ntokens = len(word_vocab)
    nlabels = len(label_vocab)

    with mlflow.start_run():
        # Log your hyperparameters
        mlflow.log_param("hidden_size", hidden_size)
        mlflow.log_param("embedding_size", embedding_size)
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("token_size", token_size)
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("bidirectional", bidirectional)

        model = model_training(train_data, val_data)

        y_pred, acc = prediction(model, test_data, y_test)

        # Log test accuracy
        mlflow.log_metric("test_accuracy", acc)

        (rmse, mae, r2) = eval_metrics(y_test, y_pred)

        print("SVC model (hidden_size=%f):" % (hidden_size))
        print("SVC model (embedding_size=%f):" % (embedding_size))
        print("  RMSE: %s" % rmse)
        print("  MAE: %s" % mae)
        print("  R2: %s" % r2)

        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("mae", mae)

        # Save the trained model with mlflow
        mlflow.pytorch.save_model(model, "models")