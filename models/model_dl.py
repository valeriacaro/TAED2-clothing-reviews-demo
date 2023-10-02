# IMPORTS
import argparse
import mlflow.pytorch
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
import random
import pandas as pd
import torch  # Deep learning framework
import time
from sklearn.model_selection import train_test_split
import nltk
from nltk.corpus import stopwords
import sys
from nltk.stem import SnowballStemmer
from nltk.tokenize import word_tokenize
import string
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


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


def read_data() -> pd.DataFrame:
    """ Load the preprocessed data"""
    pass


class Dictionary(object):
    def __init__(self):
        self.token2idx = {}
        self.idx2token = []

    def add_token(self, token):
        if token not in self.token2idx:
            self.idx2token.append(token)
            self.token2idx[token] = len(self.idx2token) - 1
        return self.token2idx[token]

    def __len__(self):
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
    """Create a vocabulary for word tokens."""
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
    """Create a vocabulary for labels."""
    label_vocab = Dictionary()
    # Create a set of labels from the training labels (0 or 1)
    labels = set(y)
    for label in sorted(labels):
        label_vocab.add_token(label)
    return label_vocab


def convert_data_to_indices(x, y, vocab, label_vocab):
    """Convert data to token indices."""
    x_idx = [np.array([vocab.token2idx[word] for word in line.split()]) for line in x]
    y_idx = np.array([label_vocab.token2idx[label] for label in y])
    return x_idx, y_idx


def split_data(x, y):
    """Split data into training, validation, and test sets."""
    x_train, x_temp, y_train, y_temp = train_test_split(x, y, test_size=0.3, random_state=seed)
    x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, test_size=0.5, random_state=seed)
    return (x_train, y_train), (x_val, y_val), (x_test, y_test)


def batch_generator(data, batch_size, token_size):
    """Yield elements from data in chunks with a maximum of batch_size sequences and token_size tokens."""
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
    """Sort within buckets, then batch, then shuffle batches.
    Partitions data into chunks of size 100*token_size, sorts examples within
    each chunk, then batch these examples and shuffle the batches.
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
        super().__init__()
        self.model = model.lower()
        self.hidden_size = hidden_size
        self.embed = torch.nn.Embedding(input_size, embedding_size, padding_idx=pad_idx)
        if self.model == "gru":
            self.rnn = torch.nn.GRU(embedding_size, hidden_size, num_layers, bidirectional=bidirectional)
        elif self.model == "lstm":
            self.rnn = torch.nn.LSTM(embedding_size, 2*hidden_size, num_layers, bidirectional=bidirectional)
        self.h2o = torch.nn.Linear(2*hidden_size, output_size)
        self.dropout = torch.nn.Dropout(0.1, inplace=True)


    def forward(self, input, input_lengths):
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
        output = self.h2o(output)  # no hay prob actualizada
        # B x O
        return output


def train(model, optimizer, data, batch_size, token_size, max_norm=1, log=False):
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
            # Agregar las predicciones a la lista
            predictions.extend(torch.max(answer, 1)[1].tolist())

        dev_acc = 100 * ncorrect / nsentences
    return predictions, dev_acc


def get_model():
    model = CharRNNClassifier(ntokens, embedding_size, hidden_size, nlabels, bidirectional=bidirectional,
                              pad_idx=pad_index)
    optimizer = torch.optim.Adam(model.parameters())
    return model, optimizer

def prediction(model, test_data, y_test):
    y_pred = validate(model, test_data, batch_size, token_size)[0]
    accuracy = accuracy_score(y_test, y_pred)
    return y_pred, accuracy

def classification_task(model, x_train_scaled, y_train, x_test_scaled, y_test, predic, model_name):
    perf_df = pd.DataFrame(
        {'Train_Score': model.score(x_train_scaled, y_train), "Test_Score": model.score(x_test_scaled, y_test),
         "Precision_Score": precision_score(y_test, predic), "Recall_Score": recall_score(y_test, predic),
         "F1_Score": f1_score(y_test, predic), "accuracy": accuracy_score(y_test, predic)}, index=[model_name])
    return perf_df

def eval_metrics(actual, pred):
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
    #mlflow.set_tracking_uri("https://github.com/MLOps-essi-upc/clothing-reviews/tree/develop/models")
    mlflow.set_experiment("experiment_dl")

    # Load and preprocess the data
    #df = read_data()
    df = pd.read_csv("/Users/esther/Desktop/GCED/Q7/TAED2/LAB/Womens Clothing E-Commerce Reviews.csv")
    df.drop(["Unnamed: 0"], axis=1, inplace = True)
    df['Title'].isnull().count()
    df.drop(['Title'], axis=1, inplace = True)
    df.drop(['Positive Feedback Count'], axis = 1, inplace = True)
    df = df.drop_duplicates()
    df['Division Name'].value_counts()
    df['Department Name'].value_counts()
    df['Class Name'].value_counts()
    df.loc[df['Rating'] <= 4, 'Recommended IND'] = 0
    df['Recommended IND'].value_counts()
    df.rename(columns={'Recommended IND': 'Top Product'}, inplace=True)
    df.drop(['Rating'], axis = 1, inplace=True)
    df.drop(['Division Name'], axis = 1, inplace=True)
    df.drop(['Department Name'], axis = 1, inplace=True)
    df.drop(['Class Name'], axis = 1, inplace=True)
    df.drop(['Age'], axis = 1, inplace=True)
    df.drop(['Clothing ID'], axis = 1, inplace=True)

    nltk.download("punkt")
    nltk.download("stopwords")

    # List of stopwords
    english_sw = set(stopwords.words('english') + list(string.punctuation))

    df.astype(str)

    # Erasing
    for index in df.index:
        text=word_tokenize(str(df.at[index,'Review Text']))
        text = [w.lower() for w in text if w.lower() not in english_sw]
        df.at[index,'Review Text']=text

    for index in df.index:
        text = [w for w in df.at[index,'Review Text'] if w not in "'s"]
        df.at[index,'Review Text']=text


    stem = SnowballStemmer('english')

    stemmed_text=[]
    for index in df.index:
        stemmed_text.append([stem.stem(w) for w in df.at[index,'Review Text']])

    df['Stemmed Review Text']=stemmed_text

    df['Stemmed Review Text'] = df['Stemmed Review Text'].apply(
    lambda x: " ".join([str(i) for i in x])
    )

    df['Review Text'] = df['Review Text'].apply(
    lambda x: " ".join([str(i) for i in x])
    )

    df.head()

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

    # Close the MLflow run
    mlflow.end_run()
