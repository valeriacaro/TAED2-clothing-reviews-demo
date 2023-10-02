from predict_model import *
from src.data.get_and_save_data import *

TRAIN_ALL_MODEL = False

if __name__ == '__main__':

    # Init random seed to get reproducible results
    set_seed()

    path_to_processed_data = "./data/processed/processed_data.csv"

    df = get_data_from_local(path_to_processed_data)

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

    ntokens = len(word_vocab)
    nlabels = len(label_vocab)

    if TRAIN_ALL_MODEL:
        model = model_training(train_data, val_data, ntokens, nlabels, pad_index)

    y_pred, acc = prediction(model, test_data, y_test)

    (rmse, mae, r2) = eval_metrics(y_test, y_pred)

    print("SVC model (hidden_size=%f):" % (hidden_size))
    print("SVC model (embedding_size=%f):" % (embedding_size))
    print("  RMSE: %s" % rmse)
    print("  MAE: %s" % mae)
    print("  R2: %s" % r2)

