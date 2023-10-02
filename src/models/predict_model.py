from train_model import  *
from sklearn.metrics import accuracy_score

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

if __name__ == '__main__':

    pass