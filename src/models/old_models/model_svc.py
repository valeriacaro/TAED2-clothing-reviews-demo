"""
This module is created to store all functions
related to SVC model and get the model itself.
"""

import joblib
from sklearn.svm import SVC
from src.data.get_and_save_data import get_data_from_local
from src.models.old_models.model_random_forest import (
tracking, vectorization, classification_task, stemming
)
from src import PROCESSED_TEST_DATA_PATH, PROCESSED_TRAIN_DATA_PATH

def train_and_save_svc_model(x_train, y_train, stem=True):
    """
    Trains and saves a Support Vector Machine (SVM) model.
    Args:
        x_train: Training data.
        y_train: Training labels.
        stem (bool): A flag indicating whether stemming is applied.
    Returns:
        None
    """
    svc = SVC(random_state=0, C=0.2, kernel='rbf')
    svc.fit(x_train, y_train)

    # Determine the model file name based on whether stemming is applied
    filename = "model/model_svc_stem" if stem else "model/model_svc"

    # Save the trained SVM model
    joblib.dump(svc, filename)


def prediction(model, x_test) -> list:
    """
        Makes predictions using a loaded model.
            Args:
                model: The loaded model.
                x_test: Testing dataset.
            Returns:
                predictions: Predicted values.
    """
    predictions = model.predict(x_test)
    return predictions


def loading(stem=True) -> SVC:
    """
        Loads a trained model based on the stemming flag.
        Args:
            stem (bool): A flag indicating whether stemming is applied.
        Returns:
            SVC: The loaded SVM model.
    """
    # Determine the model file name based on whether stemming is applied
    filename = "model/model_rf_stem" if stem else "model/model_rf"
    # Load the model
    model = joblib.load(filename)
    return model


if __name__ == '__main__':

    tracking()
    # Load and preprocess the data
    test = get_data_from_local(PROCESSED_TEST_DATA_PATH)
    train = get_data_from_local(PROCESSED_TRAIN_DATA_PATH)

    # Set this flag based on whether stemming is applied or not
    USE_STEMMING = True
    x_train, y_train = stemming(test, USE_STEMMING)
    x_test, y_test = stemming(test, USE_STEMMING)

    # Vectorize the text data
    x_train_tf_idf, x_test_tf_idf = vectorization(x_train, x_test)

    # Train and save the SVM model
    train_and_save_svc_model(x_train_tf_idf, y_train, USE_STEMMING)

    # Load the trained model
    loaded_model = loading(USE_STEMMING)

    # Predict
    pred_svc = prediction(loaded_model, x_test_tf_idf)

    # Evaluate and print the model's performance
    eval_svc = classification_task(
        loaded_model, x_train_tf_idf,
        y_train, x_test_tf_idf, y_test,
        pred_svc, "SVC"
    )
    print(eval_svc)
