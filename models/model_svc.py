# IMPORTS
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import \
    TfidfVectorizer  # TF-IDF are word frequency scores that try to highlight words that are more interesting,
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score


# FUNCTIONS

def classification_task(model, x_train_scaled, y_train, x_test_scaled, y_test, predic, model_name):
    perf_df = pd.DataFrame(
        {'Train_Score': model.score(x_train_scaled, y_train), "Test_Score": model.score(x_test_scaled, y_test),
         "Precision_Score": precision_score(y_test, predic), "Recall_Score": recall_score(y_test, predic),
         "F1_Score": f1_score(y_test, predic), "accuracy": accuracy_score(y_test, predic)}, index=[model_name])
    return perf_df


def vectorization(x_train, x_test):
    tf_idf_vectorizer = TfidfVectorizer()
    x_train_tf_idf = tf_idf_vectorizer.fit_transform(x_train)
    x_test_tf_idf = tf_idf_vectorizer.transform(x_test)
    x_train_tf_idf.toarray()
    pd.DataFrame(x_train_tf_idf.toarray(), columns=tf_idf_vectorizer.get_feature_names_out())
    return x_train_tf_idf, x_test_tf_idf


def main():
    # aquí s'hauria de llegir les dades
    df = read_data()

    stem = True
    if stem:
        x = df["Stemmed Review Text"]
    else:
        x = df["Review Text"]
    y = df["Top Product"]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, stratify=y, random_state=101)

    x_train_tf_idf, x_test_tf_idf = vectorization(x_train, x_test)

    svc = SVC(random_state=0, C=0.2, kernel='rbf')
    # fitting model
    svc.fit(x_train_tf_idf, y_train)

    # Saving the model as an artifact.
    if stem:
        filename = "models/model_svc_stem"
    else:
        filename = "models/model_svc"

    # save model
    joblib.dump(svc, filename)

    # load model
    loaded_model = joblib.load(filename)

    # predict
    pred_svc = loaded_model.predict(x_test_tf_idf)

    eval_svc = classification_task(svc, x_train_tf_idf, y_train, x_test_tf_idf, y_test, pred_svc, "SVC")
    print(eval_svc)


if __name__ == '__main__':
    main()
