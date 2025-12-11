import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

#all this will be part of the results table
class ModelResult:
    def __init__(self, name, accuracy, precision, recall, f1, confusion, model):
        self.name = name
        self.accuracy = accuracy
        self.precision = precision
        self.recall = recall
        self.f1 = f1
        self.confusion = confusion
        self.model = model


def train_test_split_tfidf(texts, targets, test_size=0.2):
    """
    Split data by training data and test data. Compute TF-IDF
    variables: texts and targets are both a list
    """
    try:
        random_state = int(input("Enter a random positive seed value >=1 (or 0 for random seed): ") or 0)
        # none will just make it a random shuffle of data each time
    except ValueError:
        random_state = 0

    print("You entered: ", random_state)

    # random state is the seed value, unless it's 0. then seed value is none (random seed)
    random_seed_value = random_state if random_state != 0  else None

    X_train_text, X_test_text, y_train, y_test = train_test_split(
        texts, targets, test_size=test_size, stratify=targets, random_state=random_seed_value
    )

    # https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html
    vectorizer = TfidfVectorizer(
        stop_words="english",
        #encoding="utf-8", #latin?
        #strip_accents="unicode",
        #use_idf=True,
        #min_df=1,
        #max_features=1000,
        #binary=False,
        #use_idf=True,
        max_features=3000,
        ngram_range=(1, 2),
    )
    X_train = vectorizer.fit_transform(X_train_text)
    X_test = vectorizer.transform(X_test_text)

    return X_train, X_test, y_train, y_test, vectorizer


def evaluate_model(name, model, X_test, y_test):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    conf = confusion_matrix(y_test, y_pred)
    """
    self.name = name
    self.accuracy = accuracy
    self.precision = precision
    self.recall = recall
    self.f1 = f1
    self.confusion = confusion
    self.model = model
    """
    return ModelResult(
        name=name,
        accuracy=acc,
        precision=prec,
        recall=rec,
        f1=f1,
        confusion=conf,
        model=model,
    )


def train_all_models(X_train, y_train, X_test, y_test):
    """
    Train some three simple models often used in spam research.
    """
    results= {}

    # Multinomial Naive Bayes
    #https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.MultinomialNB.html
    nb = MultinomialNB()
    nb.fit(X_train, y_train)
    results["naive_bayes"] = evaluate_model("Naive Bayes", nb, X_test, y_test)

    # Logistic Regression
    #https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_train, y_train)
    results["log_reg"] = evaluate_model("Logistic Regression", lr, X_test, y_test)

    # KNN
    #https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
    knn = KNeighborsClassifier(n_neighbors=5) #user input? predefined? deciding predefined for sake of convenience 12/9
    knn.fit(X_train, y_train)
    results["knn"] = evaluate_model("KNN(k=5)", knn, X_test, y_test)

    return results


def results_to_dataframe(results, X_test, y_test):
    rows = []
    y_test_array = np.array(y_test)
    num_spam_actual = int((y_test_array == 1).sum())
    num_ham_actual = int((y_test_array == 0).sum())
    for key, res in results.items():
        all_predictions = res.model.predict(X_test)
        #num_spam = int((all_predictions == 1)) #use pandas/
        #num_ham = int((all_predictions == 0))
        num_spam = int((all_predictions == 1).sum())
        num_ham = int((all_predictions == 0).sum())

        rows.append(
            {
                "model": res.name,
                "accuracy": res.accuracy,
                "precision": res.precision,
                "recall": res.recall,
                "f1": res.f1,
                "predicted_spam": num_spam,
                "predicted_ham": num_ham,
                "actual_spam": num_spam_actual,
                "actual_ham": num_ham_actual,
            }
        )
    return pd.DataFrame(rows)
