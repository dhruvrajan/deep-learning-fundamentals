from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

iris_data = load_iris()

def run_clf(clf_type):
    clf = clf_type()
    X_train, X_test, y_train, y_test = train_test_split(iris_data.data, iris_data.target, test_size=0.33)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    return accuracy_score(y_test, y_pred)

clfs = [LogisticRegression, MLPClassifier, MultinomialNB]


for clf_type in clfs:
    accuracies = [run_clf(clf_type) for _ in range(100)]
    print(str(clf_type), sum(accuracies) / len(accuracies))



