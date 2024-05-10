from sklearn.neural_network import MLPClassifier

# from sklearn.datasets import fetch_openml
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# get the dataset
# X, y = fetch_openml('mnist_784', return_X_y = True)
digits = load_digits()
X = digits.data
y = digits.target

# create train and test datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7)

# create our MLP
model = MLPClassifier(
    hidden_layer_sizes=(
        100,
        100,
        100,
    )
)

# train the model
model.fit(X_train, y_train)

# test the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(accuracy)
