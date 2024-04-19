import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Load the Iris dataset
iris = sns.load_dataset("iris")
# Display the first few rows of the dataset
print(iris.head())
# sns.pairplot(iris, hue="species")
# plt.show()
# We can see some very clear correlations between the diff classes
# To train the data, we don;t want it to be too easy so we'll remove the classes


X_iris = iris.drop("species", axis=1)  # we've removed the species column
Y_iris = iris["species"]  # we're just keeping the species column
print(X_iris.shape, Y_iris.shape, iris.shape)

# create test and train sets, random_state makes results reproducible
x_train, x_test, y_train, y_test = train_test_split(
    X_iris, Y_iris, test_size=0.8, train_size=0.2, random_state=5
)


# We're going to classify using naive bayes under the assumption that the distributions are all gaussian
model = GaussianNB()
# This trains the model
model.fit(x_train, y_train)

# make predictions of test data using the model
y_pred = model.predict(x_test)

# See how accurate the labels were
accuracy = accuracy_score(y_test, y_pred)
print(accuracy)
