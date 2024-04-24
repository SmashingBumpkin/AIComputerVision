############ DIGIT CLASSIFICATION WITH SCIKIT LEARN
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix
import random
import seaborn as sns

digits = load_digits()

# width_plot = 41
# total_plots = width_plot**2
# random.seed(0)
# randoms = random.sample(range(0, digits.images.shape[0]), total_plots)

# fig, axes = plt.subplots(
#     width_plot,
#     width_plot,
#     figsize=(8, 8),
#     subplot_kw={"xticks": [], "yticks": [], "frame_on": False},
#     gridspec_kw=dict(hspace=0.05, wspace=0.2),
# )

# for i, ax in enumerate(axes.flat):
#     index = randoms[i]
#     ax.imshow(digits.images[index], cmap="binary")
#     ax.text(-1, 5, str(digits.target[index]), color="green")
# # xtick are the numbers on the x axis, y for y
# plt.show()
x = digits.data
y = digits.target

Xtrain, Xtest, Ytrain, Ytest = train_test_split(x, y, random_state=5, train_size=0.75)

model = GaussianNB()

model.fit(Xtrain, Ytrain)
y_pred = model.predict(Xtest)

print(accuracy_score(Ytest, y_pred))

mat = confusion_matrix(Ytest, y_pred)

sns.heatmap(mat, square=True, annot=True, cbar=False)
plt.xlabel("pred val")
plt.ylabel("tru val")
plt.show()
