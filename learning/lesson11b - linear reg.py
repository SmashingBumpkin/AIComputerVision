import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression


# Reproducibly random number
rng = np.random.RandomState(0)
x = 10 * rng.rand(50)
y = 2 * x - 1 + 2 * rng.rand(50)

# plt.scatter(x, y)
# plt.show()

model = LinearRegression(fit_intercept=True)

# Here we do a reshape, the same as reshape but diff syntax
X = x[:, np.newaxis]

model.fit(X, y)

xfit = np.linspace(-1, 11, 50)

Xfit = xfit[:, np.newaxis]
y_pred = model.predict(Xfit)

plt.scatter(x, y)
plt.plot(Xfit, y_pred)
plt.show()
