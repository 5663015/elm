# Final edited date: 2018.3.7
# Author: Li Xudong, from NSSC.CAS Beijing
# Version: 1.0
# Description: Test the model with some simple dataset to solve classification problems and
#              regression problem. The datasets of classification problems are irises dataset
#              and handwritten number dataset. The dataset of regression problem is randomly
#              generated.
import elm
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.datasets import load_iris, load_digits, load_diabetes, make_regression
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

stdsc = StandardScaler()
# **********************************
# irises dataset classification
# **********************************
print("irises dataset classification>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
# load dataset
iris = load_iris()
irx, iry = stdsc.fit_transform(iris.data), iris.target
print("irx shape:", irx.shape)
print("iry shape:", iry.shape)
x_train, x_test, y_train, y_test = train_test_split(irx, iry, test_size=0.2)

# built model and train
model = elm.elm(hidden_units=32, activation_function='relu', random_type='normal', x=x_train, y=y_train, C=0.1, elm_type='clf')
beta, train_accuracy, running_time = model.fit('faster2')
print("classifier beta:\n", beta)
print("classifier train accuracy:", train_accuracy)
print('classifier running time:', running_time)

# test
prediction = model.predict(x_test)
print("classifier test prediction:", prediction)
print('classifier test accuracy:', model.score(x_test, y_test))



# *******************************
# handwritten number dataset
# *******************************
print("handwritten number dataset classification>>>>>>>>>>>>>>>>>>>>>>>>")
# load dataset
digits = load_digits()
dgx, dgy = stdsc.fit_transform(digits.data/16.0), digits.target
print("dgx shape:", dgx.shape)
print("dgy shape:", dgy.shape)
x_train, x_test, y_train, y_test = train_test_split(irx, iry, test_size=0.2)

# build model and train
model = elm.elm(hidden_units=32, activation_function='relu', random_type='normal', x=x_train, y=y_train, C=0.1, elm_type='clf')
beta, train_accuracy, running_time = model.fit('faster2')
print("classifier beta:\n", beta)
print("classifier train accuracy:", train_accuracy)
print('classifier running time:', running_time)

# test
prediction = model.predict(x_test)
print("classifier test prediction:", prediction)
print('classifier test accuracy:', model.score(x_test, y_test))



# **************************
# regression problem
# **************************
print("regression problem>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
# generate dataset
x = np.arange(0.25, 20, 0.1).reshape(-1, 1)
y = x * np.cos(x) + 0.5 * np.sqrt(x) * np.random.randn(x.shape[0]).reshape(-1, 1)
xtoy, ytoy = stdsc.fit_transform(x), stdsc.fit_transform(y)

# build model and train
model = elm.elm(hidden_units=32, activation_function='sigmoid', random_type='normal', x=xtoy, y=ytoy, C=1, elm_type='reg')  # normal分布效果好
beta, train_score, running_time = model.fit('no_re')
print("regression beta:\n", beta)
print("regression train score:", train_score)
print("regression running time:", running_time)

# test
prediction = model.predict(xtoy)
print("regression result:", prediction.reshape(-1, ))
print("regression score:", model.score(xtoy, ytoy))

# plot
plt.plot(xtoy, ytoy)
plt.plot(xtoy, prediction)
plt.title('xtoy, ytoy')
plt.legend()
plt.show()