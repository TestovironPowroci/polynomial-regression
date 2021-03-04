from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
import numpy as np
from matplotlib import pyplot as plt

msetest = []
mseval = []

learning = np.loadtxt('traindata.csv', delimiter=',', dtype=np.float32)
input_learning = (learning[:, 0:-1])
target_learning = (learning[:, [-1]])

testing = np.loadtxt('testdata.csv', delimiter=',', dtype=np.float32)
input_testing = (testing[:, 0:-1])
target_testing = (testing[:, [-1]])

validation = np.loadtxt('validationdata.csv', delimiter=',', dtype=np.float32)
input_validation = (validation[:, 0:-1])
target_validation = (validation[:, [-1]])



poly = PolynomialFeatures(degree=3)
input_learning_ = poly.fit_transform(input_learning)

testing_predict = poly.fit_transform(input_testing)
validation_predict = poly.fit_transform(input_validation)
input_learning_ = poly.fit_transform(input_learning)
clf = linear_model.LinearRegression()
clf.fit(input_learning_, target_learning)


print("________________uczące_________________")
mselearning = target_learning - clf.predict(input_learning_)
mselearning = mselearning**2
print((mselearning.sum())/len(input_learning))





print("____________test____________________")

print(clf.predict(testing_predict))

plt.title("Wykres dokładności predykcji modelu wielomianowego na zbiorze testowym")
plt.xlabel("iteracja")
plt.ylabel("średnica wejściowa[μm]")
plt.plot(clf.predict(testing_predict)*10000,"ob", label='predykcja', linestyle='-')
plt.xticks(np.arange(len(clf.predict(testing_predict))), np.arange(1, len(clf.predict(testing_predict))+1))
plt.plot(target_testing*10000, "go", label='wartość rzeczywista', linestyle='-')
plt.xticks(np.arange(len(target_testing)), np.arange(1, len(target_testing)+1))

leg = plt.legend()
plt.show()

print("MSE testu")
msetest = target_testing - clf.predict(testing_predict)
msetest = msetest**2

print((msetest.sum())/len(input_testing))



print("____________walidacja_______________")
print(clf.predict(validation_predict))

plt.title("Wykres dokładności predykcji modelu wielomianowego na zbiorze walidacyjnym")
plt.xlabel("iteracja")
plt.ylabel("średnica wejściowa[μm]")
plt.plot(clf.predict(validation_predict)*10000,"ob", label='predykcja', linestyle='-')
plt.xticks(np.arange(len(clf.predict(validation_predict))), np.arange(1, len(clf.predict(validation_predict))+1))
plt.plot(target_validation*10000, "go", label='wartość rzeczywista', linestyle='-')
plt.xticks(np.arange(len(target_validation)), np.arange(1, len(target_validation)+1))
leg = plt.legend()
plt.show()

print("MSE walidacji")
mseval = target_validation - clf.predict(validation_predict)
mseval = mseval**2

print(mseval.sum()/len(input_validation))