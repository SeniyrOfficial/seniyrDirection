from sklearn import svm
import pandas as pd
import numpy as np

df = pd.read_csv('data/trainData/trainData.csv')

data_numpy = df.to_numpy()
X = data_numpy[:, :34]
Y = data_numpy[:, 34]

clf = svm.SVC(decision_function_shape='ovo')
clf.fit(X, Y)

test_1 = np.array([[1,1,1,1,1,0,0,1,0,0,1,0,0,3,0,0,0,1,0,1,0,0,0,0,1,1,0,4,1,91,97,74,87,85]]) # 0
test_2 = np.array([[0,0,0,0,0,0,0,1,1,0,0,1,0,3,1,0,1,1,1,0,1,1,1,0,1,1,0,4,3,81,90,84,67,82]]) # 1
test_3 = np.array([[1,0,1,0,1,0,0,0,0,0,0,1,0,3,0,0,0,0,1,0,0,1,1,0,0,0,1,6,0,94,96,83,99,98]])
test_4 = np.array([[1,0,0,0,0,0,0,1,1,1,0,1,1,4,1,0,1,1,0,0,1,0,0,1,0,1,0,5,1,71,97,88,88,91]])


print("Test_1 Label: ", clf.predict(test_1))
print("Test_2 Label: ", clf.predict(test_2))
print("Test_3 Label: ", clf.predict(test_3))
print("Test_4 Label: ", clf.predict(test_4))

correct = 0
wrong = 0
for i, data_sample in enumerate(X[:]):
    if clf.predict(np.array([data_sample])) == Y[i]:
        correct += 1
    else:
        wrong += 1
total_data_samples = i
print("Training Accuracy: ", correct/total_data_samples)


df_test = pd.read_csv('data/testData/testData.csv')

data_numpy = df_test.to_numpy()
X_test = data_numpy[:, :34]
Y_test = data_numpy[:, 34]

correct = 0
wrong = 0
for i, data_sample in enumerate(X_test[:]):
    if clf.predict(np.array([data_sample])) == Y_test[i]:
        correct += 1
    else:
        wrong += 1
total_test_samples = i
print("Test Accuracy: ", correct/total_test_samples)
print("Total training/test samples: ", total_data_samples, total_test_samples)