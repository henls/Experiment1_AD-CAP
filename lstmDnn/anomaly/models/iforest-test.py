#_*_coding:utf-8_*_
import matplotlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
import pandas as pd
from sklearn.metrics import confusion_matrix
import seaborn as sns

sns.set()
rng = np.random.RandomState(42)

data = pd.read_excel(r'/home/wxh/lstmDnn/data/proceed/2022-3-19@10:37:58/10:38:08-normal.xlsx')
data = np.array(data)

#split data into train and test
X_train = data[47:447, :]
X_test_outliers = data[:47, :] 
X_test_normal = data[447:, :]

# fit the model

wxh = IsolationForest(behaviour='new', max_samples=1000,
                    random_state=rng, contamination='auto'
                    )
wxh.fit(X_train)
y_pred_normal = wxh.predict(X_test_normal)
y_pred_outliers = wxh.predict(X_test_outliers)
print('y_pred_normal', y_pred_normal)
print('y_pred_outliers', y_pred_outliers)
normal = confusion_matrix(np.ones(len(y_pred_normal)), y_pred_normal)
outlier = confusion_matrix(-1*np.ones(len(y_pred_outliers)), y_pred_outliers, labels=[0,1])
f, ax = plt.subplots()
figure = sns.heatmap(outlier, annot = True, ax = ax)
ax.set_title('outliers data')
ax.set_xlabel('predict')
ax.set_ylabel('true')
a = figure.get_figure()
a.savefig('outliers.png', bbox_inches = 'tight')

# Generate  train data
X = 0.3 * rng.randn(100, 2)
X_train = np.r_[X + 2, X - 2]
X = 0.3 * rng.randn(20, 2)
X_test = np.r_[X + 2, X - 2]
X_outliers = rng.uniform(low=-4, high=4, size=(20, 2))

# fit the model
clf = IsolationForest(behaviour='new', max_samples=100,
                      random_state=rng, contamination='auto')
clf.fit(X_train)
y_pred_train = clf.predict(X_train)
y_pred_test = clf.predict(X_outliers)
print(y_pred_test)
xx, yy = np.meshgrid(np.linspace(-5, 5, 50), np.linspace(-5, 5, 50))
Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
 
plt.title("IsolationForest")
plt.contourf(xx, yy, Z, camp=plt.cm.Blues_r)
b1 = plt.scatter(X_train[:, 0], X_train[:, 1], c='white',
                 s=20, edgecolor='k')
b2 = plt.scatter(X_test[:, 0], X_test[:, 1], c='green',
                 s=20, edgecolor='k')
c = plt.scatter(X_outliers[:, 0], X_outliers[:, 1], c='red',
                s=20, edgecolor='k')
plt.axis('tight')
plt.xlim((-5, 5))
plt.ylim((-5, 5))
plt.legend([b1, b2, c],
           ["training observations",
            "new regular observations", "new abnormal observations"],
           loc="upper left")
plt.savefig('table.png')
