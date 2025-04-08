import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split

# 加载原始数据
X = np.load("X.npy")
y = np.load("y.npy", allow_pickle=True)
print(f"y.shape: {y.shape}, y.dtype: {y.dtype}")


# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 训练 One-class SVM
clf = svm.OneClassSVM(nu=0.1, kernel="rbf", gamma=0.01)
clf.fit(X_train)  # 使用高维数据直接训练

import joblib
joblib.dump(clf, "one_class_svm_model.pkl")
print("模型训练完成！")
