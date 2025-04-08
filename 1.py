import os
import numpy as np
import librosa
from sklearn import svm
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# 音频数据加载函数
def load_audio_data(folder_path):
    data = []
    labels = []
    for label_folder in os.listdir(folder_path):  # 每个子文件夹是一个类别
        label_path = os.path.join(folder_path, label_folder)
        if os.path.isdir(label_path):
            for file_name in os.listdir(label_path):
                file_path = os.path.join(label_path, file_name)
                if file_path.endswith('.wav'):
                    audio_data, sr = librosa.load(file_path, sr=None)
                    data.append(audio_data[:24000])  # 保证每个样本为1*24000
                    labels.append(label_folder)
    return np.array(data), np.array(labels)

# 数据读取
data_folder = r"F:\PycharmProjects\gihub\2025年\1.29\新奇svm\电抗器声纹样本（新）"  # 替换为实际文件夹路径
X, y = load_audio_data(data_folder)

# 数据划分
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 训练 One-class SVM
clf = svm.OneClassSVM(nu=0.1, kernel="rbf", gamma=0.01)
clf.fit(X_train)  # 使用高维数据直接训练

# 降维用于可视化
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

# 生成随机异常样本并降维
X_outliers = np.random.uniform(-1, 1, (50, X_train.shape[1]))
X_outliers_pca = pca.transform(X_outliers)

# 绘制可视化
xx, yy = np.meshgrid(np.linspace(-5, 5, 500), np.linspace(-5, 5, 500))
Z = clf.decision_function(pca.inverse_transform(np.c_[xx.ravel(), yy.ravel()]))
Z = Z.reshape(xx.shape)

plt.title("Novelty Detection")
plt.contourf(xx, yy, Z, levels=np.linspace(Z.min(), 0, 7), cmap=plt.cm.PuBu)
a = plt.contour(xx, yy, Z, levels=[0], linewidths=2, colors='darkred')
plt.contourf(xx, yy, Z, levels=[0, Z.max()], colors='palevioletred')

# 绘制训练集、测试集和异常样本点
s = 40
b1 = plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1], c='white', s=s, edgecolors='k', label="Training data")
b2 = plt.scatter(X_test_pca[:, 0], X_test_pca[:, 1], c='blueviolet', s=s, edgecolors='k', label="Test regular data")
c = plt.scatter(X_outliers_pca[:, 0], X_outliers_pca[:, 1], c='gold', s=s, edgecolors='k', label="Test abnormal data")

plt.axis('tight')
plt.xlim((-5, 5))
plt.ylim((-5, 5))
plt.legend(loc="upper left")
plt.xlabel("Reduced Dimension 1")
plt.ylabel("Reduced Dimension 2")
plt.show()
