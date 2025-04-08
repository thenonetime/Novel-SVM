import os
import numpy as np
import librosa
from sklearn.decomposition import PCA

# 音频数据加载函数
def load_audio_data(folder_path):
    data = []
    labels = []
    for label_folder in os.listdir(folder_path):
        label_path = os.path.join(folder_path, label_folder)
        if os.path.isdir(label_path):
            for file_name in os.listdir(label_path):
                file_path = os.path.join(label_path, file_name)
                if file_path.endswith('.wav'):
                    audio_data, sr = librosa.load(file_path, sr=None)
                    data.append(audio_data[:24000])  # 裁剪为1*24000
                    labels.append(label_folder)
    return np.array(data), np.array(labels, dtype=object)  # 明确设定为对象数组


# 数据读取
data_folder = r"F:\PycharmProjects\gihub\2025年\1.29\新奇svm\电抗器声纹样本（新）"  # 替换为你的数据路径
X, y = load_audio_data(data_folder)
print(f"y.shape: {y.shape}, y.dtype: {y.dtype}")

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
np.save("X_pca.npy", X_pca)
np.save("X.npy", X)
np.save("y.npy", np.array(y, dtype=object))  # 强制确保 y 是一维数组

print("数据加载与降维完成！")
