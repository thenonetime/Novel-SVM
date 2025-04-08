import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np
import joblib
from tqdm import tqdm
import seaborn as sns
plt.rcParams['font.sans-serif']=['Simhei'] #解决中文显示问题，目前只知道黑体可行
plt.rcParams['axes.unicode_minus']=False #解决负数坐标显示问题
# **加载数据**
X_pca = np.load("X_pca.npy")  # 降维后的数据
X = np.load("X.npy")  # 原始高维数据
y = np.load("y.npy", allow_pickle=True)  # 重新加载类别数据
y = np.array(y, dtype=str)  # 确保 y 仍然是一维字符串数组
clf = joblib.load("one_class_svm_model.pkl")  # 加载训练好的 SVM 模型

print("加载的类别:", np.unique(y))  # 打印所有类别名称，检查数据类别

# **降维用于可视化**
pca = PCA(n_components=2)
X_pca_train = pca.fit_transform(X)

# **生成 500 个随机陌生样本**
num_novel_samples = 500
X_outliers = np.random.uniform(-1, 1, (num_novel_samples, X.shape[1]))
X_outliers_pca = pca.transform(X_outliers)

# **生成网格**
xx, yy = np.meshgrid(np.linspace(-5, 5, 200), np.linspace(-5, 5, 200))
print("Step 1: 创建网格.")

# **计算决策边界**
Z = np.zeros(xx.shape).ravel()
for i, (x, yy_grid) in enumerate(tqdm(np.c_[xx.ravel(), yy.ravel()], desc="计算决策边界")):
    Z[i] = clf.decision_function(pca.inverse_transform([[x, yy_grid]]))[0]
Z = Z.reshape(xx.shape)
print("Step 2: 计算决策边界完成.")

# **绘制决策区域**
plt.contourf(xx, yy, Z, levels=np.linspace(Z.min(), 0, 7), cmap=plt.cm.PuBu, alpha=0.6)
plt.contourf(xx, yy, Z, levels=[0, Z.max()], colors='palevioletred', alpha=0.3)
print("Step 3: 绘制决策区域.")

# **绘制决策边界**
if Z.min() < 0 and Z.max() > 0:
    plt.contour(xx, yy, Z, levels=[0], linewidths=2, colors='darkred')
    plt.scatter([], [], c='darkred', marker='_', label="决策边界")  # **在图例中添加决策边界**
    print("Step 4: 绘制决策边界.")

# **采样训练数据**
X_pca_train_sample = X_pca_train
y_sample = y  # **确保 y 仍然是一维数组**
print("Step 5: 训练数据采样完成.")

# **自动分配类别颜色**
unique_labels = np.unique(y_sample)
palette = sns.color_palette("husl", len(unique_labels))  # 生成颜色
label_color_map = {label: palette[i] for i, label in enumerate(unique_labels)}

# **绘制训练数据**
s = 40
for label in unique_labels:
    indices = y_sample == label
    plt.scatter(X_pca_train_sample[indices, 0], X_pca_train_sample[indices, 1],
                c=[label_color_map[label]], s=s, edgecolors='k', label=f"{label}")

# **绘制陌生样本**
plt.scatter(X_outliers_pca[:, 0], X_outliers_pca[:, 1], c='gold', s=s, edgecolors='k', label="未知类别")
print("Step 6: 绘制数据点.")
'''
# **标注最小决策边界点**
min_Z_index = np.unravel_index(np.argmin(Z), Z.shape)
min_point = np.array([xx[min_Z_index], yy[min_Z_index]])
plt.scatter(min_point[0], min_point[1], c='red', s=100, edgecolors='black', marker='o', label="最小决策点")
'''
# **优化图例**
plt.axis('tight')
plt.xlim((-5, 5))
plt.ylim((-5, 5))
plt.legend(loc="upper left", fontsize=10, frameon=True)
plt.xlabel("降维维度 1")
plt.ylabel("降维维度 2")

# **保存图像**
output_path = "novelty_detection_plot.png"
plt.savefig(output_path, dpi=300, bbox_inches="tight")
print(f"Step 7: 图像已保存到 {output_path}.")

# **显示图**
plt.show()
print("Step 8: 图像已显示.")
