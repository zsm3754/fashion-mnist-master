#随机森林训练OOD
import numpy as np
import gzip
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# 加载Fashion-MNIST
def load_mnist(path, kind='train'):
    labels_path = os.path.join(path, f'{kind}-labels-idx1-ubyte.gz')
    images_path = os.path.join(path, f'{kind}-images-idx3-ubyte.gz')
    with gzip.open(labels_path, 'rb') as f:
        labels = np.frombuffer(f.read(), dtype=np.uint8, offset=8)
    with gzip.open(images_path, 'rb') as f:
        images = np.frombuffer(f.read(), dtype=np.uint8, offset=16).reshape(len(labels), 784)
    return images, labels

# 数据路径
train_x, train_y = load_mnist('./data/fashion', 'train')
test_x, test_y = load_mnist('./data/fashion', 't10k')

# 训练随机森林
print("训练随机森林...")
rf = RandomForestClassifier(n_estimators=100, n_jobs=-1)
rf.fit(train_x, train_y)

# 分布内测试
pred_in = rf.predict(test_x)
acc_in = accuracy_score(test_y, pred_in)

# ===================== OOD 测试（已修复） =====================
ood_images = np.load('./data/fashion/ood_test_images.npy')
ood_labels = np.load('./data/fashion/test_labels.npy')

# 正确的彩色转灰度
ood_gray = ood_images.reshape(-1, 3).mean(axis=1).reshape(-1, 784)
pred_ood = rf.predict(ood_gray)
acc_ood = accuracy_score(ood_labels, pred_ood)

# 输出
print("==================== 随机森林结果 ===================")
print(f"分布内准确率：{acc_in:.4f}")
print(f"OOD 准确率：{acc_ood:.4f}")