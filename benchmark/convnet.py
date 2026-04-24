import numpy as np
import tensorflow as tf
import os
import gzip
import matplotlib.pyplot as plt

# 自己实现 shuffle，不依赖 sklearn
def shuffle(a, b):
    p = np.random.permutation(len(a))
    return a[p], b[p]


# 读取数据（和你原来完全一样）
def load_mnist(path, kind='train'):
    labels_path = os.path.join(path, f'{kind}-labels-idx1-ubyte.gz')
    images_path = os.path.join(path, f'{kind}-images-idx3-ubyte.gz')

    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8, offset=8)
    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8, offset=16).reshape(len(labels), 784)
    return images, labels


# 路径
DATA_DIR = '../data/fashion'

# ====================== 模型和你原来完全一样 ======================
model = tf.keras.Sequential([
    tf.keras.layers.Reshape((28, 28, 1), input_shape=(784,)),

    # Conv1
    tf.keras.layers.Conv2D(32, (5, 5), padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),

    # Conv2
    tf.keras.layers.Conv2D(64, (5, 5), padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),

    tf.keras.layers.Flatten(),

    # Dense + Dropout
    tf.keras.layers.Dense(1024, activation='relu'),
    tf.keras.layers.Dropout(0.4),

    # Logits
    tf.keras.layers.Dense(10)
])

# 优化器、损失、学习率 完全不变
model.compile(
    optimizer=tf.keras.optimizers.SGD(learning_rate=0.001),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

# ====================== 数据加载（和你原来一样） ======================
train_data, train_labels = load_mnist(DATA_DIR, kind='train')
eval_data, eval_labels = load_mnist(DATA_DIR, kind='t10k')

train_data = train_data.astype(np.float32)
eval_data = eval_data.astype(np.float32)

train_labels = train_labels.astype(np.int32)
eval_labels = eval_labels.astype(np.int32)

train_data, train_labels = shuffle(train_data, train_labels)
eval_data, eval_labels = shuffle(eval_data, eval_labels)

# ===================== 训练循环 =====================
acc_history = []   # 用来真实存储每一轮准确率
for j in range(100):
    print(f"\n========== 第 {j + 1} 轮训练 ==========")
    model.fit(train_data, train_labels, batch_size=400, epochs=1, shuffle=True)
    print(f"\n========== 第 {j + 1} 轮评估 ==========")
    loss, acc = model.evaluate(eval_data, eval_labels)
    print(f"准确率: {acc:.4f}")
    acc_history.append(acc)

# 绘制真实 100 轮准确率曲线
plt.figure(figsize=(9,4))
plt.plot(range(1, 101), acc_history, color='#2c3e50', linewidth=1.5)
plt.grid(alpha=0.25)
plt.xlabel('Epochs')
plt.ylabel('Test Accuracy')
plt.title('CNN Model Training Accuracy (100 Epochs)')
plt.tight_layout()
plt.savefig('../cnn_100ep_real.png', dpi=300)
plt.show()
# ===================== 直接加在 convnet.py 最后 =====================

ood_images = np.load('../data/fashion/ood_test_images.npy')
ood_labels = np.load('../data/fashion/test_labels.npy')
ood_gray = ood_images.reshape(-1,3).mean(axis=1).reshape(-1,784)/255
print("\nOOD 准确率:", model.evaluate(ood_gray, ood_labels, verbose=0)[1])