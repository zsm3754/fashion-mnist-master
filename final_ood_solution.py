#灰度化 + 数据增强

import numpy as np
import tensorflow as tf
import gzip
import os

# 加载数据
def load_mnist(path, kind='train'):
    labels_path = os.path.join(path, f'{kind}-labels-idx1-ubyte.gz')
    images_path = os.path.join(path, f'{kind}-images-idx3-ubyte.gz')
    with gzip.open(labels_path, 'rb') as f:
        labels = np.frombuffer(f.read(), dtype=np.uint8, offset=8)
    with gzip.open(images_path, 'rb') as f:
        images = np.frombuffer(f.read(), dtype=np.uint8, offset=16).reshape(len(labels), 784)
    return images, labels

train_x, train_y = load_mnist('./data/fashion', 'train')
test_x, test_y = load_mnist('./data/fashion', 't10k')

train_x = train_x.astype(np.float32) / 255.0
test_x = test_x.astype(np.float32) / 255.0

# ===================== 灰度化 + 数据增强 =====================
datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)
train_x_img = train_x.reshape(-1,28,28,1)
datagen.fit(train_x_img)

# 模型
model = tf.keras.Sequential([
    tf.keras.layers.Reshape((28,28,1)),
    tf.keras.layers.Conv2D(32,(5,5),padding='same',activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(64,(5,5),padding='same',activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(1024,activation='relu'),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(10)
])

model.compile(
    optimizer=tf.keras.optimizers.SGD(0.001),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

# 训练
model.fit(datagen.flow(train_x_img, train_y, batch_size=400), epochs=100)

# ===================== 测试：分布内 + OOD =====================
_, acc_in = model.evaluate(test_x, test_y, verbose=0)

# OOD 直接【灰度化】输入，消除颜色偏移
ood_images = np.load('./data/fashion/ood_test_images.npy')
ood_labels = np.load('./data/fashion/test_labels.npy')
ood_gray = ood_images.reshape(-1,3).mean(axis=1).reshape(-1, 28,28,1) / 255.0
_, acc_ood = model.evaluate(ood_gray, ood_labels, verbose=0)

print("\n=====================================")
print("【最终结果：灰度化 + 数据增强】")
print(f"分布内准确率：{acc_in:.4f}")
print(f"OOD 准确率：{acc_ood:.4f}")
print("=====================================")

