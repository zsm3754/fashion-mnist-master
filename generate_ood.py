import numpy as np
import random
import os
import gzip


def load_mnist(path, kind='train'):
    labels_path = os.path.join(path, f'{kind}-labels-idx1-ubyte.gz')
    images_path = os.path.join(path, f'{kind}-images-idx3-ubyte.gz')
    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8, offset=8)
    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8, offset=16).reshape(len(labels), 784)
    return images, labels


def generate_ood_images(images):
    ood_images = []
    for img in images:
        img_2d = img.reshape(28, 28)

        # 随机背景颜色
        bg_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        # 前景颜色，和背景不同
        while True:
            fg_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            if fg_color != bg_color:
                break

        # 生成 28x28x3 彩色图像
        ood_img = np.zeros((28, 28, 3), dtype=np.uint8)
        for i in range(28):
            for j in range(28):
                if img_2d[i, j] == 0:
                    ood_img[i, j] = bg_color
                else:
                    ood_img[i, j] = fg_color
        ood_images.append(ood_img.flatten())
    return np.array(ood_images)


DATA_DIR = './data/fashion'
test_images, test_labels = load_mnist(DATA_DIR, kind='t10k')

# 生成 OOD 数据
ood_test_images = generate_ood_images(test_images)

# 保存
np.save('./data/fashion/ood_test_images.npy', ood_test_images)
np.save('./data/fashion/test_labels.npy', test_labels)
print("✅ OOD 彩色测试集生成完成！")