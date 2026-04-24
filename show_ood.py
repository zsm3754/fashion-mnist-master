import numpy as np
import matplotlib.pyplot as plt

# 加载你生成好的 OOD 数据
ood_imgs = np.load("./data/fashion/ood_test_images.npy")
ood_labs = np.load("./data/fashion/test_labels.npy")

# 展示 12 张图
sample_num = 12
plt.figure(figsize=(12, 8))

for i in range(sample_num):
    img = ood_imgs[i].reshape(28, 28, 3)
    plt.subplot(3, 4, i+1)  # 这里改成 3行4列，才能放12张
    plt.imshow(img)
    plt.title(f"label:{ood_labs[i]}")
    plt.axis("off")

plt.tight_layout()
plt.show()