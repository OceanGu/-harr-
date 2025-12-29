import cv2
import numpy as np
import time
import matplotlib.pyplot as plt

# 设置matplotlib中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def compute_rect_sum_direct(img, x, y, w, h):
    """直接计算矩形区域像素和（慢）"""
    return np.sum(img[y:y + h, x:x + w])


def compute_rect_sum_integral(integral_img, x, y, w, h):
    """使用积分图计算矩形区域像素和（快）"""
    # 积分图坐标
    x1, y1 = x, y
    x2, y2 = x + w, y + h
    # 公式：Sum = I(D) + I(A) - I(B) - I(C)
    return (integral_img[y2, x2] + integral_img[y1, x1]
            - integral_img[y2, x1] - integral_img[y1, x2])


# 1. 准备一张测试图片
img = cv2.imread('test_face.jpg', cv2.IMREAD_GRAYSCALE)
h, w = img.shape

# 2. 预计算积分图
integral_img = cv2.integral(img)[1:, 1:]  # OpenCV的积分图会多一行一列

# 3. 设计对比实验：计算不同大小矩形区域的时间
rect_sizes = [(10, 10), (30, 30), (50, 50), (100, 100)]
times_direct = []
times_integral = []

for rw, rh in rect_sizes:
    # 随机选择矩形位置
    x = np.random.randint(0, w - rw)
    y = np.random.randint(0, h - rh)

    # 方法A: 直接计算
    start = time.perf_counter()
    for _ in range(1000):  # 重复多次以获得可测量时间
        _ = compute_rect_sum_direct(img, x, y, rw, rh)
    times_direct.append(time.perf_counter() - start)

    # 方法B: 积分图计算
    start = time.perf_counter()
    for _ in range(1000):
        _ = compute_rect_sum_integral(integral_img, x, y, rw, rh)
    times_integral.append(time.perf_counter() - start)

# 4. 可视化结果
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

# 柱状图对比
labels = [f'{w}x{h}' for w, h in rect_sizes]
x = np.arange(len(labels))
width = 0.35
ax1.bar(x - width / 2, times_direct, width, label='Direct Calculation', color='salmon')
ax1.bar(x + width / 2, times_integral, width, label='Integral Image', color='lightgreen')
ax1.set_xlabel('Rectangle Size')
ax1.set_ylabel('Time for 1000 calculations (seconds)')
ax1.set_title('Integral Image Speedup Verification')
ax1.set_xticks(x)
ax1.set_xticklabels(labels)
ax1.legend()
ax1.grid(True, alpha=0.3)

# 加速比曲线
speedup = np.array(times_direct) / np.array(times_integral)
ax2.plot(labels, speedup, 'o-', linewidth=2, markersize=8, color='blue')
ax2.axhline(y=1, color='r', linestyle='--', alpha=0.5)
ax2.set_xlabel('Rectangle Size')
ax2.set_ylabel('Speedup Ratio (Direct/Integral)')
ax2.set_title('Speedup Ratio vs Rectangle Size')
ax2.grid(True, alpha=0.3)
for i, sp in enumerate(speedup):
    ax2.text(i, sp + 0.1, f'{sp:.1f}x', ha='center')

plt.tight_layout()
plt.savefig('integral_image_speedup.png', dpi=300, bbox_inches='tight')
plt.show()