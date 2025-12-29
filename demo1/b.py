# 注意：此实验需要访问级联分类器的内部特征，OpenCV的Python接口可能不直接提供。
# 一种替代方案是使用OpenCV的C++接口或使用预计算的Haar特征进行模拟展示。
# 以下是使用OpenCV可视化检测结果，并标注特征区域的示意性方法。
import cv2
import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from PIL import Image, ImageDraw, ImageFont
def put_chinese_text(img, text, position, font_size=20, color=(255, 255, 255)):
    """在OpenCV图像上添加中文文字"""
    # 将OpenCV图像转换为PIL图像
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    
    try:
        # 尝试使用系统中文字体
        font = ImageFont.truetype("C:/Windows/Fonts/simhei.ttf", font_size)
    except:
        try:
            # 备选字体
            font = ImageFont.truetype("C:/Windows/Fonts/msyh.ttf", font_size)
        except:
            # 如果没有中文字体，使用默认字体
            font = ImageFont.load_default()
    
    # 添加文字
    draw.text(position, text, font=font, fill=color)
    
    # 转换回OpenCV格式
    img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    return img_cv

def visualize_haar_features_on_face(img_path, cascade_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cascade_path)

    # 检测人脸
    faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        # 绘制人脸框
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # 在每个人脸区域上，手动绘制几种典型的Haar特征模板，以解释其原理
        roi = img[y:y + h, x:x + w]
        roi_h, roi_w = roi.shape[:2]

        # 示例1: 边缘特征（模拟眼睛区域，上暗下亮）
        edge_x, edge_y = int(roi_w * 0.2), int(roi_h * 0.3)
        edge_w, edge_h = int(roi_w * 0.6), int(roi_h * 0.1)
        cv2.rectangle(img, (x + edge_x, y + edge_y),
                      (x + edge_x + edge_w, y + edge_y + edge_h // 2),
                      (0, 0, 255), 1)  # 上矩形（模拟暗区域）
        cv2.rectangle(img, (x + edge_x, y + edge_y + edge_h // 2),
                      (x + edge_x + edge_w, y + edge_y + edge_h),
                      (0, 255, 255), 1)  # 下矩形（模拟亮区域）
        # 使用中文文字
        img = put_chinese_text(img, '边缘特征(眉毛)', 
                              (x + edge_x, y + edge_y - 25), 
                              font_size=16, color=(0, 0, 255))

        # 示例2: 线特征（模拟鼻梁，两侧暗中间亮）
        line_x, line_y = int(roi_w * 0.4), int(roi_h * 0.4)
        line_w, line_h = int(roi_w * 0.2), int(roi_h * 0.3)
        cv2.rectangle(img, (x + line_x, y + line_y),
                      (x + line_x + line_w // 3, y + line_y + line_h),
                      (0, 0, 255), 1)  # 左矩形（暗）
        cv2.rectangle(img, (x + line_x + line_w // 3, y + line_y),
                      (x + line_x + 2 * line_w // 3, y + line_y + line_h),
                      (0, 255, 255), 1)  # 中矩形（亮）
        cv2.rectangle(img, (x + line_x + 2 * line_w // 3, y + line_y),
                      (x + line_x + line_w, y + line_y + line_h),
                      (0, 0, 255), 1)  # 右矩形（暗）
        # 使用中文文字
        img = put_chinese_text(img, '线特征(鼻梁)', 
                              (x + line_x, y + line_y - 25), 
                              font_size=16, color=(0, 0, 255))

        # 示例3: 中心环绕特征（模拟嘴巴，周围暗中间亮）
        center_x, center_y = int(roi_w * 0.3), int(roi_h * 0.6)
        center_w, center_h = int(roi_w * 0.4), int(roi_h * 0.15)
        cv2.rectangle(img, (x + center_x, y + center_y),
                      (x + center_x + center_w, y + center_y + center_h),
                      (255, 0, 255), 2)  # 外框
        cv2.rectangle(img, (x + center_x + center_w // 4, y + center_y + center_h // 4),
                      (x + center_x + 3 * center_w // 4, y + center_y + 3 * center_h // 4),
                      (0, 255, 255), 1)  # 内框
        # 使用中文文字
        img = put_chinese_text(img, '中心环绕特征(嘴巴)', 
                              (x + center_x, y + center_y - 40), 
                              font_size=16, color=(255, 0, 255))

    # 设置matplotlib中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    plt.figure(figsize=(12, 10))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title('Haar-like特征模板在人脸区域的可视化', fontsize=16)
    plt.axis('off')
    plt.savefig('haar_features_visualization.png', dpi=300, bbox_inches='tight')
    plt.show()


# 运行可视化
visualize_haar_features_on_face('test_face.jpg',
                                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
