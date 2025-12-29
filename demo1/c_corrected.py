import cv2
import numpy as np
import time
import matplotlib.pyplot as plt


def simulate_cascade_process(image_path, cascade_path, num_stages_to_log=5):
    """
    模拟并记录一个子窗口通过级联分类器前几级的状态
    修复版本：更准确地模拟级联分类器的行为
    """
    img = cv2.imread(image_path)
    if img is None:
        print(f"错误：无法加载图像 {image_path}")
        return None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    original_cascade = cv2.CascadeClassifier(cascade_path)

    if original_cascade.empty():
        print(f"错误：无法加载级联分类器 {cascade_path}")
        return None

    print(f"图像尺寸: {gray.shape}")
    print(f"级联分类器加载成功")

    # 首先在整张图上检测人脸
    faces = original_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    # 获取图片尺寸
    img_height, img_width = gray.shape

    # 准备ROI区域
    rois = []

    if len(faces) > 0:
        # 使用第一个检测到的人脸作为正样本
        x, y, w, h = faces[0]
        rois.append(('Real Face', (x, y, w, h)))
        print(f"检测到真实人脸: ({x}, {y}, {w}, {h})")
    else:
        print("未检测到人脸，使用图像中心区域作为正样本示例")
        # 使用图像中心区域作为示例
        center_x, center_y = img_width // 2, img_height // 2
        roi_size = min(img_width, img_height) // 3
        rois.append(
            ('Center Region (Face-like)', (center_x - roi_size // 2, center_y - roi_size // 2, roi_size, roi_size)))

    # 添加一个明确的背景区域作为负样本
    bg_size = min(img_width, img_height) // 4
    rois.append(('Background (Non-face)', (0, 0, bg_size, bg_size)))

    # 如果图像足够大，再添加一个背景区域
    if img_width > bg_size * 2 and img_height > bg_size * 2:
        rois.append(('Background Corner', (img_width - bg_size, img_height - bg_size, bg_size, bg_size)))

    print(f"\n选择的ROI区域:")
    for name, (x, y, w, h) in rois:
        print(f"  {name}: ({x}, {y}, {w}, {h})")

    results = []

    for roi_name, (x, y, w, h) in rois:
        # 确保ROI在图像边界内
        x = max(0, min(x, img_width - 1))
        y = max(0, min(y, img_height - 1))
        w = min(w, img_width - x)
        h = min(h, img_height - y)

        if w <= 0 or h <= 0:
            print(f"警告：ROI {roi_name} 超出图像边界，跳过")
            continue

        # 裁剪区域
        sub_window = gray[y:y + h, x:x + w]

        # 检查裁剪后的图像是否为空
        if sub_window.size == 0:
            print(f"警告：ROI {roi_name} 裁剪后为空，跳过")
            continue

        print(f"\n处理ROI {roi_name}: 子窗口尺寸 {sub_window.shape}")

        # 模拟级联分类器的多个阶段，每个阶段逐渐变严格
        # 这更接近真实级联分类器的工作原理
        cascade_stages = [
            {'name': 'Stage 1 (Coarse)', 'scaleFactor': 1.3, 'minNeighbors': 1, 'minSize': (20, 20)},
            {'name': 'Stage 2 (Medium)', 'scaleFactor': 1.2, 'minNeighbors': 2, 'minSize': (22, 22)},
            {'name': 'Stage 3 (Fine)', 'scaleFactor': 1.1, 'minNeighbors': 3, 'minSize': (24, 24)},
            {'name': 'Stage 4 (Strict)', 'scaleFactor': 1.05, 'minNeighbors': 5, 'minSize': (24, 24)},
            {'name': 'Stage 5 (Very Strict)', 'scaleFactor': 1.02, 'minNeighbors': 8, 'minSize': (24, 24)},
        ]

        stage_status = []

        for i, stage in enumerate(cascade_stages):
            try:
                # 将子窗口缩放到标准大小进行检测
                sub_img = cv2.resize(sub_window, (24, 24))

                # 在这个阶段进行检测
                faces_in_stage = original_cascade.detectMultiScale(
                    sub_img,
                    scaleFactor=stage['scaleFactor'],
                    minNeighbors=stage['minNeighbors'],
                    minSize=stage['minSize'],
                    flags=cv2.CASCADE_SCALE_IMAGE
                )

                passed = len(faces_in_stage) > 0

                stage_status.append({
                    'stage_num': i + 1,
                    'stage_name': stage['name'],
                    'scaleFactor': stage['scaleFactor'],
                    'minNeighbors': stage['minNeighbors'],
                    'passed': passed,
                    'num_detections': len(faces_in_stage),
                    'detection_coords': faces_in_stage.tolist() if passed else []
                })

                status_str = "PASS" if passed else "REJECT"
                print(f"  {stage['name']}: {status_str} (检测到 {len(faces_in_stage)} 个人脸)")

                # 如果当前阶段未通过，模拟级联分类器的早期拒绝机制
                if not passed:
                    print(f"       -> 在阶段 {i + 1} 被早期拒绝，不再继续后续阶段")
                    break

            except Exception as e:
                print(f"  阶段 {i + 1} 出错: {e}")
                stage_status.append({
                    'stage_num': i + 1,
                    'stage_name': stage['name'],
                    'passed': False,
                    'num_detections': 0,
                    'error': str(e)
                })
                break

        results.append({
            'roi_name': roi_name,
            'roi_coord': (x, y, w, h),
            'stage_status': stage_status
        })

    # 打印详细的模拟结果
    print("\n" + "=" * 80)
    print("CASCADE CLASSIFIER PROCESS SIMULATION RESULTS")
    print("=" * 80)

    for res in results:
        print(f"\nROI: {res['roi_name']} at {res['roi_coord']}")
        print("Stage | Name                | Scale | Neighbors | Passed? | #Detections")
        print("-" * 75)

        for stage in res['stage_status']:
            status = "PASS" if stage['passed'] else "REJECT"
            scale = stage.get('scaleFactor', 'N/A')
            neighbors = stage.get('minNeighbors', 'N/A')

            print(
                f"  {stage['stage_num']:2d}  | {stage['stage_name']:<18s} | {scale:>5} | {neighbors:>8} |   {status:^7s} |      {stage['num_detections']}")

            if not stage['passed']:
                print(f"       -> Early Rejection at stage {stage['stage_num']}")
                break

        if res['stage_status'] and all([s['passed'] for s in res['stage_status']]):
            print("       -> PASSED ALL STAGES (Classified as FACE)")
        elif res['stage_status']:
            print(f"       -> REJECTED at stage {len(res['stage_status'])} (Classified as NON-FACE)")

    # 可视化结果
    img_disp = img.copy()

    # 定义颜色
    colors = {
        'Real Face': (0, 255, 0),  # 绿色
        'Center Region (Face-like)': (255, 255, 0),  # 黄色
        'Background (Non-face)': (0, 0, 255),  # 红色
        'Background Corner': (255, 0, 255)  # 紫色
    }

    for res in results:
        x, y, w, h = res['roi_coord']
        color = colors.get(res['roi_name'], (255, 255, 255))

        # 绘制ROI框
        cv2.rectangle(img_disp, (x, y), (x + w, y + h), color, 2)

        # 添加ROI标签
        label = res['roi_name']
        cv2.putText(img_disp, label, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # 添加最终结果
        if res['stage_status']:
            final_result = "FACE" if res['stage_status'][-1]['passed'] else "NON-FACE"
            result_color = (0, 255, 0) if final_result == "FACE" else (0, 0, 255)
            cv2.putText(img_disp, f"Result: {final_result}", (x, y + h + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, result_color, 2)

    # 显示结果
    plt.figure(figsize=(12, 10))
    plt.imshow(cv2.cvtColor(img_disp, cv2.COLOR_BGR2RGB))
    plt.title('Cascade Classifier Early Rejection Mechanism Simulation', fontsize=14)
    plt.axis('off')

    # 添加图例
    legend_elements = []
    for roi_name, color in colors.items():
        if any(r['roi_name'] == roi_name for r in results):
            legend_elements.append(plt.Rectangle((0, 0), 1, 1, fc=[c / 255 for c in color], label=roi_name))

    if legend_elements:
        plt.legend(handles=legend_elements, loc='upper right')

    plt.tight_layout()
    plt.savefig('cascade_simulation_corrected.png', dpi=300, bbox_inches='tight')
    plt.show()

    return results


# 运行模拟
if __name__ == "__main__":
    try:
        # 确保图像文件存在
        image_path = 'test_face.jpg'
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'

        print("开始级联分类器过程模拟...")
        print("=" * 50)

        sim_results = simulate_cascade_process(image_path, cascade_path)

        if sim_results is None:
            print("模拟过程失败，请检查图像路径和分类器文件")
        else:
            print("\n模拟过程完成！")
            print(f"总共处理了 {len(sim_results)} 个ROI区域")

    except Exception as e:
        print(f"程序执行出错: {e}")
        import traceback

        traceback.print_exc()