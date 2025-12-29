#!/usr/bin/env python3
"""
Haar人脸检测器参数调优研究
系统分析scaleFactor和minNeighbors对检测性能的影响
"""

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import time
import json
from tqdm import tqdm
import os
from sklearn.metrics import precision_score, recall_score, f1_score
import warnings

# 设置中文字体
import matplotlib as mpl

try:
    # Windows系统
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun']
    # Mac系统
    # plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'Heiti TC', 'STHeiti']
except:
    print("警告: 无法设置中文字体，图表可能显示乱码")

# 解决负号显示问题
plt.rcParams['axes.unicode_minus'] = False

warnings.filterwarnings('ignore')


class HaarParameterTuner:
    """Haar检测器参数调优分析类"""

    def __init__(self, cascade_path, test_data_dir, output_dir="results"):
        """
        初始化参数调优器

        参数:
        - cascade_path: Haar级联分类器文件路径
        - test_data_dir: 测试数据集目录
        - output_dir: 结果输出目录
        """
        self.cascade_path = cascade_path
        self.test_data_dir = Path(test_data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 加载Haar分类器
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        if self.face_cascade.empty():
            raise ValueError(f"无法加载级联分类器: {cascade_path}")

        # 存储实验结果
        self.results = []
        self.dataset_info = {}

        print(f"✓ Haar参数调优器初始化完成")
        print(f"  级联文件: {cascade_path}")
        print(f"  测试数据: {test_data_dir}")
        print(f"  输出目录: {output_dir}")

    def load_dataset(self, annotations_file=None):
        """
        加载测试数据集

        参数:
        - annotations_file: 标注文件路径（可选）
        """
        print("\n=== 加载测试数据集 ===")

        # 先尝试加载标注文件，如果存在则只使用有标注的图片
        if annotations_file and Path(annotations_file).exists():
            print("先加载标注文件确定图片范围...")
            temp_annotations = self._load_annotations_for_matching(annotations_file)
            if temp_annotations:
                # 只使用标注文件中存在的图片
                self.images = []
                annotated_image_names = {Path(k).name for k in temp_annotations.keys()}

                # 搜索图片，但只包含有标注的图片
                image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
                for ext in image_extensions:
                    found_images = list(self.test_data_dir.glob(f"**/{ext}"))
                    found_images.extend(list(self.test_data_dir.glob(f"**/{ext.upper()}")))

                    for img_path in found_images:
                        if img_path.name in annotated_image_names:
                            self.images.append(img_path)

                print(f"根据标注文件找到 {len(self.images)} 张有标注的图片")
            else:
                # 如果标注文件加载失败，使用所有图片
                self._find_all_images()
        else:
            # 没有标注文件，使用所有图片
            self._find_all_images()

        print(f"最终使用 {len(self.images)} 张测试图片")

        # 加载标注
        self.annotations = self._load_annotations(annotations_file)

        # 数据集统计
        total_faces = sum(len(boxes) for boxes in self.annotations.values())
        self.dataset_info = {
            'total_images': len(self.images),
            'total_faces': total_faces,
            'avg_faces_per_image': total_faces / len(self.images) if self.images else 0
        }

        print(f"数据集统计:")
        print(f"  图片数量: {self.dataset_info['total_images']}")
        print(f"  总人脸数: {self.dataset_info['total_faces']}")
        print(f"  平均每张图片人脸数: {self.dataset_info['avg_faces_per_image']:.2f}")

        return self.images, self.annotations

    def _find_all_images(self):
        """查找所有图片"""
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
        self.images = []

        for ext in image_extensions:
            self.images.extend(list(self.test_data_dir.glob(f"**/{ext}")))
            self.images.extend(list(self.test_data_dir.glob(f"**/{ext.upper()}")))

        # 去重
        self.images = list(set(self.images))

    def _load_annotations_for_matching(self, annotations_file):
        """仅用于匹配图片的标注加载"""
        if not annotations_file or not Path(annotations_file).exists():
            return {}

        try:
            with open(annotations_file, 'r', encoding='utf-8') as f:
                annotation_data = json.load(f)

            if isinstance(annotation_data, dict):
                if 'annotations' in annotation_data:
                    return annotation_data['annotations']
                else:
                    return annotation_data
            return {}
        except Exception as e:
            print(f"加载标注文件用于匹配失败: {e}")
            return {}

    def _load_annotations(self, annotations_file):
        """加载或生成标注数据"""
        annotations = {}

        print(f"尝试加载标注文件: {annotations_file}")

        if annotations_file and Path(annotations_file).exists():
            print("标注文件存在，尝试加载...")
            try:
                with open(annotations_file, 'r', encoding='utf-8') as f:
                    annotation_data = json.load(f)

                print(f"原始标注数据格式: {type(annotation_data)}")

                # 支持多种标注格式
                if isinstance(annotation_data, dict):
                    if 'annotations' in annotation_data:
                        # 格式1: 包含元数据的标注文件
                        raw_annotations = annotation_data['annotations']
                        print("使用格式1（包含元数据）")
                    else:
                        # 格式2: 直接的路径到框的映射
                        raw_annotations = annotation_data
                        print("使用格式2（直接映射）")
                else:
                    print(f"不支持的标注格式: {type(annotation_data)}")
                    raise ValueError("标注文件格式不支持")

                print(f"原始标注包含 {len(raw_annotations)} 个条目")

                # 转换路径格式以确保匹配
                loaded_count = 0
                for img_path_str, boxes in raw_annotations.items():
                    print(f"处理标注: {img_path_str} -> {len(boxes)} 个框")

                    # 将标注文件中的路径转换为与当前图片路径匹配的格式
                    matched = False
                    for img_path in self.images:
                        # 多种匹配方式
                        img_path_str_clean = str(img_path).replace('\\', '/')
                        annotation_path_clean = img_path_str.replace('\\', '/')

                        if (img_path_str_clean == annotation_path_clean or
                                Path(img_path).name == Path(annotation_path_clean).name):
                            annotations[str(img_path)] = boxes
                            loaded_count += 1
                            matched = True
                            print(f"  匹配成功: {Path(img_path).name}")
                            break

                    if not matched:
                        print(f"  警告: 未找到匹配的图片: {img_path_str}")

                print(f"✓ 成功加载 {loaded_count} 张图片的标注")

                if loaded_count == 0:
                    print("警告: 没有成功匹配任何标注，将使用伪标注")
                    return self._generate_pseudo_annotations()

                return annotations

            except Exception as e:
                print(f"加载标注文件失败: {e}")
                print("将使用自动生成的伪标注")
                return self._generate_pseudo_annotations()
        else:
            print(f"标注文件不存在: {annotations_file}")
            print("将使用自动生成的伪标注")
            return self._generate_pseudo_annotations()

    def _generate_pseudo_annotations(self):
        """生成伪标注"""
        print("生成伪标注...")
        annotations = {}
        for img_path in self.images:
            image = cv2.imread(str(img_path))
            if image is not None:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                # 使用中等严格的参数检测"真实"人脸
                faces = self.face_cascade.detectMultiScale(
                    gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
                )
                annotations[str(img_path)] = faces.tolist() if len(faces) > 0 else []
        return annotations

    def calculate_iou(self, box1, box2):
        """计算两个边界框的IOU（交并比）"""
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2

        # 计算交集坐标
        xi1 = max(x1, x2)
        yi1 = max(y1, y2)
        xi2 = min(x1 + w1, x2 + w2)
        yi2 = min(y1 + h1, y2 + h2)

        if xi2 <= xi1 or yi2 <= yi1:
            return 0.0

        # 计算交集和并集面积
        inter_area = (xi2 - xi1) * (yi2 - yi1)
        box1_area = w1 * h1
        box2_area = w2 * h2
        union_area = box1_area + box2_area - inter_area

        return inter_area / union_area if union_area > 0 else 0.0

    def evaluate_detection(self, detected_boxes, true_boxes, iou_threshold=0.5):
        """评估单张图片的检测性能"""
        if len(true_boxes) == 0:
            if len(detected_boxes) == 0:
                return 1.0, 1.0, 1.0, 0, 0, 0, 0.0  # 完美情况：无目标，无检测
            else:
                return 0.0, 0.0, 0.0, 0, len(detected_boxes), 0, 0.0  # 只有误检

        # 匹配检测框和真实框
        matched_true = [False] * len(true_boxes)
        matched_detections = []
        ious = []

        for det_idx, det_box in enumerate(detected_boxes):
            best_iou = 0
            best_true_idx = -1

            for true_idx, true_box in enumerate(true_boxes):
                if matched_true[true_idx]:
                    continue

                iou = self.calculate_iou(det_box, true_box)
                if iou > best_iou:
                    best_iou = iou
                    best_true_idx = true_idx

            if best_iou >= iou_threshold:
                matched_true[best_true_idx] = True
                matched_detections.append(det_idx)
                ious.append(best_iou)

        # 计算指标
        tp = len(matched_detections)  # 真正例
        fp = len(detected_boxes) - tp  # 假正例
        fn = len(true_boxes) - tp  # 假反例

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        avg_iou = np.mean(ious) if ious else 0

        return precision, recall, f1, tp, fp, fn, avg_iou

    def run_parameter_sweep(self, param_name, param_values, fixed_params=None,
                            sample_size=None, iou_threshold=0.5):
        """
        运行参数扫描实验

        参数:
        - param_name: 要扫描的参数名 ('scaleFactor' 或 'minNeighbors')
        - param_values: 参数值列表
        - fixed_params: 固定参数字典
        - sample_size: 采样图片数量（None表示使用全部）
        - iou_threshold: IOU阈值
        """
        if fixed_params is None:
            fixed_params = {}

        print(f"\n=== 扫描参数: {param_name} ===")
        print(f"参数值范围: {param_values}")
        print(f"固定参数: {fixed_params}")

        # 选择测试图片子集
        test_images = self.images[:sample_size] if sample_size else self.images
        print(f"使用 {len(test_images)} 张图片进行测试")

        results = []

        for param_value in tqdm(param_values, desc=f"扫描{param_name}"):
            # 设置检测参数
            detect_params = fixed_params.copy()
            detect_params[param_name] = param_value

            # 确保必要的参数存在
            if 'minSize' not in detect_params:
                detect_params['minSize'] = (30, 30)

            # 批量评估
            batch_results = self._evaluate_batch(test_images, detect_params, iou_threshold)

            # 汇总结果
            summary = self._summarize_batch_results(batch_results)
            summary['param_name'] = param_name
            summary['param_value'] = param_value
            summary['detect_params'] = detect_params

            results.append(summary)

            # 打印进度
            print(f"  {param_name}={param_value}: "
                  f"F1={summary['avg_f1']:.3f}, "
                  f"Precision={summary['avg_precision']:.3f}, "
                  f"Recall={summary['avg_recall']:.3f}, "
                  f"FPS={summary['fps']:.1f}")

        return results

    def _evaluate_batch(self, image_paths, detect_params, iou_threshold=0.5):
        """批量评估一组图片"""
        batch_results = []
        total_time = 0

        for img_path in image_paths:
            img_path_str = str(img_path)

            # 读取图片
            image = cv2.imread(img_path_str)
            if image is None:
                continue

            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            true_boxes = self.annotations.get(img_path_str, [])

            # 执行检测并计时
            start_time = time.time()
            detected_boxes = self.face_cascade.detectMultiScale(gray, **detect_params)
            detection_time = time.time() - start_time
            total_time += detection_time

            # 转换为列表格式
            detected_boxes = detected_boxes.tolist() if len(detected_boxes) > 0 else []

            # 评估检测结果
            precision, recall, f1, tp, fp, fn, avg_iou = self.evaluate_detection(
                detected_boxes, true_boxes, iou_threshold
            )

            batch_results.append({
                'image_path': img_path_str,
                'detected_boxes': detected_boxes,
                'true_boxes': true_boxes,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'tp': tp,
                'fp': fp,
                'fn': fn,
                'avg_iou': avg_iou,
                'detection_time': detection_time,
                'num_detections': len(detected_boxes)
            })

        return batch_results

    def _summarize_batch_results(self, batch_results):
        """汇总批量评估结果"""
        if not batch_results:
            return {}

        df = pd.DataFrame(batch_results)

        summary = {
            'avg_precision': df['precision'].mean(),
            'avg_recall': df['recall'].mean(),
            'avg_f1': df['f1'].mean(),
            'avg_iou': df['avg_iou'].mean(),
            'total_time': df['detection_time'].sum(),
            'avg_time': df['detection_time'].mean(),
            'fps': len(df) / df['detection_time'].sum() if df['detection_time'].sum() > 0 else 0,
            'total_tp': df['tp'].sum(),
            'total_fp': df['fp'].sum(),
            'total_fn': df['fn'].sum(),
            'total_detections': df['num_detections'].sum(),
            'num_images': len(df)
        }

        return summary

    def run_2d_parameter_sweep(self, scale_factors, min_neighbors_list,
                               fixed_params=None, sample_size=50):
        """运行二维参数网格搜索"""
        if fixed_params is None:
            fixed_params = {}

        print(f"\n=== 二维参数网格搜索 ===")
        print(f"scaleFactor: {scale_factors}")
        print(f"minNeighbors: {min_neighbors_list}")

        results_2d = []
        test_images = self.images[:sample_size] if sample_size else self.images

        total_combinations = len(scale_factors) * len(min_neighbors_list)
        pbar = tqdm(total=total_combinations, desc="二维参数扫描")

        for sf in scale_factors:
            for mn in min_neighbors_list:
                detect_params = fixed_params.copy()
                detect_params.update({
                    'scaleFactor': sf,
                    'minNeighbors': mn,
                    'minSize': (30, 30)
                })

                # 评估当前参数组合
                batch_results = self._evaluate_batch(test_images, detect_params)
                summary = self._summarize_batch_results(batch_results)

                results_2d.append({
                    'scaleFactor': sf,
                    'minNeighbors': mn,
                    'f1': summary['avg_f1'],
                    'precision': summary['avg_precision'],
                    'recall': summary['avg_recall'],
                    'fps': summary['fps'],
                    'avg_time': summary['avg_time'],
                    'total_tp': summary['total_tp'],
                    'total_fp': summary['total_fp'],
                    'total_fn': summary['total_fn']
                })

                pbar.update(1)
                pbar.set_postfix({
                    'sf': sf, 'mn': mn,
                    'f1': f"{summary['avg_f1']:.3f}",
                    'fps': f"{summary['fps']:.1f}"
                })

        pbar.close()
        return pd.DataFrame(results_2d)

    def visualize_parameter_analysis(self, results_1d, param_name, save_path=None):
        """可视化一维参数分析结果"""
        if not results_1d:
            print("没有结果数据可可视化")
            return

        # 确保保存目录存在
        if save_path:
            save_dir = Path(save_path).parent
            save_dir.mkdir(parents=True, exist_ok=True)

        # 准备数据
        param_values = [r['param_value'] for r in results_1d]
        metrics = ['avg_f1', 'avg_precision', 'avg_recall', 'fps']
        metric_labels = ['F1-Score', '精确率', '召回率', 'FPS']
        colors = ['blue', 'green', 'red', 'orange']

        # 创建图表
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()

        for idx, (metric, label, color) in enumerate(zip(metrics, metric_labels, colors)):
            metric_values = [r[metric] for r in results_1d]

            axes[idx].plot(param_values, metric_values, 'o-',
                           linewidth=2, markersize=6, color=color, label=label)
            axes[idx].set_xlabel(param_name, fontsize=12)
            axes[idx].set_ylabel(label, fontsize=12)
            axes[idx].set_title(f'{label} vs {param_name}', fontsize=14)
            axes[idx].grid(True, alpha=0.3)
            axes[idx].legend()

            # 标记最佳值
            if metric != 'fps':  # 对于精度指标，值越大越好
                best_idx = np.argmax(metric_values)
                best_value = metric_values[best_idx]
                best_param = param_values[best_idx]
                axes[idx].plot(best_param, best_value, 'r*', markersize=15,
                               label=f'最佳: {best_param}')
            else:  # 对于FPS，值越大越好
                best_idx = np.argmax(metric_values)
                best_value = metric_values[best_idx]
                best_param = param_values[best_idx]
                axes[idx].plot(best_param, best_value, 'g*', markersize=15,
                               label=f'最佳: {best_param}')

            axes[idx].legend()

        plt.suptitle(f'Haar检测器参数调优分析: {param_name}', fontsize=16, y=1.02)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"图表已保存: {save_path}")

        plt.show()

        return fig

    def visualize_2d_heatmap(self, results_2d, save_path=None):
        """可视化二维参数热力图"""
        if results_2d.empty:
            print("没有二维结果数据可可视化")
            return

        # 确保保存目录存在
        if save_path:
            save_dir = Path(save_path).parent
            save_dir.mkdir(parents=True, exist_ok=True)

        # 创建数据透视表
        pivot_f1 = results_2d.pivot(index='scaleFactor', columns='minNeighbors', values='f1')
        pivot_fps = results_2d.pivot(index='scaleFactor', columns='minNeighbors', values='fps')
        pivot_precision = results_2d.pivot(index='scaleFactor', columns='minNeighbors', values='precision')
        pivot_recall = results_2d.pivot(index='scaleFactor', columns='minNeighbors', values='recall')

        # 创建图表
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # F1热力图
        sns.heatmap(pivot_f1, annot=True, fmt='.3f', cmap='YlOrRd',
                    ax=axes[0, 0], cbar_kws={'label': 'F1-Score'})
        axes[0, 0].set_title('F1-Score 热力图', fontsize=14)
        axes[0, 0].set_xlabel('minNeighbors', fontsize=12)
        axes[0, 0].set_ylabel('scaleFactor', fontsize=12)

        # FPS热力图
        sns.heatmap(pivot_fps, annot=True, fmt='.1f', cmap='YlGnBu',
                    ax=axes[0, 1], cbar_kws={'label': 'FPS'})
        axes[0, 1].set_title('检测速度 (FPS) 热力图', fontsize=14)
        axes[0, 1].set_xlabel('minNeighbors', fontsize=12)
        axes[0, 1].set_ylabel('scaleFactor', fontsize=12)

        # Precision热力图
        sns.heatmap(pivot_precision, annot=True, fmt='.3f', cmap='Greens',
                    ax=axes[1, 0], cbar_kws={'label': '精确率'})
        axes[1, 0].set_title('精确率 (Precision) 热力图', fontsize=14)
        axes[1, 0].set_xlabel('minNeighbors', fontsize=12)
        axes[1, 0].set_ylabel('scaleFactor', fontsize=12)

        # Recall热力图
        sns.heatmap(pivot_recall, annot=True, fmt='.3f', cmap='Reds',
                    ax=axes[1, 1], cbar_kws={'label': '召回率'})
        axes[1, 1].set_title('召回率 (Recall) 热力图', fontsize=14)
        axes[1, 1].set_xlabel('minNeighbors', fontsize=12)
        axes[1, 1].set_ylabel('scaleFactor', fontsize=12)

        plt.suptitle('Haar检测器二维参数扫描结果', fontsize=16, y=1.02)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"热力图已保存: {save_path}")

        plt.show()

        return fig

    def find_optimal_parameters(self, results_2d, precision_weight=0.5, recall_weight=0.5, fps_weight=0.0):
        """根据权重找到最优参数组合"""
        if results_2d.empty:
            return None

        # 归一化指标
        df = results_2d.copy()
        df['f1_norm'] = (df['f1'] - df['f1'].min()) / (df['f1'].max() - df['f1'].min())
        df['precision_norm'] = (df['precision'] - df['precision'].min()) / (
                df['precision'].max() - df['precision'].min())
        df['recall_norm'] = (df['recall'] - df['recall'].min()) / (df['recall'].max() - df['recall'].min())
        df['fps_norm'] = (df['fps'] - df['fps'].min()) / (df['fps'].max() - df['fps'].min())

        # 计算加权得分
        df['weighted_score'] = (
                precision_weight * df['precision_norm'] +
                recall_weight * df['recall_norm'] +
                fps_weight * df['fps_norm']
        )

        # 找到最佳组合
        best_idx = df['weighted_score'].idxmax()
        best_params = df.loc[best_idx]

        print("\n=== 最优参数分析 ===")
        print(f"权重设置: 精确率={precision_weight}, 召回率={recall_weight}, FPS={fps_weight}")
        print(f"最佳参数组合:")
        print(f"  scaleFactor = {best_params['scaleFactor']}")
        print(f"  minNeighbors = {best_params['minNeighbors']}")
        print(f"  性能指标:")
        print(f"    F1-Score: {best_params['f1']:.4f}")
        print(f"    精确率: {best_params['precision']:.4f}")
        print(f"    召回率: {best_params['recall']:.4f}")
        print(f"    FPS: {best_params['fps']:.2f}")
        print(f"    加权得分: {best_params['weighted_score']:.4f}")

        return best_params

    def save_results(self, results_1d_sf, results_1d_mn, results_2d, optimal_params):
        """保存所有实验结果"""
        # 保存一维结果
        pd.DataFrame(results_1d_sf).to_csv(self.output_dir / 'scale_factor_results.csv', index=False)
        pd.DataFrame(results_1d_mn).to_csv(self.output_dir / 'min_neighbors_results.csv', index=False)

        # 保存二维结果
        results_2d.to_csv(self.output_dir / '2d_parameter_sweep.csv', index=False)

        # 保存最优参数
        with open(self.output_dir / 'optimal_parameters.json', 'w') as f:
            json.dump(optimal_params.to_dict(), f, indent=2)

        # 保存数据集信息
        with open(self.output_dir / 'dataset_info.json', 'w') as f:
            json.dump(self.dataset_info, f, indent=2)

        print(f"\n✓ 所有结果已保存到: {self.output_dir}")


def main():
    """主实验函数"""
    print("=== Haar人脸检测器参数调优实验 ===")

    # 1. 初始化参数调优器
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    test_data_dir = "data/images"  # 修改为您的测试数据集路径
    output_dir = "results"

    # 指定标注文件路径
    annotations_file = "./data/annotations/annotations.json"  # 修改为您的标注文件路径

    tuner = HaarParameterTuner(cascade_path, test_data_dir, output_dir)

    # 2. 加载数据集
    images, annotations = tuner.load_dataset(annotations_file)

    if len(images) == 0:
        print("错误: 未找到测试图片！")
        print("请将测试图片放入 data/images/ 目录")
        return

    # 3. 实验1: 扫描scaleFactor参数
    print("\n" + "=" * 60)
    print("实验1: scaleFactor参数影响分析")
    print("=" * 60)

    scale_factors = [1.01, 1.05, 1.1, 1.2, 1.3, 1.5, 2.0]
    fixed_params = {'minNeighbors': 5, 'minSize': (30, 30)}

    results_sf = tuner.run_parameter_sweep(
        'scaleFactor', scale_factors, fixed_params, sample_size=50
    )

    # 4. 实验2: 扫描minNeighbors参数
    print("\n" + "=" * 60)
    print("实验2: minNeighbors参数影响分析")
    print("=" * 60)

    min_neighbors_list = [1, 2, 3, 4, 5, 6, 8, 10, 15, 20]
    fixed_params = {'scaleFactor': 1.1, 'minSize': (30, 30)}

    results_mn = tuner.run_parameter_sweep(
        'minNeighbors', min_neighbors_list, fixed_params, sample_size=50
    )

    # 5. 实验3: 二维参数网格搜索
    print("\n" + "=" * 60)
    print("实验3: 二维参数网格搜索")
    print("=" * 60)

    scale_factors_2d = [1.01, 1.05, 1.1, 1.2, 1.3]
    min_neighbors_2d = [1, 2, 3, 4, 5, 6, 8]

    results_2d = tuner.run_2d_parameter_sweep(
        scale_factors_2d, min_neighbors_2d, sample_size=50
    )

    # 6. 可视化结果
    print("\n" + "=" * 60)
    print("生成可视化结果")
    print("=" * 60)

    # 创建plots目录
    (tuner.output_dir / 'plots').mkdir(exist_ok=True)

    # 一维参数分析图
    tuner.visualize_parameter_analysis(
        results_sf, 'scaleFactor',
        save_path=str(tuner.output_dir / 'plots' / 'scale_factor_analysis.png')
    )

    tuner.visualize_parameter_analysis(
        results_mn, 'minNeighbors',
        save_path=str(tuner.output_dir / 'plots' / 'min_neighbors_analysis.png')
    )

    # 二维热力图
    tuner.visualize_2d_heatmap(
        results_2d,
        save_path=str(tuner.output_dir / 'plots' / 'parameter_heatmap.png')
    )

    # 7. 寻找最优参数
    print("\n" + "=" * 60)
    print("最优参数分析")
    print("=" * 60)

    # 不同权重的最优参数
    optimal_balanced = tuner.find_optimal_parameters(results_2d, 0.5, 0.5, 0.0)  # 平衡
    optimal_precision = tuner.find_optimal_parameters(results_2d, 0.8, 0.2, 0.0)  # 精确率优先
    optimal_recall = tuner.find_optimal_parameters(results_2d, 0.2, 0.8, 0.0)  # 召回率优先
    optimal_fast = tuner.find_optimal_parameters(results_2d, 0.3, 0.3, 0.4)  # 速度优先

    # 8. 保存结果
    tuner.save_results(results_sf, results_mn, results_2d, optimal_balanced)

    print("\n" + "=" * 60)
    print("实验完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()