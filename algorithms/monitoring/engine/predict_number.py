import os
import sys
import traceback
from pathlib import Path

current_dir = Path(__file__).resolve().parent
monitoring_dir = current_dir.parent
if str(monitoring_dir) not in sys.path:
    sys.path.insert(0, str(monitoring_dir))

import ultralytics
from ultralytics import YOLO

# 拼接成绝对路径: .../algorithms/monitoring/weights/detect_best.pt
MODEL_PATH = str(monitoring_dir / "weights" / "detect_best.pt")


def run_number_detection(source_path):
    """
    运行数字检测，供界面调用
    返回: (结果图路径, 标签txt路径)
    """
    try:
        if not source_path:
            print("数字识别输入为空")
            return None, None
        if not os.path.exists(source_path):
            print(f"数字识别输入不存在: {source_path}")
            return None, None
        if not os.path.exists(MODEL_PATH):
            print(f"数字识别模型不存在: {MODEL_PATH}")
            return None, None

        print(f"当前ultralytics来源: {ultralytics.__file__}")
        print(f"正在加载数字识别模型: {MODEL_PATH}")
        model = YOLO(MODEL_PATH)

        results = model.predict(
            source=source_path,
            save=True,
            save_txt=True,
            save_conf=True,
            conf=0.25,
            iou=0.50,
            show_boxes=True,
        )

        if not results:
            print("数字识别无返回结果")
            return None, None

        save_dir = results[0].save_dir
        filename = os.path.basename(source_path)
        result_img_path = os.path.join(save_dir, filename)

        if not os.path.exists(result_img_path):
            base = os.path.splitext(result_img_path)[0]
            for ext in [".png", ".jpg", ".jpeg", ".bmp"]:
                candidate = base + ext
                if os.path.exists(candidate):
                    result_img_path = candidate
                    break

        txt_name = os.path.splitext(filename)[0] + ".txt"
        label_path = os.path.join(save_dir, "labels", txt_name)

        if not os.path.exists(result_img_path):
            print(f"数字识别完成但未找到输出图像: {result_img_path}")
            return None, None

        print(f"识别完成，Label路径: {label_path}")
        return result_img_path, label_path

    except Exception as e:
        print(f"数字识别出错: {e}")
        print(traceback.format_exc())
        return None, None


if __name__ == "__main__":
    pass
