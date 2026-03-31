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

# 拼接成绝对路径: .../algorithms/monitoring/weights/segment_best.pt
MODEL_PATH = str(monitoring_dir / "weights" / "segment_best.pt")


def run_segmentation(source_path):
    """
    运行分割模型，供界面调用
    """
    try:
        if not source_path:
            print("分割输入为空")
            return None
        if not os.path.exists(source_path):
            print(f"分割输入不存在: {source_path}")
            return None
        if not os.path.exists(MODEL_PATH):
            print(f"分割模型不存在: {MODEL_PATH}")
            return None

        print(f"当前ultralytics来源: {ultralytics.__file__}")
        print(f"正在加载分割模型: {MODEL_PATH}")
        model = YOLO(MODEL_PATH)

        results = model.predict(
            source=source_path,
            save=True,
            save_txt=True,
            save_conf=True,
            conf=0.25,
            iou=0.50,
            show_boxes=False,
            show_labels=True,
            show_conf=True,
        )

        if not results:
            print("分割无返回结果")
            return None

        save_dir = results[0].save_dir
        file_name = os.path.basename(source_path)
        result_path = os.path.join(save_dir, file_name)

        if not os.path.exists(result_path):
            base = os.path.splitext(result_path)[0]
            for ext in [".png", ".jpg", ".jpeg", ".bmp"]:
                candidate = base + ext
                if os.path.exists(candidate):
                    result_path = candidate
                    break

        if not os.path.exists(result_path):
            print(f"分割完成但未找到输出文件: {result_path}")
            return None

        print(f"分割完成，保存路径: {result_path}")
        return result_path

    except Exception as e:
        print(f"分割过程出错: {e}")
        print(traceback.format_exc())
        return None


if __name__ == "__main__":
    pass
