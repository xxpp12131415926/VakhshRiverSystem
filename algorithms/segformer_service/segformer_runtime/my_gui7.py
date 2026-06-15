import sys
import types
import os
import cv2
import numpy as np
import torch
import warnings
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

warnings.filterwarnings("ignore")


# 修复 torch.fx 问题（针对旧版 PyTorch）
def patch_pytorch_fx():
    """为旧版 PyTorch 添加 torch.fx 支持"""
    if not hasattr(torch, 'fx'):
        class MockFX:
            @staticmethod
            def wrap(func):
                return func

        torch.fx = MockFX()
        print("Applied torch.fx patch for older PyTorch version")
    elif not hasattr(torch.fx, 'wrap'):
        def safe_fx_wrap(func):
            return func

        torch.fx.wrap = safe_fx_wrap
        print("Applied torch.fx.wrap patch")


patch_pytorch_fx()


# 修补 ext_loader.load_ext 函数
def patched_load_ext(name, funcs):
    mock_module = types.ModuleType(f'mmcv.{name}')
    for func_name in funcs:
        def make_mock_func(fname):
            def mock_func(*args, **kwargs):
                raise NotImplementedError(
                    f"Extension function '{fname}' is not available. "
                    f"This happens when mmcv-full is not properly installed with CUDA extensions. "
                    f"Install with: pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu102/torch1.12.1/index.html"
                )

            return mock_func

        setattr(mock_module, func_name, make_mock_func(func_name))
    return mock_module


try:
    import mmcv.utils.ext_loader

    mmcv.utils.ext_loader.load_ext = patched_load_ext
except ImportError:
    pass

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QMessageBox, QProgressBar,
    QComboBox, QGroupBox, QCheckBox
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QPixmap, QImage, QFont

# 地理空间处理库
try:
    from osgeo import gdal, ogr, osr
    import geopandas as gpd
    from shapely.geometry import Polygon
    import rasterio
    from rasterio.features import shapes
    from shapely.geometry import shape
    import pandas as pd
    from shapely.ops import unary_union
    from shapely.geometry import MultiPolygon

    HAS_GEO_LIBS = True
    print("✓ 地理空间处理库加载成功")
except ImportError as e:
    print(f"✗ 缺少地理空间处理库: {e}")
    print("请安装: conda install -c conda-forge gdal geopandas rasterio shapely fiona -y")
    HAS_GEO_LIBS = False

# 现在可以安全导入 mmseg
from mmseg.apis import inference_segmentor, init_segmentor
from mmseg.core.evaluation import get_palette


class FastGeoSegmenter:
    """快速地理空间分割类，优化数据格式保存"""

    def __init__(self):
        self.temp_dir = "./temp_geo_output"
        os.makedirs(self.temp_dir, exist_ok=True)

    def read_geotiff(self, tif_path):
        """
        读取GeoTIFF文件，返回图像数据和地理变换参数
        """
        if not HAS_GEO_LIBS:
            raise ImportError(
                "需要安装地理空间处理库: conda install -c conda-forge gdal geopandas rasterio shapely fiona -y")

        dataset = gdal.Open(tif_path)
        if dataset is None:
            raise ValueError(f"无法打开GeoTIFF文件: {tif_path}")

        width = dataset.RasterXSize
        height = dataset.RasterYSize
        bands = dataset.RasterCount

        if bands >= 3:
            r_band = dataset.GetRasterBand(1).ReadAsArray()
            g_band = dataset.GetRasterBand(2).ReadAsArray()
            b_band = dataset.GetRasterBand(3).ReadAsArray()
            if bands > 3:
                image_array = np.stack([r_band, g_band, b_band], axis=2)
            else:
                image_array = np.stack([r_band, g_band, b_band], axis=2)
        else:
            single_band = dataset.GetRasterBand(1).ReadAsArray()
            image_array = np.stack([single_band, single_band, single_band], axis=2)

        geo_transform = dataset.GetGeoTransform()
        projection = dataset.GetProjection()

        dataset = None

        return image_array, geo_transform, projection, (width, height)

    def array_to_geotiff(self, array, output_path, geo_transform, projection):
        """
        将数组保存为GeoTIFF文件
        """
        if not HAS_GEO_LIBS:
            raise ImportError(
                "需要安装地理空间处理库: conda install -c conda-forge gdal geopandas rasterio shapely fiona -y")

        height, width = array.shape[:2]

        driver = gdal.GetDriverByName('GTiff')
        if len(array.shape) == 3:
            bands = array.shape[2]
        else:
            bands = 1
            array = array[:, :, np.newaxis]

        out_dataset = driver.Create(output_path, width, height, bands, gdal.GDT_Byte)
        out_dataset.SetGeoTransform(geo_transform)
        out_dataset.SetProjection(projection)

        for i in range(bands):
            band = out_dataset.GetRasterBand(i + 1)
            band.WriteArray(array[:, :, i])

        out_dataset.FlushCache()
        out_dataset = None

    def fast_mask_to_vector(self, mask, geo_transform, projection, output_shp_path, simplify_tolerance=0.5):
        """
        快速将分割掩码转换为矢量文件(SHP) - 使用简化算法提高速度
        """
        if not HAS_GEO_LIBS:
            raise ImportError(
                "需要安装地理空间处理库: conda install -c conda-forge gdal geopandas rasterio shapely fiona -y")

        print(f"      开始快速栅格转矢量...")
        print(f"      掩码尺寸: {mask.shape}")
        print(f"      简化容差: {simplify_tolerance}")

        # 创建临时GeoTIFF文件
        temp_tif = os.path.join(self.temp_dir, "temp_mask.tif")
        self.array_to_geotiff(mask.astype(np.uint8), temp_tif, geo_transform, projection)

        start_time = time.time()

        with rasterio.open(temp_tif) as src:
            image = src.read(1)

            # 只处理目标值（1）
            mask_binary = image == 1
            print(f"      目标区域像素数: {np.sum(mask_binary)}")

            # 使用shapes提取多边形，但限制最大面积以跳过噪声
            shapes_gen = shapes(image, mask=mask_binary, transform=src.transform)

            polygons = []
            total_polygons = 0

            for i, (geom, val) in enumerate(shapes_gen):
                if val == 1:  # 只处理目标类别
                    shapely_geom = shape(geom)

                    # 简化多边形以提高性能
                    if simplify_tolerance > 0:
                        shapely_geom = shapely_geom.simplify(simplify_tolerance, preserve_topology=True)

                    # 跳过非常小的多边形（可能是噪声）
                    if shapely_geom.area > 0.001:  # 阈值可根据需要调整
                        polygons.append(shapely_geom)
                        total_polygons += 1

                    if i % 10000 == 0:  # 每处理10000个形状更新一次进度
                        print(f"      处理进度: {i} 个形状")

        print(f"      共处理 {total_polygons} 个多边形")

        if polygons:
            # 合并相邻的多边形以减少数量
            print(f"      合并相邻多边形...")
            if len(polygons) > 1:
                merged_polygon = unary_union(polygons)
                if isinstance(merged_polygon, MultiPolygon):
                    final_polygons = list(merged_polygon.geoms)
                else:
                    final_polygons = [merged_polygon]
            else:
                final_polygons = polygons

            print(f"      合并后剩余 {len(final_polygons)} 个多边形")

            # 创建GeoDataFrame并保存
            print(f"      创建GeoDataFrame并保存SHP...")
            gdf = gpd.GeoDataFrame({'geometry': final_polygons})
            gdf.crs = src.crs
            gdf.to_file(output_shp_path, driver='ESRI Shapefile')
            print(f"      ✓ 矢量文件已保存: {output_shp_path}")
        else:
            print(f"      没有检测到有效的分割区域")

        # 清理临时文件
        if os.path.exists(temp_tif):
            os.remove(temp_tif)

        elapsed_time = time.time() - start_time
        print(f"      快速矢量化完成，耗时: {elapsed_time:.2f}s")

    def fast_mask_to_geojson(self, mask, geo_transform, projection, output_geojson_path, simplify_tolerance=0.5):
        """
        快速将分割掩码转换为GeoJSON文件 - 使用简化算法
        """
        if not HAS_GEO_LIBS:
            raise ImportError(
                "需要安装地理空间处理库: conda install -c conda-forge gdal geopandas rasterio shapely fiona -y")

        print(f"      开始快速栅格转GeoJSON...")
        print(f"      掩码尺寸: {mask.shape}")
        print(f"      简化容差: {simplify_tolerance}")

        # 创建临时GeoTIFF文件
        temp_tif = os.path.join(self.temp_dir, "temp_mask.tif")
        self.array_to_geotiff(mask.astype(np.uint8), temp_tif, geo_transform, projection)

        start_time = time.time()

        with rasterio.open(temp_tif) as src:
            image = src.read(1)

            # 只处理目标值（1）
            mask_binary = image == 1
            print(f"      目标区域像素数: {np.sum(mask_binary)}")

            # 使用shapes提取多边形
            shapes_gen = shapes(image, mask=mask_binary, transform=src.transform)

            polygons = []
            total_polygons = 0

            for i, (geom, val) in enumerate(shapes_gen):
                if val == 1:  # 只处理目标类别
                    shapely_geom = shape(geom)

                    # 简化多边形
                    if simplify_tolerance > 0:
                        shapely_geom = shapely_geom.simplify(simplify_tolerance, preserve_topology=True)

                    # 跳过非常小的多边形
                    if shapely_geom.area > 0.001:
                        polygons.append(shapely_geom)
                        total_polygons += 1

                    if i % 10000 == 0:
                        print(f"      处理进度: {i} 个形状")

        print(f"      共处理 {total_polygons} 个多边形")

        if polygons:
            # 合并相邻多边形
            print(f"      合并相邻多边形...")
            if len(polygons) > 1:
                merged_polygon = unary_union(polygons)
                if isinstance(merged_polygon, MultiPolygon):
                    final_polygons = list(merged_polygon.geoms)
                else:
                    final_polygons = [merged_polygon]
            else:
                final_polygons = polygons

            print(f"      合并后剩余 {len(final_polygons)} 个多边形")

            # 创建GeoDataFrame并保存
            print(f"      创建GeoDataFrame并保存GeoJSON...")
            gdf = gpd.GeoDataFrame({'geometry': final_polygons})
            gdf.crs = src.crs
            gdf.to_file(output_geojson_path, driver='GeoJSON')
            print(f"      ✓ GeoJSON文件已保存: {output_geojson_path}")
        else:
            print(f"      没有检测到有效的分割区域")

        # 清理临时文件
        if os.path.exists(temp_tif):
            os.remove(temp_tif)

        elapsed_time = time.time() - start_time
        print(f"      快速GeoJSON化完成，耗时: {elapsed_time:.2f}s")

    def sliding_predict_and_save(
            self,
            tif_path,
            model,
            output_path,
            tile_size=1024,
            overlap=128):

        dataset = gdal.Open(tif_path)
        width = dataset.RasterXSize
        height = dataset.RasterYSize

        geo_transform = dataset.GetGeoTransform()
        projection = dataset.GetProjection()

        step = tile_size - overlap
        driver = gdal.GetDriverByName("GTiff")

        # 创建输出 mask
        out_dataset = driver.Create(
            output_path,
            width,
            height,
            1,
            gdal.GDT_Byte
        )
        out_dataset.SetGeoTransform(geo_transform)
        out_dataset.SetProjection(projection)
        out_band = out_dataset.GetRasterBand(1)

        # 计算总 tile 数量（确保覆盖整个图像）
        total_x_tiles = (width + step - 1) // step
        total_y_tiles = (height + step - 1) // step
        total = total_x_tiles * total_y_tiles
        index = 0

        print(f"开始大图推理: {tif_path}")
        print(f"图像尺寸: {width}x{height}")
        print(f"Tile大小: {tile_size}, 重叠: {overlap}")
        print(f"预计处理 {total} 个tiles...")

        for y in range(0, height, step):
            read_h = min(tile_size, height - y)
            for x in range(0, width, step):
                read_w = min(tile_size, width - x)
                index += 1

                # 读取原图 tile
                bands = []
                for i in range(3):
                    band = dataset.GetRasterBand(i + 1)
                    arr = band.ReadAsArray(x, y, read_w, read_h)
                    bands.append(arr)
                tile = np.stack(bands, axis=2)

                # 填充到 tile_size
                if tile.shape[0] < tile_size or tile.shape[1] < tile_size:
                    pad_tile = np.zeros((tile_size, tile_size, 3), dtype=np.uint8)
                    pad_tile[:read_h, :read_w] = tile
                    tile = pad_tile

                # 推理
                pred_full = inference_segmentor(model, tile)[0]
                # 只取有效区域
                pred = pred_full[:read_h, :read_w]

                # 写入输出 mask
                out_band.WriteArray(pred, xoff=x, yoff=y)

                # 打印 tile 进度
                if index % 10 == 0:  # 每10个tile打印一次进度
                    progress = (index / total) * 100
                    print(
                        f"[滑窗] 处理第 {index}/{total} 个tile ({progress:.1f}%), 位置: x:{x}-{x + read_w}, y:{y}-{y + read_h}")

        print(f"大图推理完成，结果保存到: {output_path}")
        out_band.FlushCache()
        out_dataset = None
        dataset = None

        return geo_transform, projection

    def create_result_thumbnail(
            self,
            original_tif,
            mask_tif,
            scale=0.03):

        ds_img = gdal.Open(
            original_tif
        )

        ds_mask = gdal.Open(
            mask_tif
        )

        w = ds_img.RasterXSize
        h = ds_img.RasterYSize

        tw = int(w * scale)
        th = int(h * scale)

        bands = []

        for i in range(3):
            arr = ds_img.GetRasterBand(
                i + 1
            ).ReadAsArray(
                buf_xsize=tw,
                buf_ysize=th
            )

            bands.append(arr)

        img = np.stack(
            bands,
            axis=2
        )

        mask = ds_mask.GetRasterBand(
            1
        ).ReadAsArray(
            buf_xsize=tw,
            buf_ysize=th
        )

        color = np.zeros_like(
            img
        )

        color[mask == 1] = [
            0,
            0,
            255
        ]

        result = (
                img * 0.6 +
                color * 0.4
        ).astype(
            np.uint8
        )

        ds_img = None
        ds_mask = None

        return result

    def read_mask_from_geotiff(self, mask_tif_path):
        """从GeoTIFF文件中读取mask数据"""
        dataset = gdal.Open(mask_tif_path)
        if dataset is None:
            raise ValueError(f"无法打开GeoTIFF文件: {mask_tif_path}")

        mask = dataset.GetRasterBand(1).ReadAsArray()
        dataset = None

        return mask


class FastAutoSaveThread(QThread):
    """快速后台自动保存线程"""
    progress_update = pyqtSignal(int, str)
    finished_signal = pyqtSignal(str)  # 传递完成信息

    def __init__(self, geo_segmenter, pred_mask, geo_transform, projection,
                 original_file_path, save_geotiff=True, save_shp=True, save_geojson=True,
                 is_large_tif=False, saved_mask_path=None):
        super().__init__()
        self.geo_segmenter = geo_segmenter
        self.pred_mask = pred_mask
        self.geo_transform = geo_transform
        self.projection = projection
        self.original_file_path = original_file_path
        self.save_geotiff = save_geotiff
        self.save_shp = save_shp  # 是否保存SHP
        self.save_geojson = save_geojson  # 是否保存GeoJSON
        self.is_large_tif = is_large_tif
        self.saved_mask_path = saved_mask_path
        self.running = True

    def run(self):
        try:
            print(f"\n=== 开始快速自动保存数据格式 ===")
            print(f"原始文件: {self.original_file_path}")
            print(f"文件类型: {'大TIF推理结果' if self.is_large_tif else '普通推理结果'}")
            print(f"保存选项: GeoTIFF={self.save_geotiff}, SHP={self.save_shp}, GeoJSON={self.save_geojson}")

            # 如果都不需要保存，直接完成
            if not any([self.save_geotiff, self.save_shp, self.save_geojson]):
                print("没有选择任何保存格式，跳过保存")
                self.progress_update.emit(100, "未选择任何保存格式")
                self.finished_signal.emit("未选择任何保存格式，跳过保存")
                return

            # 获取基础文件名（不含扩展名）
            base_name = os.path.splitext(os.path.basename(self.original_file_path))[0]
            output_dir = os.path.dirname(self.original_file_path)

            # 如果是大TIF推理且pred_mask为空，则从文件读取
            if self.is_large_tif and self.pred_mask is None:
                print(f"从文件读取预测掩码: {self.saved_mask_path}")
                self.pred_mask = self.geo_segmenter.read_mask_from_geotiff(self.saved_mask_path)

            # 统计信息
            total_pixels = self.pred_mask.size
            snow_pixels = np.sum(self.pred_mask == 1)
            snow_percentage = (snow_pixels / total_pixels) * 100
            print(f"总像素数: {total_pixels:,}")
            print(f"目标像素数: {snow_pixels:,} ({snow_percentage:.2f}%)")

            # 准备保存任务
            tasks = []
            task_names = []

            # 1. GeoTIFF格式
            if self.save_geotiff:
                geotiff_path = os.path.join(output_dir, f"{base_name}_result.tif")
                color_map = np.zeros((2, 3), dtype=np.uint8)
                color_map[0] = [255, 255, 255]
                color_map[1] = [0, 0, 255]
                result_rgb = color_map[self.pred_mask]

                tasks.append(("geotiff",
                              lambda: self.geo_segmenter.array_to_geotiff(result_rgb, geotiff_path, self.geo_transform,
                                                                          self.projection)))
                task_names.append(geotiff_path)

            # 2. SHP格式
            if self.save_shp:
                shp_path = os.path.join(output_dir, f"{base_name}_result.shp")
                tasks.append(("shp", lambda: self.geo_segmenter.fast_mask_to_vector(self.pred_mask, self.geo_transform,
                                                                                    self.projection, shp_path)))
                task_names.append(shp_path)

            # 3. GeoJSON格式
            if self.save_geojson:
                geojson_path = os.path.join(output_dir, f"{base_name}_result.geojson")
                tasks.append(("geojson",
                              lambda: self.geo_segmenter.fast_mask_to_geojson(self.pred_mask, self.geo_transform,
                                                                              self.projection, geojson_path)))
                task_names.append(geojson_path)

            if tasks:
                print(f"开始并行处理 {len(tasks)} 个保存任务...")

                # 使用线程池并行执行
                with ThreadPoolExecutor(max_workers=min(len(tasks), 3)) as executor:
                    future_to_task = {executor.submit(task[1]): task[0] for task in tasks}

                    completed = 0
                    total_tasks = len(tasks)
                    for future in as_completed(future_to_task):
                        task_type = future_to_task[future]
                        try:
                            future.result()
                            completed += 1
                            # 计算进度：从20%到100%，根据完成的任务数分配
                            progress = 20 + int((completed / total_tasks) * 80)  # 20-100%之间
                            self.progress_update.emit(progress, f"已完成 {task_type.upper()} 保存")
                            print(f"✓ {task_type.upper()} 保存完成")
                        except Exception as e:
                            print(f"✗ {task_type.upper()} 保存失败: {str(e)}")

            if self.running:
                self.progress_update.emit(100, "快速自动保存完成")

                # 构建完成信息
                saved_files = []
                if self.save_geotiff:
                    saved_files.append(f"- GeoTIFF: {geotiff_path}")
                if self.save_shp:
                    saved_files.append(f"- SHP: {shp_path}")
                if self.save_geojson:
                    saved_files.append(f"- GeoJSON: {geojson_path}")

                print(f"=== 快速自动保存完成 ===")
                print(f"保存的文件:")
                for f in saved_files:
                    print(f"  {f}")

                if saved_files:
                    finish_msg = f"快速自动保存完成:\n" + "\n".join(saved_files)
                else:
                    finish_msg = "未选择任何保存格式"

                self.finished_signal.emit(finish_msg)
            else:
                self.finished_signal.emit("快速自动保存被取消")

        except Exception as e:
            error_msg = f"快速自动保存失败: {str(e)}"
            print(f"❌ {error_msg}")
            self.finished_signal.emit(error_msg)


def load_mmseg_model(config_path, checkpoint_path, device='cuda:0'):
    """使用 MMsegmentation API 加载模型"""
    model = init_segmentor(config_path, checkpoint_path, device=device)
    return model


def predict_image_with_coords(image_path, model, device='cuda:0'):
    """处理带坐标的图像分割"""
    geo_segmenter = FastGeoSegmenter()

    if image_path.lower().endswith(('.tif', '.tiff')):
        print(f"读取GeoTIFF文件: {image_path}")
        image_array, geo_transform, projection, img_size = geo_segmenter.read_geotiff(image_path)
        print(f"图像尺寸: {img_size}")

        temp_img_path = os.path.join(geo_segmenter.temp_dir, "temp_input.jpg")
        cv2.imwrite(temp_img_path, cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR))

        print(f"开始推理...")
        start_time = time.time()
        result = inference_segmentor(model, temp_img_path)
        pred_mask = result[0]
        inference_time = time.time() - start_time
        print(f"推理完成，耗时: {inference_time:.2f}s")

        if os.path.exists(temp_img_path):
            os.remove(temp_img_path)

        return pred_mask, geo_transform, projection, image_array
    else:
        print(f"读取图像: {image_path}")
        start_time = time.time()
        result = inference_segmentor(model, image_path)
        pred_mask = result[0]
        inference_time = time.time() - start_time
        print(f"推理完成，耗时: {inference_time:.2f}s")

        img = cv2.imread(image_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        return pred_mask, None, None, img_rgb


class SegFormerGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("积雪/水体区域识别")
        self.setGeometry(100, 100, 1400, 950)  # 增加高度以容纳新控件
        self.config_paths = {
            "积雪配置": "./my_config_snow.py",  # 默认配置
            "水体配置": "./my_config_water.py"  # 第二个配置
        }

        if not HAS_GEO_LIBS:
            QMessageBox.warning(self, "警告",
                                "缺少地理空间处理库！\n功能受限。\n\n"
                                "请安装: conda install -c conda-forge gdal geopandas rasterio shapely fiona -y")

        self.geo_segmenter = FastGeoSegmenter()

        # 模型配置
        self.config_path = "./my_config_snow.py"
        self.checkpoint_path = "work_dirs/segformer.b0.snow/iter_10000.pth"
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        # 模型字典，用于快速切换
        self.models = {}
        self.current_model = None
        self.current_config = None
        self.current_checkpoint = None

        # 检查默认模型是否存在
        if not os.path.exists(self.config_path):
            QMessageBox.warning(self, "错误", f"配置文件不存在: {self.config_path}")
        if not os.path.exists(self.checkpoint_path):
            QMessageBox.warning(self, "错误", f"模型文件不存在: {self.checkpoint_path}")

        self.pred_mask = None
        self.geo_transform = None
        self.projection = None
        self.original_image = None
        self.current_file_path = None
        self.saved_mask_path = None  # 添加这个变量来保存mask文件路径
        self.original_geo_info = None  # 保存原始图像的地理信息
        self.auto_save_thread = None  # 保存自动保存线程
        self.init_ui()

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        # 标题
        title = QLabel("积雪/水体区域识别")
        title.setAlignment(Qt.AlignCenter)
        title.setFont(QFont("Arial", 16, QFont.Bold))
        layout.addWidget(title)

        # 模型选择区域
        model_group = QGroupBox("模型管理")
        model_layout = QHBoxLayout()
        # 在 model_layout 中添加
        self.config_combo = QComboBox()
        for config_name in self.config_paths.keys():
            self.config_combo.addItem(config_name)
        model_layout.addWidget(QLabel("选择配置:"))
        model_layout.addWidget(self.config_combo)
        self.model_combo = QComboBox()
        self.model_combo.addItem("默认模型 (segformer.b0.snow)",
                                 {"config": self.config_paths["积雪配置"], "checkpoint": self.checkpoint_path})

        self.btn_add_model = QPushButton("添加模型")
        self.btn_load_model = QPushButton("加载选中模型")
        self.btn_refresh_models = QPushButton("刷新模型列表")

        self.btn_add_model.clicked.connect(self.add_model)
        self.btn_load_model.clicked.connect(self.load_selected_model)
        self.btn_refresh_models.clicked.connect(self.refresh_model_list)

        model_layout.addWidget(QLabel("选择模型:"))
        model_layout.addWidget(self.model_combo)
        model_layout.addWidget(self.btn_add_model)
        model_layout.addWidget(self.btn_load_model)
        model_layout.addWidget(self.btn_refresh_models)
        model_group.setLayout(model_layout)
        layout.addWidget(model_group)

        # 自动保存选项区域 - 修改为三个独立的复选框
        save_group = QGroupBox("自动保存设置")
        save_layout = QHBoxLayout()

        self.save_geotiff_checkbox = QCheckBox("保存GeoTIFF格式")
        self.save_geotiff_checkbox.setChecked(True)  # 默认勾选

        self.save_shp_checkbox = QCheckBox("保存SHP格式 (最耗时)")
        self.save_shp_checkbox.setChecked(True)  # 默认勾选

        self.save_geojson_checkbox = QCheckBox("保存GeoJSON格式")
        self.save_geojson_checkbox.setChecked(True)  # 默认勾选

        save_layout.addWidget(self.save_geotiff_checkbox)
        save_layout.addWidget(self.save_shp_checkbox)
        save_layout.addWidget(self.save_geojson_checkbox)
        save_group.setLayout(save_layout)
        layout.addWidget(save_group)

        # 主要操作按钮
        btn_layout = QHBoxLayout()
        self.btn_load = QPushButton("选择图像并推理")
        self.btn_big_predict = QPushButton("超大GeoTIFF推理")
        self.btn_big_predict.clicked.connect(
            self.large_tif_predict
        )

        btn_layout.addWidget(
            self.btn_big_predict
        )
        self.btn_export_shp = QPushButton("导出SHP矢量")
        self.btn_export_geotiff = QPushButton("导出GeoTIFF")
        self.btn_export_geojson = QPushButton("导出GeoJSON")

        self.btn_load.clicked.connect(self.load_and_predict)
        self.btn_export_shp.clicked.connect(self.export_shp)
        self.btn_export_geotiff.clicked.connect(self.export_geotiff)
        self.btn_export_geojson.clicked.connect(self.export_geojson)

        btn_layout.addWidget(self.btn_load)
        btn_layout.addWidget(self.btn_big_predict)
        btn_layout.addWidget(self.btn_export_shp)
        btn_layout.addWidget(self.btn_export_geotiff)
        btn_layout.addWidget(self.btn_export_geojson)
        layout.addLayout(btn_layout)

        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        # 图像显示区域
        img_layout = QHBoxLayout()
        self.label_left = QLabel("原图")
        self.label_right = QLabel("推理结果")
        self.label_left.setAlignment(Qt.AlignCenter)
        self.label_right.setAlignment(Qt.AlignCenter)
        self.label_left.setMinimumSize(600, 500)
        self.label_right.setMinimumSize(600, 500)
        self.label_left.setStyleSheet("border: 1px solid #ccc;")
        self.label_right.setStyleSheet("border: 1px solid #ccc;")
        img_layout.addWidget(self.label_left)
        img_layout.addWidget(self.label_right)
        layout.addLayout(img_layout)

        # 信息标签
        self.info_label = QLabel("就绪")
        self.info_label.setAlignment(Qt.AlignLeft)
        layout.addWidget(self.info_label)

        self.statusBar().showMessage("就绪 - 请先选择并加载模型")

    def add_model(self):
        """添加新模型"""
        config_path, _ = QFileDialog.getOpenFileName(
            self, "选择配置文件", "", "Config Files (*.py)"
        )
        if not config_path:
            return

        checkpoint_path, _ = QFileDialog.getOpenFileName(
            self, "选择模型权重", "", "Checkpoint Files (*.pth *.pt)"
        )
        if not checkpoint_path:
            return
        # 在 add_model 方法中
        selected_config = self.config_combo.currentText()
        config_path = self.config_paths[selected_config]
        model_name = os.path.basename(checkpoint_path).split('.')[0]
        self.model_combo.addItem(f"{model_name}",
                                 {"config": config_path, "checkpoint": checkpoint_path})

        QMessageBox.information(self, "成功", f"模型已添加: {model_name}")

    def refresh_model_list(self):
        """刷新模型列表"""
        self.model_combo.clear()

        # 定义允许的配置文件名
        allowed_configs = ['my_config_snow.py', 'my_config_water.py']  # 你的两个配置文件

        # 递归查找所有 .py 和 .pth 文件
        for root, dirs, files in os.walk("."):
            for file in files:
                if file.endswith('.py') and file in allowed_configs:  # 只处理指定的配置文件
                    config_path = os.path.join(root, file)
                    # 查找同目录下的 .pth 文件
                    pth_files = [f for f in files if f.endswith('.pth')]
                    if pth_files:
                        for pth_file in pth_files:
                            checkpoint_path = os.path.join(root, pth_file)
                            model_name = os.path.basename(pth_file).split('.')[0]
                            self.model_combo.addItem(f"{model_name} ({pth_file}) [{file}]",
                                                     {"config": config_path, "checkpoint": checkpoint_path})

    def load_selected_model(self):
        """加载选中的模型"""
        current_data = self.model_combo.currentData()
        if not current_data:
            QMessageBox.warning(self, "警告", "请选择一个模型！")
            return

        config_path = current_data["config"]
        checkpoint_path = current_data["checkpoint"]

        if not os.path.exists(config_path):
            QMessageBox.warning(self, "错误", f"配置文件不存在: {config_path}")
            return
        if not os.path.exists(checkpoint_path):
            QMessageBox.warning(self, "错误", f"模型文件不存在: {checkpoint_path}")
            return

        try:
            self.progress_bar.setVisible(True)
            self.progress_bar.setValue(0)
            self.statusBar().showMessage("正在加载模型...")

            self.current_model = load_mmseg_model(config_path, checkpoint_path, self.device)
            self.current_config = config_path
            self.current_checkpoint = checkpoint_path

            self.progress_bar.setValue(100)
            self.statusBar().showMessage(f"模型已加载: {os.path.basename(checkpoint_path)}")
            QMessageBox.information(self, "成功", f"模型加载成功！\n{os.path.basename(checkpoint_path)}")

        except Exception as e:
            QMessageBox.critical(self, "错误", f"模型加载失败：\n{str(e)[:200]}...")
            self.statusBar().showMessage("模型加载失败")
        finally:
            self.progress_bar.setVisible(False)

    def load_and_predict(self):
        if not self.current_model:
            QMessageBox.warning(self, "警告", "请先选择并加载模型！")
            return

        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择图像", "", "Image Files (*.jpg *.jpeg *.png *.bmp *.tif *.tiff)"
        )
        if not file_path:
            return

        try:
            self.progress_bar.setVisible(True)
            self.progress_bar.setValue(0)
            self.statusBar().showMessage("正在推理...")

            result = predict_image_with_coords(file_path, self.current_model, self.device)

            if len(result) == 4:
                self.pred_mask, self.geo_transform, self.projection, self.original_image = result
                self.current_file_path = file_path
                if self.geo_transform is not None:
                    self.info_label.setText(f"文件: {os.path.basename(file_path)} | 坐标系统: 已获取")
                else:
                    self.info_label.setText(f"文件: {os.path.basename(file_path)} | 坐标系统: 无")
            else:
                self.pred_mask, _, _, self.original_image = result + (None, None)
                self.info_label.setText(f"文件: {os.path.basename(file_path)} | 坐标系统: 无")

            self.progress_bar.setValue(70)

            h, w, ch = self.original_image.shape
            bytes_per_line = ch * w
            q_img = QImage(self.original_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_img)
            self.label_left.setPixmap(pixmap.scaled(
                self.label_left.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
            ))

            color_map = np.zeros((2, 3), dtype=np.uint8)
            color_map[0] = [255, 255, 255]
            color_map[1] = [0, 0, 255]

            pred_color = color_map[self.pred_mask].astype(np.uint8)
            overlay = (self.original_image * 0.6 + pred_color * 0.4).astype(np.uint8)

            h, w, ch = overlay.shape
            bytes_per_line = ch * w
            q_img_out = QImage(overlay.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pixmap_out = QPixmap.fromImage(q_img_out)
            self.label_right.setPixmap(pixmap_out.scaled(
                self.label_right.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
            ))

            self.progress_bar.setValue(90)
            self.statusBar().showMessage(f"推理完成：{file_path}")

            # 统计积雪面积
            snow_pixels = np.sum(self.pred_mask == 1)
            total_pixels = self.pred_mask.size
            snow_percentage = (snow_pixels / total_pixels) * 100

            # 获取用户的保存选项
            save_geotiff = self.save_geotiff_checkbox.isChecked()
            save_shp = self.save_shp_checkbox.isChecked()
            save_geojson = self.save_geojson_checkbox.isChecked()

            # 检查是否有任何格式被选中
            if not any([save_geotiff, save_shp, save_geojson]):
                QMessageBox.information(self, "完成",
                                        f"推理完成！\n"
                                        f"模型: {os.path.basename(self.current_checkpoint)}\n"
                                        f"积雪像素数: {snow_pixels:,}\n"
                                        f"总面积占比: {snow_percentage:.2f}%\n"
                                        f"坐标系统: {'有' if self.geo_transform is not None else '无'}\n"
                                        f"注意: 您没有选择任何保存格式")
                return

            # 启动后台自动保存线程
            self.start_fast_auto_save_thread(file_path, save_geotiff, save_shp, save_geojson)

            # 构建保存信息
            save_info = []
            if save_geotiff:
                save_info.append("GeoTIFF")
            if save_shp:
                save_info.append("SHP")
            if save_geojson:
                save_info.append("GeoJSON")

            QMessageBox.information(self, "完成",
                                    f"推理完成！\n"
                                    f"模型: {os.path.basename(self.current_checkpoint)}\n"
                                    f"积雪像素数: {snow_pixels:,}\n"
                                    f"总面积占比: {snow_percentage:.2f}%\n"
                                    f"坐标系统: {'有' if self.geo_transform is not None else '无'}\n"
                                    f"正在后台快速保存: {', '.join(save_info)} 格式...")

        except Exception as e:
            QMessageBox.critical(self, "错误", f"推理失败：\n{str(e)[:200]}...")
            self.statusBar().showMessage("推理失败")
        finally:
            self.progress_bar.setValue(100)

    def start_fast_auto_save_thread(self, original_file_path, save_geotiff=True, save_shp=True, save_geojson=True):
        """启动快速后台自动保存线程"""
        if not HAS_GEO_LIBS:
            QMessageBox.warning(self, "警告",
                                "缺少地理空间处理库！\n无法自动保存地理空间格式。\n请安装: conda install -c conda-forge gdal geopandas rasterio shapely fiona -y")
            return

        # 如果都不需要保存，直接返回
        if not any([save_geotiff, save_shp, save_geojson]):
            print("没有选择任何保存格式，跳过保存")
            return

        # 如果已有自动保存线程在运行，先停止它
        if self.auto_save_thread and self.auto_save_thread.isRunning():
            self.auto_save_thread.terminate()
            self.auto_save_thread.wait()

        # 创建新的快速自动保存线程
        self.auto_save_thread = FastAutoSaveThread(
            self.geo_segmenter,
            self.pred_mask,
            self.geo_transform,
            self.projection,
            original_file_path,
            save_geotiff=save_geotiff,
            save_shp=save_shp,
            save_geojson=save_geojson,
            is_large_tif=False
        )

        # 连接信号
        self.auto_save_thread.progress_update.connect(self.on_auto_save_progress)
        self.auto_save_thread.finished_signal.connect(self.on_auto_save_finished)

        # 启动线程
        self.auto_save_thread.start()

    def on_auto_save_progress(self, value, message):
        """自动保存进度更新"""
        self.progress_bar.setValue(value)
        self.statusBar().showMessage(message)

    def on_auto_save_finished(self, message):
        """自动保存完成"""
        self.statusBar().showMessage(message)
        QMessageBox.information(self, "快速自动保存完成", message)

    def export_shp(self):
        if self.pred_mask is None and self.saved_mask_path is None:
            QMessageBox.warning(self, "警告", "请先进行推理！")
            return

        if not HAS_GEO_LIBS:
            QMessageBox.warning(self, "警告",
                                "缺少地理空间处理库！\n请安装: conda install -c conda-forge gdal geopandas rasterio shapely fiona -y")
            return

        # 如果pred_mask为空，但saved_mask_path存在，说明是超大图推理的结果
        if self.pred_mask is None and self.saved_mask_path is not None:
            # 从保存的GeoTIFF文件中读取mask
            try:
                self.pred_mask = self.geo_segmenter.read_mask_from_geotiff(self.saved_mask_path)
            except Exception as e:
                QMessageBox.critical(self, "错误", f"无法读取mask文件：{str(e)}")
                return

        # 检查地理信息是否可用
        if self.geo_transform is None or self.projection is None:
            if self.current_file_path and os.path.exists(self.current_file_path):
                # 尝试从原始文件获取地理信息
                try:
                    _, self.geo_transform, self.projection, _ = self.geo_segmenter.read_geotiff(self.current_file_path)
                except Exception as e:
                    QMessageBox.warning(self, "警告",
                                        f"无法获取地理信息：{str(e)}\n当前文件不是GeoTIFF格式，无法导出带坐标的矢量文件！")
                    return
            else:
                QMessageBox.warning(self, "警告", "当前文件不是GeoTIFF格式，无法导出带坐标的矢量文件！")
                return

        save_path, _ = QFileDialog.getSaveFileName(
            self, "保存SHP文件", "", "Shape Files (*.shp)"
        )
        if not save_path:
            return

        try:
            self.progress_bar.setVisible(True)
            self.progress_bar.setValue(0)
            self.statusBar().showMessage("正在导出矢量文件...")

            # 使用快速方法
            self.geo_segmenter.fast_mask_to_vector(self.pred_mask, self.geo_transform, self.projection, save_path)

            self.progress_bar.setValue(100)
            self.statusBar().showMessage(f"矢量文件已保存: {save_path}")
            QMessageBox.information(self, "完成", f"SHP文件已保存到:\n{save_path}")

        except Exception as e:
            QMessageBox.critical(self, "错误", f"导出失败：\n{str(e)[:200]}...")
        finally:
            self.progress_bar.setVisible(False)

    def export_geotiff(self):
        if self.pred_mask is None and self.saved_mask_path is None:
            QMessageBox.warning(self, "警告", "请先进行推理！")
            return

        if not HAS_GEO_LIBS:
            QMessageBox.warning(self, "警告",
                                "缺少地理空间处理库！\n请安装: conda install -c conda-forge gdal geopandas rasterio shapely fiona -y")
            return

        # 如果pred_mask为空，但saved_mask_path存在，说明是超大图推理的结果
        if self.pred_mask is None and self.saved_mask_path is not None:
            # 从保存的GeoTIFF文件中读取mask
            try:
                self.pred_mask = self.geo_segmenter.read_mask_from_geotiff(self.saved_mask_path)
            except Exception as e:
                QMessageBox.critical(self, "错误", f"无法读取mask文件：{str(e)}")
                return

        # 检查地理信息是否可用
        if self.geo_transform is None or self.projection is None:
            if self.current_file_path and os.path.exists(self.current_file_path):
                # 尝试从原始文件获取地理信息
                try:
                    _, self.geo_transform, self.projection, _ = self.geo_segmenter.read_geotiff(self.current_file_path)
                except Exception as e:
                    QMessageBox.warning(self, "警告",
                                        f"无法获取地理信息：{str(e)}\n当前文件不是GeoTIFF格式，无法导出带坐标的GeoTIFF！")
                    return
            else:
                QMessageBox.warning(self, "警告", "当前文件不是GeoTIFF格式，无法导出带坐标的GeoTIFF！")
                return

        save_path, _ = QFileDialog.getSaveFileName(
            self, "保存GeoTIFF文件", "", "GeoTIFF Files (*.tif *.tiff)"
        )
        if not save_path:
            return

        try:
            self.progress_bar.setVisible(True)
            self.progress_bar.setValue(0)
            self.statusBar().showMessage("正在导出GeoTIFF文件...")

            color_map = np.zeros((2, 3), dtype=np.uint8)
            color_map[0] = [255, 255, 255]
            color_map[1] = [0, 0, 255]
            result_rgb = color_map[self.pred_mask]

            self.geo_segmenter.array_to_geotiff(result_rgb, save_path, self.geo_transform, self.projection)

            self.progress_bar.setValue(100)
            self.statusBar().showMessage(f"GeoTIFF文件已保存: {save_path}")
            QMessageBox.information(self, "完成", f"GeoTIFF文件已保存到:\n{save_path}")

        except Exception as e:
            QMessageBox.critical(self, "错误", f"导出失败：\n{str(e)[:200]}...")
        finally:
            self.progress_bar.setVisible(False)

    def export_geojson(self):
        """导出GeoJSON格式"""
        if self.pred_mask is None and self.saved_mask_path is None:
            QMessageBox.warning(self, "警告", "请先进行推理！")
            return

        if not HAS_GEO_LIBS:
            QMessageBox.warning(self, "警告",
                                "缺少地理空间处理库！\n请安装: conda install -c conda-forge gdal geopandas rasterio shapely fiona -y")
            return

        # 如果pred_mask为空，但saved_mask_path存在，说明是超大图推理的结果
        if self.pred_mask is None and self.saved_mask_path is not None:
            # 从保存的GeoTIFF文件中读取mask
            try:
                self.pred_mask = self.geo_segmenter.read_mask_from_geotiff(self.saved_mask_path)
            except Exception as e:
                QMessageBox.critical(self, "错误", f"无法读取mask文件：{str(e)}")
                return

        # 检查地理信息是否可用
        if self.geo_transform is None or self.projection is None:
            if self.current_file_path and os.path.exists(self.current_file_path):
                # 尝试从原始文件获取地理信息
                try:
                    _, self.geo_transform, self.projection, _ = self.geo_segmenter.read_geotiff(self.current_file_path)
                except Exception as e:
                    QMessageBox.warning(self, "警告",
                                        f"无法获取地理信息：{str(e)}\n当前文件不是GeoTIFF格式，无法导出带坐标的GeoJSON！")
                    return
            else:
                QMessageBox.warning(self, "警告", "当前文件不是GeoTIFF格式，无法导出带坐标的GeoJSON！")
                return

        save_path, _ = QFileDialog.getSaveFileName(
            self, "保存GeoJSON文件", "", "GeoJSON Files (*.geojson)"
        )
        if not save_path:
            return

        try:
            self.progress_bar.setVisible(True)
            self.progress_bar.setValue(0)
            self.statusBar().showMessage("正在导出GeoJSON文件...")

            # 使用快速方法
            self.geo_segmenter.fast_mask_to_geojson(self.pred_mask, self.geo_transform, self.projection, save_path)

            self.progress_bar.setValue(100)
            self.statusBar().showMessage(f"GeoJSON文件已保存: {save_path}")
            QMessageBox.information(self, "完成", f"GeoJSON文件已保存到:\n{save_path}")

        except Exception as e:
            QMessageBox.critical(self, "错误", f"导出失败：\n{str(e)[:200]}...")
        finally:
            self.progress_bar.setVisible(False)

    def large_tif_predict(self):

        if not self.current_model:
            QMessageBox.warning(
                self,
                "警告",
                "先加载模型"
            )

            return

        file_path, _ = \
            QFileDialog.getOpenFileName(
                self,
                "选择GeoTIFF",
                "",
                "*.tif *.tiff"
            )

        if not file_path:
            return

        try:

            self.progress_bar.setVisible(
                True
            )

            self.statusBar().showMessage(
                "超大图推理中..."
            )

            out_path = os.path.join(
                os.path.dirname(
                    file_path
                ),
                "predict_mask.tif"
            )

            geo, \
                proj = \
                self.geo_segmenter.sliding_predict_and_save(
                    file_path,
                    self.current_model,
                    out_path,
                    tile_size=1024,
                    overlap=128
                )

            # 保存地理信息
            self.geo_transform = geo
            self.projection = proj
            self.saved_mask_path = out_path
            self.current_file_path = file_path

            # 读取原始图像用于显示原图
            original_thumb = self.geo_segmenter.create_result_thumbnail(
                file_path,
                file_path,  # 显示原图，而不是结果图
                scale=0.03
            )

            # 显示原图缩略图
            h, w, c = original_thumb.shape
            qimg_orig = QImage(
                original_thumb.data,
                w,
                h,
                w * c,
                QImage.Format_RGB888
            )
            pix_orig = QPixmap.fromImage(
                qimg_orig
            )
            self.label_left.setPixmap(
                pix_orig.scaled(
                    self.label_left.size(),
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation
                )
            )

            # 显示结果图缩略图
            show = \
                self.geo_segmenter.create_result_thumbnail(
                    file_path,
                    out_path,
                    scale=0.03
                )

            h, w, c = show.shape

            qimg = QImage(
                show.data,
                w,
                h,
                w * c,
                QImage.Format_RGB888
            )

            pix = QPixmap.fromImage(
                qimg
            )

            self.label_right.setPixmap(
                pix.scaled(
                    self.label_right.size(),
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation
                )
            )

            # 加载预测掩码到内存以便后续导出使用
            self.pred_mask = self.geo_segmenter.read_mask_from_geotiff(out_path)

            # 获取用户的保存选项
            save_geotiff = self.save_geotiff_checkbox.isChecked()
            save_shp = self.save_shp_checkbox.isChecked()
            save_geojson = self.save_geojson_checkbox.isChecked()

            # 检查是否有任何格式被选中
            if not any([save_geotiff, save_shp, save_geojson]):
                QMessageBox.information(
                    self,
                    "完成",
                    f"结果已保存：\n{out_path}\n\n"
                    f"注意: 您没有选择任何保存格式"
                )
                return

            # 启动后台自动保存线程
            self.start_fast_auto_save_large_tif_thread(file_path, save_geotiff, save_shp, save_geojson)

            # 构建保存信息
            save_info = []
            if save_geotiff:
                save_info.append("GeoTIFF")
            if save_shp:
                save_info.append("SHP")
            if save_geojson:
                save_info.append("GeoJSON")

            QMessageBox.information(
                self,
                "完成",
                f"结果已保存：\n{out_path}\n\n"
                f"正在后台快速保存: {', '.join(save_info)} 格式..."
            )

        except Exception as e:

            QMessageBox.critical(
                self,
                "错误",
                str(e)
            )

        finally:

            self.progress_bar.setVisible(
                False
            )

    def start_fast_auto_save_large_tif_thread(self, original_file_path, save_geotiff=True, save_shp=True,
                                              save_geojson=True):
        """启动大TIF推理的快速后台自动保存线程"""
        if not HAS_GEO_LIBS:
            QMessageBox.warning(self, "警告",
                                "缺少地理空间处理库！\n无法自动保存地理空间格式。\n请安装: conda install -c conda-forge gdal geopandas rasterio shapely fiona -y")
            return

        # 如果都不需要保存，直接返回
        if not any([save_geotiff, save_shp, save_geojson]):
            print("没有选择任何保存格式，跳过保存")
            return

        # 如果已有自动保存线程在运行，先停止它
        if self.auto_save_thread and self.auto_save_thread.isRunning():
            self.auto_save_thread.terminate()
            self.auto_save_thread.wait()

        # 创建新的快速自动保存线程
        self.auto_save_thread = FastAutoSaveThread(
            self.geo_segmenter,
            self.pred_mask,
            self.geo_transform,
            self.projection,
            original_file_path,
            save_geotiff=save_geotiff,
            save_shp=save_shp,
            save_geojson=save_geojson,
            is_large_tif=True,
            saved_mask_path=self.saved_mask_path
        )

        # 连接信号
        self.auto_save_thread.progress_update.connect(self.on_auto_save_progress)
        self.auto_save_thread.finished_signal.connect(self.on_auto_save_finished)

        # 启动线程
        self.auto_save_thread.start()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = SegFormerGUI()
    window.show()
    sys.exit(app.exec_())