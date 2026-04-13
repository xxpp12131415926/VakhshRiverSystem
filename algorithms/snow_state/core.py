from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List


DEFAULT_BBOX = [70.0, 36.0, 76.5, 40.0]
DEFAULT_TASK_PREFIX = "Pamir_Snow_State"
DEFAULT_DRIVE_FOLDER = "Pamir_Snow_System_Output"

DEFAULT_SOURCES = {
    "dem_source": "USGS/SRTMGL1_003",
    "eco_source": "RESOLVE/ECOREGIONS/2017",
    "opt_source": "COPERNICUS/S2_SR_HARMONIZED",
    "lst_source": "MODIS/061/MOD11A1",
    "sar_source": "COPERNICUS/S1_GRD",
}

STATE_LABELS = {
    1: "无雪/裸地",
    2: "稳定干雪（严寒冻结）",
    3: "蓄力消融区（暖雪）",
    4: "活跃消融区（湿雪出水）",
}


def _load_ee():
    try:
        import ee
    except ImportError as exc:
        raise RuntimeError(
            "未安装 earthengine-api，无法运行积雪状态识别模块。"
            "请先执行: pip install earthengine-api"
        ) from exc
    return ee


def validate_bbox_coords(bbox_coords: Iterable[float]) -> List[float]:
    coords = [float(value) for value in bbox_coords]
    if len(coords) != 4:
        raise ValueError("范围坐标必须包含 4 个值: west,south,east,north")

    west, south, east, north = coords
    if west >= east:
        raise ValueError("范围坐标不合法: west 必须小于 east")
    if south >= north:
        raise ValueError("范围坐标不合法: south 必须小于 north")

    return coords


def parse_bbox_text(bbox_text: str) -> List[float]:
    parts = [part.strip() for part in bbox_text.split(",") if part.strip()]
    if len(parts) != 4:
        raise ValueError("请输入 4 个逗号分隔的范围值，例如: 70.0, 36.0, 76.5, 40.0")
    return validate_bbox_coords(parts)


def ensure_earth_engine(authenticate: bool = False) -> str:
    ee = _load_ee()
    try:
        ee.Initialize()
        return "Google Earth Engine 已初始化。"
    except Exception as exc:
        if not authenticate:
            raise RuntimeError(
                "Google Earth Engine 尚未初始化。"
                "请先点击“初始化 GEE”完成认证，或在命令行中先执行 Earth Engine 登录。"
            ) from exc

    try:
        ee.Authenticate()
        ee.Initialize()
    except Exception as auth_exc:
        raise RuntimeError(
            "Google Earth Engine 认证失败，请检查网络、账号权限和本机认证配置。"
        ) from auth_exc

    return "Google Earth Engine 认证并初始化成功。"


def generate_snow_state(
    target_start: str,
    target_end: str,
    ref_start: str = "2022-07-15",
    ref_end: str = "2022-08-15",
    bbox_coords: List[float] | None = None,
    dem_source: str = DEFAULT_SOURCES["dem_source"],
    eco_source: str = DEFAULT_SOURCES["eco_source"],
    opt_source: str = DEFAULT_SOURCES["opt_source"],
    lst_source: str = DEFAULT_SOURCES["lst_source"],
    sar_source: str = DEFAULT_SOURCES["sar_source"],
) -> Any:
    ee = _load_ee()
    safe_bbox = validate_bbox_coords(bbox_coords or DEFAULT_BBOX)

    safe_bbox_geometry = ee.Geometry.Rectangle(safe_bbox)

    ecoregions = ee.FeatureCollection(eco_source)
    eco_boundary = ecoregions.filter(
        ee.Filter.eq("ECO_NAME", "Pamir alpine desert and tundra")
    )
    eco_image = ee.Image.constant(0).paint(eco_boundary, 1)

    dem = ee.Image(dem_source)
    high_elevation = dem.gte(3000).clip(safe_bbox_geometry)

    pamir_raster_mask = eco_image.Or(high_elevation).clip(safe_bbox_geometry)

    local_dem = dem.updateMask(pamir_raster_mask).clip(safe_bbox_geometry)
    slope = ee.Terrain.slope(local_dem)
    aspect = ee.Terrain.aspect(local_dem)

    is_valid_terrain = slope.lt(45)
    is_sunny_slope = aspect.gt(90).And(aspect.lt(270))
    is_shady_slope = is_sunny_slope.Not()

    def add_ndsi(image):
        return image.addBands(
            image.normalizedDifference(["B3", "B11"]).rename("NDSI")
        )

    target_ndsi = (
        ee.ImageCollection(opt_source)
        .filterBounds(safe_bbox_geometry)
        .filterDate(target_start, target_end)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 50))
        .map(add_ndsi)
        .select("NDSI")
        .median()
        .updateMask(pamir_raster_mask)
        .clip(safe_bbox_geometry)
    )

    is_snow_covered = target_ndsi.gte(0.6)

    target_lst = (
        ee.ImageCollection(lst_source)
        .filterBounds(safe_bbox_geometry)
        .filterDate(target_start, target_end)
        .select("LST_Day_1km")
        .mean()
        .multiply(0.02)
        .subtract(273.15)
        .updateMask(pamir_raster_mask)
        .clip(safe_bbox_geometry)
        .resample("bilinear")
    )

    is_warm = target_lst.gte(0)

    s1 = (
        ee.ImageCollection(sar_source)
        .filterBounds(safe_bbox_geometry)
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
        .filter(ee.Filter.eq("instrumentMode", "IW"))
    )

    def to_linear(image):
        return ee.Image(10).pow(image.divide(10))

    def to_db(image):
        return ee.Image(10).multiply(image.log10())

    def get_stable_sar(start_date, end_date):
        collection = s1.filterDate(start_date, end_date)
        linear_mean = collection.map(to_linear).mean()
        db_mean = to_db(linear_mean)
        return (
            db_mean.select("VV")
            .add(db_mean.select("VH"))
            .divide(2)
            .updateMask(pamir_raster_mask)
            .clip(safe_bbox_geometry)
        )

    ref_sar = get_stable_sar(ref_start, ref_end)
    target_sar = get_stable_sar(target_start, target_end)

    initial_wet_snow = (
        target_sar.subtract(ref_sar)
        .lt(-6)
        .focal_median(radius=1.5, kernelType="circle", units="pixels")
        .And(is_valid_terrain)
    )

    mean_elev_sunny_dict = local_dem.updateMask(
        initial_wet_snow.And(is_sunny_slope)
    ).reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=safe_bbox_geometry,
        scale=500,
        maxPixels=1e13,
        tileScale=16,
    )
    val_sunny = mean_elev_sunny_dict.get("elevation")
    mean_elev_sunny = ee.Number(
        ee.Algorithms.If(
            ee.Algorithms.IsEqual(val_sunny, None),
            8000,
            val_sunny,
        )
    )

    mean_elev_shady_dict = local_dem.updateMask(
        initial_wet_snow.And(is_shady_slope)
    ).reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=safe_bbox_geometry,
        scale=500,
        maxPixels=1e13,
        tileScale=16,
    )
    val_shady = mean_elev_shady_dict.get("elevation")
    mean_elev_shady = ee.Number(
        ee.Algorithms.If(
            ee.Algorithms.IsEqual(val_shady, None),
            8000,
            val_shady,
        )
    )

    force_wet_sunny = (
        is_snow_covered.And(is_warm)
        .And(is_sunny_slope)
        .And(local_dem.lt(ee.Image.constant(mean_elev_sunny)))
    )
    force_wet_shady = (
        is_snow_covered.And(is_warm)
        .And(is_shady_slope)
        .And(local_dem.lt(ee.Image.constant(mean_elev_shady)))
    )
    final_wet_snow = initial_wet_snow.Or(force_wet_sunny).Or(force_wet_shady)

    state_image = ee.Image(0).updateMask(pamir_raster_mask).clip(safe_bbox_geometry)
    state_image = state_image.where(is_snow_covered.Not(), 1)
    state_image = state_image.where(
        is_snow_covered.And(final_wet_snow.Not()).And(is_warm.Not()),
        2,
    )
    state_image = state_image.where(
        is_snow_covered.And(final_wet_snow.Not()).And(is_warm),
        3,
    )
    state_image = state_image.where(final_wet_snow, 4)
    state_image = state_image.updateMask(target_ndsi.mask())

    return state_image


def _build_task_name(task_prefix: str, target_start: str, target_end: str) -> str:
    raw_prefix = task_prefix.strip() or DEFAULT_TASK_PREFIX
    safe_prefix = re.sub(r"[^A-Za-z0-9_-]+", "_", raw_prefix).strip("_")
    safe_prefix = safe_prefix or DEFAULT_TASK_PREFIX
    start_token = target_start.replace("-", "")
    end_token = target_end.replace("-", "")
    return f"{safe_prefix}_{start_token}_{end_token}"[:100]


def submit_snow_state_export(
    target_start: str,
    target_end: str,
    ref_start: str = "2022-07-15",
    ref_end: str = "2022-08-15",
    bbox_coords: List[float] | None = None,
    drive_folder: str = DEFAULT_DRIVE_FOLDER,
    task_prefix: str = DEFAULT_TASK_PREFIX,
    scale: int = 30,
    authenticate: bool = False,
    dem_source: str = DEFAULT_SOURCES["dem_source"],
    eco_source: str = DEFAULT_SOURCES["eco_source"],
    opt_source: str = DEFAULT_SOURCES["opt_source"],
    lst_source: str = DEFAULT_SOURCES["lst_source"],
    sar_source: str = DEFAULT_SOURCES["sar_source"],
) -> Dict[str, Any]:
    ee = _load_ee()
    ensure_earth_engine(authenticate=authenticate)

    safe_bbox = validate_bbox_coords(bbox_coords or DEFAULT_BBOX)
    safe_drive_folder = drive_folder.strip() or DEFAULT_DRIVE_FOLDER
    if scale <= 0:
        raise ValueError("导出分辨率必须为正整数")

    state_image = generate_snow_state(
        target_start=target_start,
        target_end=target_end,
        ref_start=ref_start,
        ref_end=ref_end,
        bbox_coords=safe_bbox,
        dem_source=dem_source,
        eco_source=eco_source,
        opt_source=opt_source,
        lst_source=lst_source,
        sar_source=sar_source,
    )

    description = _build_task_name(task_prefix, target_start, target_end)
    export_task = ee.batch.Export.image.toDrive(
        image=state_image,
        description=description,
        folder=safe_drive_folder,
        scale=scale,
        region=ee.Geometry.Rectangle(safe_bbox).getInfo()["coordinates"],
        maxPixels=1e13,
        fileFormat="GeoTIFF",
        formatOptions={"cloudOptimized": True},
    )
    export_task.start()

    status = {}
    try:
        status = export_task.status()
    except Exception:
        status = {}

    return {
        "description": description,
        "drive_folder": safe_drive_folder,
        "scale": scale,
        "bbox": safe_bbox,
        "task_id": status.get("id"),
        "task_state": status.get("state", "SUBMITTED"),
        "legend": STATE_LABELS.copy(),
    }
