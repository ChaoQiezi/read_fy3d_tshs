# @Author   : ChaoQiezi
# @Time     : 2024/3/24  11:49
# @FileName : FY3D_hdf2tiff.py
# @Email    : chaoqiezi.one@qq.com

"""
This script is used to 用于将FY3D TSHS产品中的MWTS_Ch_BT(MWTS 通道亮温) 和MWHS_Ch_BT(MWHS 通道亮温)两个数据集
进行重投影并输出为TIFF文件

数据说明:
FY3D TSHS 的二级产品（TSHS是MWTS/MWHS融合的产品），大气垂直探测产品（AVP）,HDF, 33KMFY3D TSHS 的二级产品
(TSHS是MWTS/MWHS融合的产品），大气垂直探测产品（AVP）,HDF, 33KM)

投影说明:
FY3D卫星是极轨卫星,此数据集中的一幅影像是一个全球的区域,但极轨卫星无法一次获取全球的区域，因此实际上其是经过多次
扫描(一次扫描, 南北极纵向绕一圈, 但很显然不可能得到全球区域),这也就是为什么数据集的维数看起来似乎有一点奇怪:
    对于亮度温度数据集, 其维数为: [Nscans,90,13],其中13指代波段数, Nscans是指扫描次数, 每次扫描都是卫星仪器覆盖地球某一区域的单次观测,
    而整个数据集记录了仪器十五个通道在每条扫描线上 90 个观测象元的亮温
可以参考：Ref/极轨卫星_扫描_类似.mp4, 其扫描完整个地球一次进行数据整合处理后类似 Ref/结果.png

需要进行重投影, 由于像元位置并非按照真实地理位置进行排列,例如对于亮度温度数据集的维数[1212, 90](1212为扫描次数),其1212并非X轴坐标,90
也并非Y轴坐标.

"""
import os.path
from glob import glob
import h5py
import numpy as np
from osgeo import gdal
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from scipy.ndimage import zoom, generic_filter


def read_h5(hdf_path, ds_path, scale=True):
    """
    读取指定数据集并进行预处理
    :param hdf_path: 数据集所在HDF文件的绝对路径
    :param ds_path: 数据集在HDF文件内的路径
    :return: 返回处理好的数据集
    """

    with h5py.File(hdf_path) as f:
        # 读取目标数据集属矩阵和相关属性
        ds = f[ds_path]
        ds_values = np.float32(ds[:])  # 获取数据集
        valid_range = ds.attrs['valid_range']  # 获取有效范围
        slope = ds.attrs['Slope'][0]  # 获取斜率(类似scale_factor)
        intercept = ds.attrs['Intercept'][0]  # 获取截距(类似add_offset)

        """"
        原始数据集可能存在缩放(可能是为存储空间全部存为整数(需要通过斜率和截距再还原为原来的值,或者是需要进行单位换算甚至物理量的计算例如
        最常见的DN值转大气层表观反射率等(这多出现于一级产品的辐射定标, 但二级产品可能因为单位换算等也可能出现));
        如果原始数据集不存在缩放, 那么Slope属性和Intercept属性分别为1和0;
        这里为了确保所有迭代的HDF文件的数据集均正常得到, 这里依旧进行slope和intercept的读取和计算(尽管可能冗余)
        """

    # 目标数据集的预处理(无效值, 范围限定等)
    ds_values[(ds_values < valid_range[0]) | (ds_values > valid_range[1])] = np.nan
    if scale:
        ds_values = ds_values * slope + intercept  # 还原缩放
        """
        Note: 这里之所以选择是否进行缩放, 原因为经纬度数据集中的slope为1, intercept也为1, 但是进行缩放后超出地理范围1°即出现了90.928
        对于纬度。其它类似, 因此认为这里可能存在问题如果进行缩放, 所以对于经纬度数据集这里不进行缩放"""

    return ds_values


# 输出TIF
def write_tiff(out_path, dataset, transform, nodata=np.nan):
    """
    输出TIFF文件
    :param out_path: 输出文件的路径
    :param dataset: 待输出的数据
    :param transform: 坐标转换信息(形式:[左上角经度, 经度分辨率, 旋转角度, 左上角纬度, 旋转角度, 纬度分辨率])
    :param nodata: 无效值
    :return: None
    """

    # 创建文件
    driver = gdal.GetDriverByName('GTiff')
    out_ds = driver.Create(out_path, dataset[0].shape[1], dataset[0].shape[0], len(dataset), gdal.GDT_Float32)

    # 设置基本信息
    out_ds.SetGeoTransform(transform)
    out_ds.SetProjection('WGS84')

    # 写入数据
    for i in range(len(dataset)):
        out_ds.GetRasterBand(i + 1).WriteArray(dataset[i])  # GetRasterBand()传入的索引从1开始, 而非0
        out_ds.GetRasterBand(i + 1).SetNoDataValue(nodata)
    out_ds.FlushCache()


def data_glt(out_path, src_ds, src_x, src_y, out_res, zoom_scale=6, glt_range=None, windows_size=9):
    """
    基于经纬度数据集对目标数据集进行GLT校正/重投影(WGS84), 并输出为TIFF文件
    :param out_path: 输出tiff文件的路径
    :param src_ds: 目标数据集
    :param src_x: 对应的横轴坐标系(对应地理坐标系的经度数据集)
    :param src_y: 对应的纵轴坐标系(对应地理坐标系的纬度数据集)
    :param out_res: 输出分辨率(单位: 度/°)
    :param zoom_scale:
    :return: None
    """
    if glt_range:
        # lon_min, lat_max, lon_max, lat_min = -180.0, 90.0, 180.0, -90.0
        lon_min, lat_max, lon_max, lat_min = glt_range
    else:
        lon_min, lat_max, lon_max, lat_min = np.nanmin(src_x), np.nanmax(src_y), \
            np.nanmax(src_x), np.nanmin(src_y)

    zoom_lon = zoom(src_x, (zoom_scale, zoom_scale), order=0)  # 0为最近邻插值
    zoom_lat = zoom(src_y, (zoom_scale, zoom_scale), order=0)
    # # 确保插值结果正常
    # zoom_lon[(zoom_lon < -180) | (zoom_lon > 180)] = np.nan
    # zoom_lat[(zoom_lat < -90) | (zoom_lat > 90)] = np.nan
    glt_cols = np.ceil((lon_max - lon_min) / out_res).astype(int)
    glt_rows = np.ceil((lat_max - lat_min) / out_res).astype(int)

    deal_bands = []
    for src_ds_band in src_ds:
        glt_ds = np.full((glt_rows, glt_cols), np.nan)
        glt_lon = np.full((glt_rows, glt_cols), np.nan)
        glt_lat = np.full((glt_rows, glt_cols), np.nan)
        geo_x_ix, geo_y_ix = np.floor((zoom_lon - lon_min) / out_res).astype(int), \
            np.floor((lat_max - zoom_lat) / out_res).astype(int)
        glt_lon[geo_y_ix, geo_x_ix] = zoom_lon
        glt_lat[geo_y_ix, geo_x_ix] = zoom_lat
        glt_x_ix, glt_y_ix = np.floor((src_x - lon_min) / out_res).astype(int), \
            np.floor((lat_max - src_y) / out_res).astype(int)
        glt_ds[glt_y_ix, glt_x_ix] = src_ds_band
        # write_tiff('H:\\Datasets\\Objects\\ReadFY3D\\Output\\py_lon.tiff', [glt_lon],
        #            [lon_min, out_res, 0, lat_max, 0, -out_res])
        # write_tiff('H:\\Datasets\\Objects\\ReadFY3D\\Output\\py_lat.tiff', [glt_lat],
        #            [lon_min, out_res, 0, lat_max, 0, -out_res])

        # # 插值
        # interpolation_ds = np.full_like(glt_ds, fill_value=np.nan)
        # jump_size = windows_size // 2
        # for row_ix in range(jump_size, glt_rows - jump_size):
        #     for col_ix in range(jump_size, glt_cols - jump_size):
        #         if ~np.isnan(glt_ds[row_ix, col_ix]):
        #             interpolation_ds[row_ix, col_ix] = glt_ds[row_ix, col_ix]
        #             continue
        #         # 定义当前窗口的边界
        #         row_start = row_ix - jump_size
        #         row_end = row_ix + jump_size + 1  # +1 因为切片不包含结束索引
        #         col_start = col_ix - jump_size
        #         col_end = col_ix + jump_size + 1
        #         rows, cols = np.ogrid[row_start:row_end, col_start:col_end]
        #         distances = np.sqrt((rows - row_ix) ** 2 + (cols - col_ix) ** 2)
        #         window_ds = glt_ds[(row_ix - jump_size):(row_ix + jump_size + 1),
        #                     (col_ix - jump_size):(col_ix + jump_size + 1)]
        #         if np.sum(~np.isnan(window_ds)) == 0:
        #             continue
        #         distances_sort_pos = np.argsort(distances.flatten())
        #         window_ds_sort = window_ds[np.unravel_index(distances_sort_pos, distances.shape)]
        #         interpolation_ds[row_ix, col_ix] = window_ds_sort[~np.isnan(window_ds_sort)][0]

        deal_bands.append(glt_ds)
        # print('处理波段: {}'.format(len(deal_bands)))
        # if len(deal_bands) == 6:
        #     break
    write_tiff(out_path, deal_bands, [lon_min, out_res, 0, lat_max, 0, -out_res])
    # write_tiff('H:\\Datasets\\Objects\\ReadFY3D\\Output\\py_underlying.tiff', [interpolation_ds], [lon_min, out_res, 0, lat_max, 0, -out_res])
    # write_tiff('H:\\Datasets\\Objects\\ReadFY3D\\Output\\py_lon.tiff', [glt_lon], [x_min, out_res, 0, y_max, 0, -out_res])
    # write_tiff('H:\\Datasets\\Objects\\ReadFY3D\\Output\\py_lat.tiff', [glt_lat], [x_min, out_res, 0, y_max, 0, -out_res])


def reform_ds(ds, lon, lat, reform_range=None):
    """
    重组数组
    :param ds: 目标数据集(三维)
    :param lon: 对应目标数据集的经度数据集()
    :param lat: 对应目标数据集的纬度数据集(二维)
    :param reform_range: 重组范围, (lon_min, lat_max, lon_max, lat_min), 若无则使用全部数据
    :return: 以元组形式依次返回: 重组好的目标数据集, 经度数据集, 纬度数据集(均为二维数组)
    """

    # 裁选指定地理范围的数据集
    if reform_range:
        lon_min, lat_max, lon_max, lat_min = reform_range
        x, y = np.where((lon > lon_min) & (lon < lon_max) & (lat > lat_min) & (lat < lat_max))
        ds = ds[:, x, y]
        lon = lon[x, y]
        lat = lat[x, y]
    else:
        ds = ds.reshape(ds.shape[0], -1)
        lon = lon.flatten()
        lat = lat.flatten()

    # 无效值去除(去除地理位置为无效值的元素)
    valid_pos = ~np.isnan(lat) & ~np.isnan(lon)
    ds = ds[:, valid_pos]
    lon = lon[valid_pos]
    lat = lat[valid_pos]

    # 重组数组的初始化
    bands = []
    for band in ds:
        reform_ds_size = np.int32(np.sqrt(band.size))  # int向下取整
        band = band[:reform_ds_size ** 2].reshape(reform_ds_size, reform_ds_size)
        bands.append(band)
    else:
        lon = lon[:reform_ds_size ** 2].reshape(reform_ds_size, reform_ds_size)
        lat = lat[:reform_ds_size ** 2].reshape(reform_ds_size, reform_ds_size)
    bands = np.array(bands)
    return bands, lon, lat


# 准备
in_dir = r'H:\Datasets\Objects\ReadFY3D\fy3d'
out_dir = r'H:\Datasets\Objects\ReadFY3D\Output'
out_res = 0.3
glt_range = [-180.0, 90.0, 180.0, -90.0]  # 重投影区域,全球

# 读取目标数据集和经纬度数据集
hdf_paths = glob(os.path.join(in_dir, 'FY3D_TSHSX*.HDF'))
for hdf_path in hdf_paths:
    # hdf_path = 'H:\\Datasets\\Objects\\ReadFY3D\\fy3d\\FY3D_TSHSX_ORBT_L2_AVP_MLT_NUL_20200502_0220_033KM_MS.HDF'
    # hdf_path = 'H:\\Datasets\\Objects\\ReadFY3D\\fy3d\\FY3D_TSHSX_ORBT_L2_AVP_MLT_NUL_20200501_0240_033KM_MS (1).HDF'
    date_str = '_'.join(os.path.basename(hdf_path).split('_')[7:9])
    out_mwhs_bt_path = os.path.join(out_dir, 'FY3D_TSHSX_mwhs_bt_{}.tiff'.format(date_str))
    out_mwts_bt_path = os.path.join(out_dir, 'FY3D_TSHSX_mwts_bt_{}.tiff'.format(date_str))
    mwhs_bt = read_h5(hdf_path, 'DATA/MWHS_Ch_BT')  # shape=(15, 1212, 90)
    mwts_ht = read_h5(hdf_path, 'DATA/MWTS_Ch_BT')
    lon = read_h5(hdf_path, 'GEO/Longitude', scale=False)  # shape=(1212, 90)
    lat = read_h5(hdf_path, 'GEO/Latitude', scale=False)  # shape=(1212, 90)
    # 重投影区域裁定和重组
    reform_mwhs_bt, reform_lon, reform_lat = reform_ds(mwhs_bt, lon, lat)
    reform_mwts_ht, _, _ = reform_ds(mwts_ht, lon, lat)
    # GLT校正
    data_glt(out_mwhs_bt_path, reform_mwhs_bt, reform_lon, reform_lat, out_res, glt_range=glt_range)
    data_glt(out_mwts_bt_path, reform_mwts_ht, reform_lon, reform_lat, out_res, glt_range=glt_range)
    print('处理: {}'.format(os.path.basename(hdf_path)))

