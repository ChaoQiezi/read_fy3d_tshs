# @Author   : ChaoQiezi
# @Time     : 2024/3/25  23:19
# @FileName : FY3D_mosaic_interp.py
# @Email    : chaoqiezi.one@qq.com

"""
This script is used to 用于对FY3D输出的tiff文件进行镶嵌(合成为每天), 并进行无效值近邻填充
"""

import os.path
from glob import glob
from osgeo import gdal
import time
from typing import Union
import numpy as np
from scipy.ndimage import generic_filter
from numpy import ceil, floor


def img_mosaic(mosaic_paths: list, return_transform: bool = True, mode: str = 'last'):
    """
    该函数用于对列表中的所有TIFF文件进行镶嵌
    :param mosaic_paths: 多个TIFF文件路径组成的字符串列表
    :param return_transform: 是否一同返回仿射变换
    :param mode: 镶嵌模式, 默认是Last(即如果有存在像元重叠, mosaic_paths中靠后影像的像元将覆盖其),
        可选: last, mean, max, min
    :return:
    """

    # 获取镶嵌范围
    x_mins, x_maxs, y_mins, y_maxs = [], [], [], []
    for mosaic_path in mosaic_paths:
        ds = gdal.Open(mosaic_path)
        x_min, x_res, _, y_max, _, y_res_negative = ds.GetGeoTransform()
        x_size, y_size = ds.RasterXSize, ds.RasterYSize
        x_mins.append(x_min)
        x_maxs.append(x_min + x_size * x_res)
        y_mins.append(y_max+ y_size * y_res_negative)
        y_maxs.append(y_max)
    else:
        y_res = -y_res_negative
        band_count = ds.RasterCount
        ds = None
    x_min, x_max, y_min, y_max = min(x_mins), max(x_maxs), min(y_mins), max(y_maxs)

    # 镶嵌
    col = ceil((x_max - x_min) / x_res).astype(int)
    row = ceil((y_max - y_min) / y_res).astype(int)
    mosaic_imgs = []  # 用于存储各个影像
    for ix, mosaic_path in enumerate(mosaic_paths):
        mosaic_img = np.full((band_count, row, col), np.nan)  # 初始化
        ds = gdal.Open(mosaic_path)
        ds_bands = ds.ReadAsArray()
        # 计算当前镶嵌范围
        start_row = floor((y_max - (y_maxs[ix] - x_res / 2)) / y_res).astype(int)
        start_col = floor(((x_mins[ix] + x_res / 2) - x_min) / x_res).astype(int)
        end_row = (start_row + ds_bands.shape[1]).astype(int)
        end_col = (start_col + ds_bands.shape[2]).astype(int)
        mosaic_img[:, start_row:end_row, start_col:end_col] = ds_bands
        mosaic_imgs.append(mosaic_img)

    # 判断镶嵌模式
    if mode == 'last':
        mosaic_img = mosaic_imgs[0].copy()
        for img in mosaic_imgs:
            mask = ~np.isnan(img)
            mosaic_img[mask] = img[mask]
    elif mode == 'mean':
        mosaic_imgs = np.asarray(mosaic_imgs)
        mask = np.isnan(mosaic_imgs)
        mosaic_img = np.ma.array(mosaic_imgs, mask=mask).mean(axis=0).filled(np.nan)
    elif mode == 'max':
        mosaic_imgs = np.asarray(mosaic_imgs)
        mask = np.isnan(mosaic_imgs)
        mosaic_img = np.ma.array(mosaic_imgs, mask=mask).max(axis=0).filled(np.nan)
    elif mode == 'min':
        mosaic_imgs = np.asarray(mosaic_imgs)
        mask = np.isnan(mosaic_imgs)
        mosaic_img = np.ma.array(mosaic_imgs, mask=mask).min(axis=0).filled(np.nan)
    else:
        raise ValueError('不支持的镶嵌模式: {}'.format(mode))

    if return_transform:
        return mosaic_img, [x_min, x_res, 0, y_max, 0, -y_res]

    return mosaic_img



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


def window_interp(arr, distances):
    if np.sum(~np.isnan(arr)) == 0:
        return np.nan
    # 距离最近的有效像元
    arr = arr.flatten()
    arr_sort = arr[np.argsort(distances)]
    if np.sum(~np.isnan(arr_sort)) == 0:
        return np.nan
    else:
        return arr_sort[~np.isnan(arr_sort)][0]



# 准备
in_dir = r'H:\Datasets\Objects\ReadFY3D\Output'  # 存放mwhs_bt和mwts_bt的tiff文件所在目录
out_dir = r'H:\Datasets\Objects\ReadFY3D\mosaic_output'

# 计算距离矩阵
window_size = 9
central_x_ix, central_y_ix = window_size // 2, window_size // 2
xs, ys = np.meshgrid(
    np.arange(0, window_size),
    np.arange(0, window_size)
)
central_x, central_y = xs[central_x_ix, central_y_ix], ys[central_x_ix, central_y_ix]
distances = np.sqrt(np.power(xs - central_x, 2) + np.power(ys - central_y, 2))
distances[distances == 0] = np.nan
distances = distances.flatten()

# 镶嵌和插值
img_mwhs_paths = glob(os.path.join(in_dir, 'FY3D_TSHSX_mwhs_bt*.tiff'))
img_mwts_paths = glob(os.path.join(in_dir, 'FY3D_TSHSX_mwts_bt*.tiff'))
for img_paths in [img_mwhs_paths, img_mwts_paths]:
    img_times = [_name.split('_')[4] for _name in img_paths]
    img_times_unique = np.unique(img_times)
    for img_time in img_times_unique:
        start_time = time.time()
        mosaic_paths = [img_paths[_ix] for _ix, _img_time in enumerate(img_times) if _img_time == img_time]
        mosaic_img, geo_transform = img_mosaic(mosaic_paths, mode='max')
        mosaic_interp_img = np.full_like(mosaic_img, np.nan)
        for ix, img in enumerate(mosaic_img):
            interp_img = generic_filter(img, window_interp, size=window_size, cval=np.nan,
                                        extra_keywords={'distances': distances})
            mosaic_interp_img[ix, :, :] = interp_img
        # 输出
        prefix_name = '_'.join(os.path.basename(mosaic_paths[0]).split('_')[:4])
        out_path = os.path.join(out_dir, prefix_name + '_mosaic_interp_{}.tiff'.format(img_time))
        write_tiff(out_path, mosaic_interp_img, transform=geo_transform)
        print('处理: {}--{}--{} s'.format(prefix_name, img_time, time.time() - start_time))
