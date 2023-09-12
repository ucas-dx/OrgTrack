#!/usr/bin/env python
# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------------
# @Project : AUOrganSeg
# @Time    : 2023/7/11 15:40
# @Author  : Deng xun
# @Email   : 38694034@qq.com
# @File    : tools.py
# @Software: PyCharm 
# -------------------------------------------------------------------------------
import numpy as np
import skimage.feature
import skimage.filters
import skimage.segmentation
import skimage.morphology
import skimage.measure
from .HelperFunctions import printRep


def Label(images: np.ndarray, foregroundThreshold):
    """
    这个函数的基本工作原理是对每个图像进行迭代，找出每个图像中像素值大于或等于给定阈值的部分（即前景），对其进行二值化和形态学开运算，
    然后为这些部分（可能包括多个连通区域）分配不同的标签。在实际应用中，这可以用于图像分割、物体检测和计数等任务。

    """

    # 创建一个与输入图像数组形状相同的全零数组，该数组将被用于存储标签信息
    labeled = np.zeros(images.shape, np.uint16)

    # 遍历每张图像
    for i in range(images.shape[0]):
        # 使用 skimage.morphology.binary_opening 函数找出图像中像素值大于或等于 foregroundThreshold 的区域，该函数会对找到的区域进行二值化处理和形态学开运算
        foregroundMask = skimage.morphology.binary_opening(images[i] >= foregroundThreshold)

        # 使用 skimage.measure.label 函数对前景区域进行标记，其中不同的连通区域会被标记为不同的标签
        labeled[i] = skimage.measure.label(foregroundMask)

    # 返回标记过的图像数组
    return labeled


def SeparateContours(images: np.ndarray, edges: np.ndarray, foregroundThreshold: float, gaussianSigma: float):
    # 打印开始分离轮廓的信息
    print("Separating contours...", end="", flush=True)

    # 创建一个与输入图像数组形状相同的全零数组，该数组将被用于存储分离后的图像
    separatedImages = np.zeros(images.shape, np.uint16)

    # 遍历每张图像
    for i in range(images.shape[0]):
        # 打印处理进度
        printRep(str(i + 1) + "/" + str(images.shape[0]))


        # 找出图像中像素值大于或等于 foregroundThreshold 的区域，二值化和形态学开运算
        foregroundMask = skimage.morphology.binary_opening(images[i] >= foregroundThreshold)

        # 使用高斯滤波平滑图像，然后对平滑后的图像取反作为分水岭算法的高度图
        smoothForeground = skimage.filters.gaussian(images[i], gaussianSigma)
        heightmap = -smoothForeground
        deg=edges[i]
        # 找出前景区域中没有被边缘信息标记的区域，这些区域将被视为各个物体的中心点
        centers = np.bitwise_and(foregroundMask, np.bitwise_not(deg))
        basins = skimage.measure.label(centers)

        # 使用分水岭算法将相互接触的物体分开
        labeled = skimage.segmentation.watershed(heightmap, basins, mask=foregroundMask)

        # 对于在分水岭算法过程中被误分的小物体，我们需要将它们找回来并添加到分离后的图像中
        unsplit = np.logical_and(foregroundMask, labeled == 0)
        unsplit_labeled = skimage.measure.label(unsplit)
        unsplit_labeled[unsplit_labeled > 0] += labeled.max() + 1
        separatedImages[i] = labeled + unsplit_labeled

    # 打印完成分离轮廓的信息
    printRep("Done.")
    printRep(None)

    # 返回分离后的图像数组
    return separatedImages



def Cleanup(images: np.ndarray, minimumArea: int, removeBorders: bool, fillHoles: bool):
    # 创建一个与输入图像形状相同的全零数组，用于存储清理后的图像
    cleanedImages = np.zeros_like(images)

    # 打印开始清理的信息
    print("Cleaning up objects...", end="", flush=True)

    # 遍历每张图像
    for i in range(images.shape[0]):
        # 打印处理进度
        printRep(str(i + 1) + "/" + str(images.shape[0]))
        integer_image = images[i].astype(int)

        # 使用 skimage.measure.regionprops 得到图像中每个区域的属性
        rps = skimage.measure.regionprops(integer_image)

        # 遍历每个区域
        for rp in rps:
            # 如果区域面积小于给定的最小面积，则忽略此区域
            if rp.area < minimumArea:
                continue

            # 如果 removeBorders 为 True 并且区域位于图像的边缘，则忽略此区域
            coords = np.asarray(rp.coords)
            if removeBorders and (0 in coords or
                                  images.shape[1]-1 in coords[:, 0] or
                                  images.shape[2]-1 in coords[:, 1]):
                continue

            # 获取当前区域的边界框
            mir, mic, mar, mac = rp.bbox

            # 如果 fillHoles 为 True，则使用填充空洞后的区域替换原图像中对应的区域；否则，使用原始区域替换
            cleanedImages[i, mir:mar, mic:mac] = np.where(
                rp.image_filled if fillHoles else rp.image, rp.label,
                cleanedImages[i, mir:mar, mic:mac])

    # 打印完成清理的信息
    printRep("Done.")
    printRep(None)

    # 返回清理后的图像
    return cleanedImages


def DetectEdges(images: np.ndarray, gaussianSigma: float,
                hysteresisMinimum: float, hysteresisMaximum: float,
                foregroundThreshold: float):
    # 打印开始检测边缘的信息
    print("Detecting edges...", end="", flush=True)

    # 创建一个与输入图像形状相同的全零数组，用于存储边缘图像
    edgeImages = np.zeros(images.shape, dtype=bool)

    # 遍历每张图像
    for i in range(images.shape[0]):
        # 打印处理进度
        printRep(str(i) + "/" + str(images.shape[0]))

        # 使用 Sobel 滤波器和高斯滤波器检测边缘，然后使用滞后阈值进一步处理检测结果
        smoothEdges = skimage.filters.gaussian(skimage.filters.sobel(images[i]), gaussianSigma)
        edges = skimage.filters.apply_hysteresis_threshold(smoothEdges, hysteresisMinimum, hysteresisMaximum)

        # 找出图像中像素值大于或等于 foregroundThreshold 的区域
        foregroundMask = skimage.morphology.binary_opening(images[i] >= foregroundThreshold)

        # 只保留在前景区域内的边缘
        edgeImages[i, :, :] = np.bitwise_and(edges, foregroundMask)

    # 打印完成检测边缘的信息
    printRep("Done.")
    printRep(None)

    # 返回边缘图像
    return edgeImages

