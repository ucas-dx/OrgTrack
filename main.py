# -*- coding: utf-8 -*-
# @Time : 2023/8/30 10:22
# @Author : Dengxun
# @Email : 38694034@qq.com
# @File : tiff_read2mask.py
# @Project : tracking

import skimage
from PIL import Image
import tifffile
from torch.utils.data import Dataset
from skimage import io
from skimage.util import img_as_ubyte
import imageio
from torchvision import transforms
from tqdm import tqdm
import sys
from pathlib import Path

import MACPNet
from tracking.AnalyzeTracking import DrawAndHighlight
sys.path.append(str(Path(".").resolve()))
import tracking.tools as outtool
from skimage.transform import resize
from tracking import Track, Inverse, Overlap
imageio.plugins.freeimage.download()
import  torch.nn as nn
import torch
import numpy as np
import copy
device = torch.device("cpu")
import torch
import random
import os
from MACPNet import MACP
from skimage.transform import warp, SimilarityTransform
from tracking.ImageHandling import LoadPILImages, GetFrames
seed=123
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
class MultiFrameTIFFDataset(Dataset):
    def __init__(self, root_dir, image_transform=None):
        self.root_dir = root_dir
        self.image_transform = image_transform
        self.image_filenames = sorted(os.listdir(self.root_dir))

    def __len__(self):
        return len(self.image_filenames)
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir, self.image_filenames[idx])
        if not os.path.exists(img_name):
            raise FileNotFoundError(f"Image file not found: {img_name}")

        image = io.imread(img_name)
        image = img_as_ubyte(image)
        transformed_frames = []
        for frame in image:
            pil_frame = transforms.ToPILImage()(frame)
            pil_frame = transforms.Resize((512, 512))(pil_frame)
            pil_frame = transforms.ToTensor()(pil_frame)
            transformed_frames.append(pil_frame)

        image_tensor =torch.stack(transformed_frames, dim=0)#转换列表为张量
        if self.image_transform:
            image_tensor = self.image_transform(image_tensor)

        return image_tensor,self.image_filenames


image_transform = transforms.Compose([
    #transforms.Normalize(0.40081004856666735,0.0877802242064718),
    lambda x: torch.as_tensor(x, dtype=torch.float32)
])

timelapse_dataset= MultiFrameTIFFDataset(r'tiff', image_transform=image_transform)#延时图像路径
timelapse_loader = torch.utils.data.DataLoader(timelapse_dataset, 1, shuffle=False)#延时数据集加载
# drug_timelapse_dataset=MultiFrameTIFFDataset(r'D:\OrganoID-master\Publication\Figure2\2d-e\BF', image_transform=image_transform)
# drug_timelapse_loader = torch.utils.data.DataLoader(drug_timelapse_dataset, 1, shuffle=False)
def time_lapse(test_data, name, show_label=True, model_name="att_swin"):
    #测试数据
    model=nn.Sequential()
    if  model_name=="MACP":
        model = MACPNet.MACP.MVCnet()
        model = model.to(device)
        model.load_state_dict(torch.load(r'MACPNet\MACP.pth', map_location='cpu'))
    model.eval()
    with torch.no_grad():
        imge = []
        for batch,image_name in tqdm(test_data):
            images = batch[0].to(device)
            outputs = model(images)
            outimg = copy.copy(outputs)
            outimg = np.squeeze(outimg, axis=1)  # 降维
            for t in outimg:
                outimg1 = torch.sigmoid(t).cpu().numpy()
                imge.append(outimg1)
    if show_label==True:
        def img2bin(imge):
            newimg = []
            for i ,tempt in enumerate(imge) :
                # 示例输入张量
                input_tensor = tempt
                np_input=np.array(input_tensor)
                #边缘检测，hysteresisMaximum=0.05,hysteresisMinimum=0.005用于分离的高低阈值参数
                output_tensor_edge=(outtool.DetectEdges(np.expand_dims(np_input,axis=0),gaussianSigma=2.0,hysteresisMaximum=0.05,hysteresisMinimum=0.005,foregroundThreshold=0.5)).astype('int8')
                #分离，阈值foregroundThreshold=0.5
                output_tensor_seperate=outtool.SeparateContours(edges=output_tensor_edge,images=np.expand_dims(np_input,axis=0),foregroundThreshold=0.5,gaussianSigma=2)
                #填孔，过滤尺寸较小的类器官minimumArea=50
                output_tensor=outtool.Cleanup(output_tensor_seperate,minimumArea=50,fillHoles=True,removeBorders=False)
                # 遍历每张图像
                newimg.append(output_tensor[0])
            newimg = np.stack(newimg, axis=0)
            return newimg
        tailuo = np.expand_dims(img2bin(imge),axis=1)
        return tailuo,image_name
# 保存输出的多帧TIFF图像
def save_as_tiff_images(images, output_folder):
    for i, image in enumerate(images):
        tiff_image = Image.fromarray(image.transpose(1, 2, 0))  # 将通道维度移到最后
        tiff_path = f"{output_folder}/frame_{i}.tiff"
        tiff_image.save(tiff_path)
# 将图像大小调整为 1024x1022
def resize_images(images, target_shape):
    resized_images = []
    for image in images:
        resized_image = resize(image, target_shape, mode='constant', anti_aliasing=True)
        resized_images.append(resized_image)
    return resized_images
# 使用更准确的上采样方法将图像大小调整为 target_shape
def upsample_images(images, target_shape):
    upsampled_images = []
    for image in images:
        image=image[0]
        transform = SimilarityTransform(scale=target_shape[0] / image.shape[0])
        upsampled_image = np.expand_dims(warp(image, transform, output_shape=target_shape, mode='reflect'),axis=0)
        upsampled_images.append(upsampled_image)

    return np.stack(upsampled_images,axis=0)

def timelapse_label(cleanedImages,img_name):
    stack = Track(np.squeeze(cleanedImages,axis=1), 1, Inverse(Overlap), trackLostCutoff=10)#trackLostCutoff为消失的代价
    out = stack  # sahpe:(8, 1, 512, 512)
    output_folder = img_name.split('.')[0] + "_label.tiff" # 替换为输出文件夹路径
    tifffile.imwrite(output_folder, out)
    originalImages = [np.asarray(i) / 255 for i in
                      GetFrames(LoadPILImages(
                          Path(r"tiff/Timelapse12hr.tiff"))[0])]  # 原始图像路径
    compressedImages = []

    for image in originalImages:
        # 使用 PIL.Image 对象来进行尺寸压缩
        pilimage = Image.fromarray(image)
        compressed_image = pilimage.resize((512, 512), resample=Image.Resampling.BICUBIC)
        # 将压缩后的图像转换回 NumPy 数组
        compressed_np_array = np.array(compressed_image)
        compressedImages.append(compressed_np_array)
    labeledImages = [np.asarray(i) for i in
                     GetFrames(
                         LoadPILImages(Path(r"Timelapse12hr_label.tiff"))[
                             0])]  # 分割mask路径

    squareMicronsPerSquarePixel = 1.31 ** 2
    areasFile = open(r"Areas.csv", "w+")  # 保存唯一ID类器官的面积变化
    areasFile.write("Time,ID,Area")
    for i, image in enumerate(labeledImages):
        for rp in skimage.measure.regionprops(image):
            label = rp.label
            area = rp.area * squareMicronsPerSquarePixel
            areasFile.write("\n%d,%d,%f" % (i * 2, label, area))
    areasFile.close()
    # mask覆盖颜色，对特定的标号类器官使用不同颜色的标签，方便追踪可视化
    labelsToHighlight = {1: (0, 154, 222),
                         2: (255, 198, 30),
                         10: (175, 88, 186),
                         33: (0, 205, 108)}

    originalImages = compressedImages
    # 可视化不同时间帧中的类器官ID
    DrawAndHighlight(labeledImages[0], originalImages[0], labelsToHighlight,
                     Path(r"Timelapse_0.png"))
    DrawAndHighlight(labeledImages[1], originalImages[1], labelsToHighlight,
                     Path(r"Timelapse_1.png"))
    DrawAndHighlight(labeledImages[2], originalImages[2], labelsToHighlight,
                     Path(r"Timelapse_2.png"))


if __name__=="__main__":
    mode = ["MACP"]
    for mode_name in mode:
        cleanedImages,img_name=time_lapse(timelapse_loader, name=mode_name + 'time', show_label=True, model_name=mode_name)
    timelapse_label(cleanedImages,img_name[0][0])





