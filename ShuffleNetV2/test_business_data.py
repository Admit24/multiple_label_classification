#!coding=utf-8
# 此脚本使用训好的宽度为1.0的ShuffleNetV2模型测试一批业务数据，并将识别结果存为txt文件

import torch
import torch.nn as nn
import torchvision
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from torchvision import transforms
from torch.utils.data import DataLoader
#from network import ShuffleNetV1
from network_108 import ShuffleNetV2
import os
import subprocess
import math
import numpy as np

def make_label(label_path):
    count = 0
    #with open(label_path,'r',encoding='utf-8') as f:
    with open(label_path,'r') as f:
        dic = []
        for line in f.readlines():
            line = line.strip('\n') # 去掉换行符
            b=line.split(':') #将每一行以冒号为分隔符转换成列表
            del b[1]
            b.append(count)
            tmp = b[1]
            b[1] = b[0]
            b[0] = tmp
            dic.append(b)
            count = count + 1
        dic=dict(dic)
    return dic

device = torch.device('cuda')


# 数据预处理
data_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),  # 把一个取值范围是[0,255]的PIL.Image或者shape为(H,W,C)的numpy.ndarray，
    # 转换成形状为[C,H,W]，取值范围是[0,1]的torch.FloadTensor
    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                         std=[0.5, 0.5, 0.5])  # 给定均值：(R,G,B) 方差：>（R，G，B），将会把Tensor正则化。
    # 即：Normalized_image=(image-mean)/std。
])


# 加载模型
print('load ShuffleNetV2 model begin!')
#model = ShuffleNetV1(group=3)
model = ShuffleNetV2(model_size='1.0x')
#model = ShuffleNetV2()
#model = nn.DataParallel(model)
#model = torch.nn.DataParallel(model)
#checkpoint = torch.load('./model_best_0_5.pth.tar')
#model.load_state_dict(checkpoint['state_dict'])
#model.load_state_dict(torch.load('./checkpoint-029.pth'))
model.load_state_dict(torch.load('./checkpoint-029.pth')['model'])
model.eval()  # 固定batchnorm，dropout等，一定要有
model = model.to(device)
print('load ShuffleNetV2 model done!')

torch.no_grad()

label_path = './labelsmap.txt' # 标签索引和对应的英文名
business_label_result = './business_label_result.txt' # 业务数据标签存放处
image_path = '/mnt/data1/gyy/multilabel/mobilenet/images/' # 业务数据文件夹
dic = make_label(label_path) # 字典dic键为类别索引，值为对应的英文名称
#print(dic[1])
#print(type(dic[1]))

print('test start!')

count = 0 # 31685张正常图片（可以测试的图片），共33345张图片
with open(business_label_result, 'w') as f: # 每次重新打开都要清空里面的内容，所以打开之后一次性把内容写完
    for jpg in os.listdir(image_path):
        path = image_path + jpg
        try:
            img = Image.open(path)
            img = data_transform(img)
            img = img.unsqueeze(0)
            img_= img.to(device)
            output = model(img_)
            pred = torch.sigmoid(output).ge(0.5)
            #print('pred:' + str(pred))
            #print('pred[0]:' + str(pred[0]))
            count = count + 1
            print('current jpg:' + str(count))
            f.write(jpg)
            for i in range(len(pred[0])):
                if pred[0][i]==1:
                    f.write(' ' + dic[i])
            f.write('\n') # 标签信息写完换行
        except Exception as e:
            continue






