#!coding=utf-8
# 此脚本用于测试集群上训练好的模型精度和召回率是否都达到80%以上

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
            dic.append(b)
            count = count + 1
        dic=dict(dic)
    return dic

def make_dataset(dir, dic):
    images = []
    for line in open(dir):
        line = line.strip('\n')
        b = line.split(' ') # 以空格为分隔符将一行字符转换为列表
        length = len(b) # 获取列表b的长度
        path = b[0] # 获取图片路径
        label_array = np.zeros(108) # 全0的标签数组
        for i in range(length - 2):
            label_array[dic[b[i + 2]]] = 1 #所属的类别置1
        item = (path, label_array)
        images.append(item)
    return images


def accuracy(output, target):
    acc = recall = 0
    pred = torch.sigmoid(output).ge(0.5).cpu()
    pred = [pred_one.nonzero().reshape(-1).numpy().tolist() for pred_one in pred]
    #print(pred)
    target = target.cpu()
    target = [target_one.nonzero().reshape(-1).numpy().tolist() for target_one in target]
    #print(target)
    #print(len(target))
    for i in range(len(target)):
        if (len(pred[i]) == 0)&(len(pred[i]) == 0):
            acc += 1.0
            recall += 1.0
        else:
            acc += len(set(pred[i])&set(target[i])) / (len(pred[i])+1e-7)
            recall += len(set(pred[i])&set(target[i])) / (len(target[i])+1e-7)
    acc = acc / len(target)
    recall = recall / len(target)
    return torch.tensor(acc).cuda().float(), torch.tensor(recall).cuda().float()


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


label_path = './labelsmap.txt'
test_path = './test.txt'
dic = make_label(label_path)
#print(dic)
images = make_dataset(test_path, dic)
length = len(images)
#print(images[0])
count = 0
acc = 0
recall = 0
print('test start!')
for i in range(length):
    print('count:' + str(count))
    item = images[i]
    path = item[0]
    label = item[1]
    label = label.reshape((1, 108)) # 把标签变成batch_size的形式，即使batch_size为1
    Tensor_label = torch.from_numpy(label) # numpy数组转Tensor
    #print('path:' + path)
    #print('label:' + str(label))
    #print('Tensor_label:' + str(Tensor_label))
    #print(type(path))
    #print(type(label))
    #print(type(Tensor_label))
    try:
        img = Image.open(path)
        img = data_transform(img)
        img = img.unsqueeze(0)
        img_ = img.to(device)
        output = model(img_)
        pred = torch.sigmoid(output).ge(0.5)
        count = count + 1
        print(pred)
        print(Tensor_label)
        acc_tmp, recall_tmp = accuracy(output, Tensor_label)
        print('acc_tmp:' + str(acc_tmp))
        print('recall_tmp:' + str(recall_tmp))
        acc = acc + acc_tmp
        recall = recall + recall_tmp
        print('acc:' + str(acc/count))
        print('recall:' + str(recall/count))
    except Exception as e:
        continue

print('final acc:' + str(acc/count))
print('final recall:' + str(recall/count))




