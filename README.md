# multiple_label_classification
multiple label classification with MobileNetV2, ShuffleNetV2 and GhostNet



1.多元分类和多标签分类的比较

多元分类是将一个样本分到一个特定的类别当中，而多标签分类是将一个样本分到多个类别当中（一个样本可能属于多个类别），所以损失函数会不一样，其次数据加载也不一样。

（1）损失函数

```python
import torch.nn as nn
loss1 = nn.CrossEntropyLoss() # 多元分类损失函数
loss2 = nn.BCEWithLogitsLoss() # 多标签分类损失函数
```

**多元分类：**

网络最后一层的输出先经过softmax函数，使得每一个元素值映射到一个概率，再对概率取对数，最后将前面的输出与Label对应的那个值取负数就得到了该样本的loss（如果有batch的话要需要取均值，且最后一步实际上就是NLLLOSS函数），将这三步合成一步就形成CrossEntropyloss函数，公式如下：
$$
loss(x,class) = -ln(\frac{exp(x[class])}{\sum_jexp(x[j])})=-x[class]+ln(\sum_jexp(x[j]))
$$
从公式当中可以直接看出来三个函数nn.Softmax()，torch.log()，nn.NLLLoss()的作用顺序，并且loss函数并没有像交叉熵函数那样计算$-[t_iln(y_i)+(1-t_i)ln(1-y_i)]$两项，实际上只取了前一项。

代码验证：

```python
import torch
import torch.nn as nn

# 分布调用函数计算损失
input = torch.randn(3,3)
input
'''
tensor([[-0.9147, -1.8131, -0.6992],
        [-1.1762,  1.1391,  1.7756],
        [-0.1639,  0.2338,  0.0073]])
'''
sm = nn.Softmax(dim=1)
sm(input)
'''
dim=1将每一行作为一个样本来计算，从网络输出映射到概率，默认dim=1
tensor([[0.3777, 0.1538, 0.4685],
        [0.0330, 0.3346, 0.6324],
        [0.2721, 0.4050, 0.3229]])
'''
torch.log(sm(input))
'''
tensor([[-0.9737, -1.8721, -0.7582],
        [-3.4101, -1.0948, -0.4583],
        [-1.3016, -0.9039, -1.1304]])
'''
loss = nn.NLLLoss()
target = torch.tensor([0,2,1])
loss(torch.log(sm(input)), target)
'''
根据标签选择计算的值，第一个样本选第0个概率取负数，为0.9737；第二个样本选第2个概率取负数，为0.4583；第三个样本选择第1个概率值取负数，为0.9039；三者取均值，得到0.7786；
tensor(0.7786)
'''

# 直接调用nn.CrossEntropyLoss()计算损失
loss = nn.CrossEntropyLoss()
loss(torch.log(sm(input)), target)
'''
tensor(0.7786)，与分步计算相同
'''
```



**多标签分类：**

网络最后一层的输出先经过sigmoid函数（网络输出的向量经过sigmoid函数，第$i$个值表示该样本属于第$i$类的概率，此时向量的值加和不为1），再经过交叉熵函数来计算loss，公式如下：
$$
loss(o,t) = -\frac{1}{n}\sum_i(t[i]*ln(o[i])+(1-t[i])*log(1-o[i]))
$$
此时的损失计算函数为nn.BCELoss()里面包括了两项，而且是对sigmoid和CrossEntropyloss函数不一样。

代码验证：

```python
import torch
import torch.nn as nn
import math

# 分步计算损失
input = torch.randn(3,3)
input
'''
tensor([[ 0.0754,  0.3111,  0.7609],
        [-0.6940, -1.2521,  0.5681],
        [-2.0807,  0.0732, -2.3314]])
'''
si = nn.Sigmoid()
si(input)
'''
tensor([[0.5189, 0.5772, 0.6816],
        [0.3331, 0.2223, 0.6383],
        [0.1110, 0.5183, 0.0886]])
'''
# 假设标签如下
target = torch.tensor([[0,1,1],[1,1,1],[0,0,0]])

r11 = 0 * math.log(0.8707) + (1-0) * math.log((1 - 0.8707))
r12 = 1 * math.log(0.7517) + (1-1) * math.log((1 - 0.7517))
r13 = 1 * math.log(0.8162) + (1-1) * math.log((1 - 0.8162))

r21 = 1 * math.log(0.3411) + (1-1) * math.log((1 - 0.3411))
r22 = 1 * math.log(0.4872) + (1-1) * math.log((1 - 0.4872))
r23 = 1 * math.log(0.6815) + (1-1) * math.log((1 - 0.6815))

r31 = 0 * math.log(0.4847) + (1-0) * math.log((1 - 0.4847))
r32 = 0 * math.log(0.6589) + (1-0) * math.log((1 - 0.6589))
r33 = 0 * math.log(0.5273) + (1-0) * math.log((1 - 0.5273))

r1 = -(r11 + r12 + r13) / 3
#0.8447112733378236
r2 = -(r21 + r22 + r23) / 3
#0.7260397266631787
r3 = -(r31 + r32 + r33) / 3
#0.8292933181294807
bceloss = (r1 + r2 + r3) / 3 
print(bceloss)
'''
0.8000147727101611
'''

# 使用BCEloss函数
loss = nn.BCELoss()
print(loss(m(input), target))
'''
tensor(0.8000)
'''

# 使用BCEWithLogitsLoss函数
loss = nn.BCEWithLogitsLoss()
print(loss(input, target))
'''
tensor(0.8000)
'''
```





（2）准确率指标衡量

多元分类：

准确率有top-k的划分，前k类有预测对的都将该样本视为正确分类。代码如下：

```python
def accuracy1(output, target, topk=(1, 5)):
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    
    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res
```



多标签分类：（多标签分类的指标较复杂，可以取简单的一种）

对于每个样本，要预测多个标签，输出值经过sigmoid函数之后设置阈值，使得大于阈值的设为1，小于阈值的设为0，这样就得到一个样本的预测值。对比预测值和标签向量，将同时为1的类别个数记为$N$，将预测值向量中1的个数记为$M_1$，将标签向量中1的个数记为$M_2$。$\frac{N}{M_1}$称为准确率，即模型预测出来的有多少是正确的；$\frac{N}{M_2}$称为召回率，即正确的类别可以召回来多少，用这两个指标来衡量每个epoch的模型效果。

代码如下：

```python
def accuracy(output, target):
    acc = recall = 0
    pred = torch.sigmoid(output).ge(0.5).cpu()
    pred = [pred_one.nonzero().reshape(-1).numpy().tolist() for pred_one in pred]
    target = target.cpu()
    target = [target_one.nonzero().reshape(-1).numpy().tolist() for target_one in target]
    for i in range(len(target)):
        #print(pred[i])
        #print(target[i])
        if len(pred[i]) != 0:
            acc += len(set(pred[i])&set(target[i])) / (len(pred[i])+1e-7)
            recall += len(set(pred[i])&set(target[i])) / (len(target[i])+1e-7)
        acc = acc / len(target)
        recall = recall / len(target)
        return torch.tensor(acc).cuda().float(), torch.tensor(recall).cuda().float()

```

由于输入的output以及target均在gpu上，在使用这两个量来计算准确率和召回率的时候需要先放到cpu上才可以进行numpy或者list操作，在计算结束后再放回gpu上。



测试两个准确度函数：

```pyhthon
import torch
import numpy as np

def accuracy1(output, target, topk=(1, 2)):
    maxk = max(topk)
    print('maxk:' + str(maxk))
    batch_size = target.size(0)
    print('batch_size:' + str(batch_size))
    _, pred = output.topk(maxk, 1, True, True) # 按照维度1的方向，返回前maxk个最大值对应的类别
    print('pred:' + str(pred))
    pred = pred.t()
    print('pred:' + str(pred))
    print('target_view:' + str(target.view(1, -1)))
    print('target_view_expand:' + str(target.view(1, -1).expand_as(pred)))
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    print('correct:' + str(correct))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

'''
def accuracy2(outputs, targets, threshold):
    acc = 0
    recall = 0
    pred = torch.sigmoid(outputs)
    print('pred:' + str(pred))
    pred = pred.ge(threshold).float()
    print('pred:' + str(pred))
    print('targets.view_as(pred):' + str(targets.view_as(pred)))
    print('pred.eq(targets.view_as(pred)):' + str(pred.eq(targets.view_as(pred))))
    return pred.eq(targets.view_as(pred)).float().mean()
'''
def accuracy2(output, target):
    acc = recall = 0
    pred = torch.sigmoid(output).ge(0.5)
    print('pred:' + str(pred))
    print('pred[1].nonzero():' + str(pred[1].nonzero()))
    print('pred[1].nonzero().reshape():' + str(pred[1].nonzero().reshape(-1)))
    pred = [pred_one.nonzero().reshape(-1).numpy().tolist() for pred_one in pred]
    target = [target_one.nonzero().reshape(-1).numpy().tolist() for target_one in target]
    print('pred:' + str(pred))
    print('target:' + str(target))
    for i in range(len(target)):
        #print(pred[i])
        #print(target[i])
        if len(pred[i]) != 0:
            print('set:' + str(set(pred[i])&set(target[i])))
            acc += len(set(pred[i])&set(target[i])) / (len(pred[i]) + 1e-7)
            recall += len(set(pred[i])&set(target[i])) / (len(target[i])+1e-7)
    acc = acc / len(target)
    recall = recall / len(target)
    return torch.tensor(acc).float(), torch.tensor(recall).float()

inputs = torch.tensor([[ 0.0754,  0.3111,  0.7609],
                      [-0.6940, -1.2521,  0.5681],
                      [-2.0807,  0.0732, -2.3314]])

label1 = torch.tensor([2, 2, 0]) # 多元分类并不是one-hot形式，每个样本的标签就是一个数值

label2 = torch.tensor([[0., 0., 1.],
                       [0., 0., 0.],
                       [0., 0., 0.]])

acc1 = accuracy1(inputs, label1, (1, 2))
print('acc:' + str(acc1))

#label2 = label2.float()
acc2, recall = accuracy2(inputs, label2)
print('acc2:' + str(acc2))
print('recall:' + str(recall))

```

多元分类和多标签分类的ImageFolder返回的标签不一样，多元分类的标签每个类别只需要返回一个数，即类别序号；多标签分类是返回一个向量，属于哪个类别即在哪个位置置1。



（3）数据加载

多元分类：

多元分类任务里面，每一张图片属于一个特定的类别，放在一个类别文件当中，然后使用pytorch中的datasets.ImageFolder去读数据集，对数据进行预处理同时返回标签；在多标签分类中，需要重新写ImageFolder函数。

先给出pytorch官方提供的datasets.ImageFolder源代码，pytorch版本为1.0：（https://pytorch.org/docs/1.2.0/_modules/torch/utils/data/dataloader.html#DataLoader）

```python
import torch.utils.data as data

from PIL import Image

import os
import os.path


def has_file_allowed_extension(filename, extensions):
    """Checks if a file is an allowed extension.

    Args:
        filename (string): path to a file

    Returns:
        bool: True if the filename ends with a known image extension
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in extensions)


def find_classes(dir):
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))] # 数据集中的所有文件类别，列表形式
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))} # 类别名字到序号，字典形式
    return classes, class_to_idx

def make_dataset(dir, class_to_idx, extensions):
    images = []
    dir = os.path.expanduser(dir) # 去除路径中的～/
    for target in sorted(os.listdir(dir)): # target为类别名称，如dog
        d = os.path.join(dir, target) # 类别路径，如xx/train/dog/
        if not os.path.isdir(d):
            continue

        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames): # 图片名称排序
                if has_file_allowed_extension(fname, extensions):
                    path = os.path.join(root, fname) # 图片路径，如xx/train/dog/1.jpg
                    item = (path, class_to_idx[target]) # 路径和所属类别的序号
                    images.append(item)

    return images # image是一个列表，元素都是二元组，存储图片和图片所属类别的序号

class DatasetFolder(data.Dataset):
    """A generic data loader where the samples are arranged in this way: ::

        root/class_x/xxx.ext
        root/class_x/xxy.ext
        root/class_x/xxz.ext

        root/class_y/123.ext
        root/class_y/nsdf3.ext
        root/class_y/asd932_.ext

    Args:
        root (string): Root directory path.
        loader (callable): A function to load a sample given its path.
        extensions (list[string]): A list of allowed extensions.
        transform (callable, optional): A function/transform that takes in
            a sample and returns a transformed version.
            E.g, ``transforms.RandomCrop`` for images.
        target_transform (callable, optional): A function/transform that takes
            in the target and transforms it.

     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        samples (list): List of (sample path, class_index) tuples
    """
    def __init__(self, root, loader, extensions, transform=None, target_transform=None):
        classes, class_to_idx = find_classes(root) # classes为列表，存储类别名字；class_to_idx为字典，存储列表名字和对应的序号
        samples = make_dataset(root, class_to_idx, extensions) # sample是一个列表，元素都是二元组，存储图片和图片所属类别的序号
        if len(samples) == 0:
            raise(RuntimeError("Found 0 files in subfolders of: " + root + "\n"
                               "Supported extensions are: " + ",".join(extensions)))

        self.root = root
        self.loader = loader
        self.extensions = extensions

        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples

        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index] # 图片路径和所属的类别的序号
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self):
        return len(self.samples)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str


IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif']


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)

class ImageFolder(DatasetFolder):
    """A generic data loader where the images are arranged in this way: ::

        root/dog/xxx.png
        root/dog/xxy.png
        root/dog/xxz.png

        root/cat/123.png
        root/cat/nsdf3.png
        root/cat/asd932_.png

    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.

     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """
    def __init__(self, root, transform=None, target_transform=None,
                 loader=default_loader):
        super(ImageFolder, self).__init__(root, loader, IMG_EXTENSIONS,
                                          transform=transform,
                                          target_transform=target_transform)
        self.imgs = self.samples
```



多标签分类：

多标签分类一般数据集需要自己整理，只有list文件用于存储类别名称和数据图片的信息，包括url和所属的类别，处理步骤如下：

1）拆分数据集为训练集和测试集

由于所有的数据都存在一个list文件里面，先读取list，再将所有图片分为训练集和测试集两部分，前350000张为测试集，剩下的接近100000张为测试集，给每张图片加上前缀路径分别存储为train.txt和test.txt。list文件格式和划分代码如下：

```python
scenery/00001002.jpg 7 scenery outdoor clouds sun tree sunset sky
selfie/selfie_part1/018DD09A-7D10-4FA3-230F-31E882A2C77020170215_00.jpg 3 person selfie outdoor
nus-wide/dataset/un_building/0348_2274244083.jpg 1 outdoor
nus-wide/dataset/throwing/0060_1477493305.jpg 0
anotated_0106/data1/25EAEDD8-38D5-522B-0D90-52A7592E77DC20171226_02.jpg 4 plants person outdoor tree
others/cizhuan/718ffa03-66ad-11e7-bdeb-2c4d544f3e490.jpg 0
nus-wide/dataset/football/0567_2391156357.jpg 7 scenery buildings outdoor clouds sky cityscape road
nus-wide/dataset/white-tailed/0217_1501598065.jpg 5 clouds outdoor sky birds animal
nus-wide/dataset/farms/0502_166859912.jpg 3 horses outdoor animal
......
```



```python
# 此脚本用于将所有图片划分测试集和训练集

#import json

source_path = 'all-train-image-labels-en_labels_shuf.list'
train_path = 'train.txt'
test_path = 'test.txt'
presuffix = '/mnt/data1/sun.zheng/all-train-val/'

count = 0 # 训练集数据个数，总数据量为448446
train_count = 350000 # 设置训练集数量为350000

with open(train_path, 'w') as f:
    for line in open(source_path):
        line = line.strip('\n') #去掉换行符\n
        line = presuffix + line
        f.write(line)
        f.write('\n')
        #print(line)
        #print(type(line)) # str
        count += 1
        print('current train count:' + str(count))
        if count > train_count:
            break

count = 0 # 当前数量重新置0
with open(test_path, 'w') as f:
    for line in open(source_path):
        count += 1
        print('current test count:' + str(count))
        if count > train_count:
            line = line.strip('\n') # 去掉换行符\n
            line = presuffix + line
            f.write(line)
            f.write('\n')
```



2）将类别名称数字化

标签类别文件为英文名称以及中文释义，可以按照从上往下的顺序依次索引为0，1，2，...，并存为label.txt文件，标签list文件格式以及代码如下：

```shell
clouds,云
coral,珊瑚
cow,牛
dog,狗
fire,火
fish,鱼
flowers,花卉
food,食物
fox,狐狸
garden,花园
glacier,冰川
grass,草
harbor,港口
......
```



```python
import numpy as np
count = 0
with open('valid-labels-107.txt','r',encoding='utf-8') as f:
    dic=[]
    for line in f.readlines():
        line=line.strip('\n') #去掉换行符\n
        b=line.split(',') #将每一行以逗号为分隔符转换成列表
        #print(type(b))
        #print(b)
        del b[1] # 删除第二个元素
        b.append(count) # 增加value
        print(b)
        dic.append(b)
        count += 1
    dic=dict(dic)
print(dic)

# 将字典写入txt文件中
with open('label.txt', 'w') as f:
    for key, value in dic.items():
        f.write(key)
        f.write(' ')
        f.write(str(value)) # 此处必须要加str，否则无法写入
        f.write('\n')
```

这里生成的label.txt后续没有使用，但是代码后续会用。



3）重写datasets.ImageFolder源码，使得适应多标签的数据

在数据文件准备好之后，可以开始重写datasets.ImageFolder类，先给出重写之后的代码：

```python
import torch.utils.data as data
import numpy as np
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import os
import os.path

def has_file_allowed_extension(filename, extensions):
    """Checks if a file is an allowed extension.

    Args:
        filename (string): path to a file

    Returns:
        bool: True if the filename ends with a known image extension
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in extensions)


def find_classes(dir):
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))] # 数据集中的所有文件类别，列表形式
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))} # 类别名字到序号，字典形式
    return classes, class_to_idx

'''
def make_dataset(dir, class_to_idx, extensions):
    images = []
    dir = os.path.expanduser(dir) # 去除路径中的～/
    for target in sorted(os.listdir(dir)): # target为类别名称，如dog
        d = os.path.join(dir, target) # 类别路径，如xx/train/dog/
        if not os.path.isdir(d):
            continue

        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames): # 图片名称排序
                if has_file_allowed_extension(fname, extensions):
                    path = os.path.join(root, fname) # 图片路径，如xx/train/dog/1.jpg
                    item = (path, class_to_idx[target]) # 路径和所属类别的序号
                    images.append(item)

    return images # image是一个列表，元素都是二元组，存储图片和图片所属类别的序号
'''

def make_label(file_path):
    count = 0
    with open(file_path,'r',encoding='utf-8') as f:
        dic=[]
        for line in f.readlines():
            line=line.strip('\n') #去掉换行符\n
            b=line.split(',') #将每一行以逗号为分隔符转换成列表
            #print(type(b))
            #print(b)
            del b[1] # 删除第二个元素
            b.append(count) # 增加value
            #print(b)
            dic.append(b)
            count += 1
        dic=dict(dic)
    return dic

def make_dataset(dir, dic):
    images = []
    # 读取list文件，返回images，和多元分类一样，image是一个列表，元素都是二元组，存储图片和图片所属类别的np.array数组
    for line in open(dir):
        line=line.strip('\n') #去掉换行符\n
        b=line.split(' ') #将每一行以空格为分隔符转换成列表
        length = len(b) # 获取列表b的长度
        # 获取图片路径
        path = b[0]

        # 获取图片所属类别的np.array数组
        label_array = np.zeros(107) # 全0的标签数组

        for i in range(length - 2):
            label_array[dic[b[i + 2]]] = 1 #所属的类别置1
        item = (path, label_array)
        images.append(item)

    return images

class DatasetFolder(data.Dataset):
    """A generic data loader where the samples are arranged in this way: ::

        root/class_x/xxx.ext
        root/class_x/xxy.ext
        root/class_x/xxz.ext

        root/class_y/123.ext
        root/class_y/nsdf3.ext
        root/class_y/asd932_.ext

    Args:
        root (string): Root directory path.
        loader (callable): A function to load a sample given its path.
        extensions (list[string]): A list of allowed extensions.
        transform (callable, optional): A function/transform that takes in
            a sample and returns a transformed version.
            E.g, ``transforms.RandomCrop`` for images.
        target_transform (callable, optional): A function/transform that takes
            in the target and transforms it.

     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        samples (list): List of (sample path, class_index) tuples
    """
    def __init__(self, root, loader, extensions, transform=None, target_transform=None):
        #classes, class_to_idx = find_classes(root) # classes为列表，存储类别名字；class_to_idx为字典，存储列表名字和对应的序号
        #samples = make_dataset(root, class_to_idx, extensions) # sample是一个列表，元素都是二元组，存储图片和图片所属类别的序号
        file_path = '/home/momo/sun.zheng/multi_label_classification/GhostNet/valid-labels-107.txt'
        dic = make_label(file_path)
        samples = make_dataset(root, dic) # 传入一个list文件路径和标签字典，从该list文件中解析出所有图片和相应的标签
        if len(samples) == 0:
            raise(RuntimeError("Found 0 files in subfolders of: " + root + "\n"
                               "Supported extensions are: " + ",".join(extensions)))

        self.root = root
        self.loader = loader
        self.extensions = extensions

        #self.classes = classes
        #self.class_to_idx = class_to_idx
        self.samples = samples

        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index] # 图片路径和所属的类别的np.array数组
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target # 返回loader(path)和该图片标签的一维np.array数组

    def __len__(self):
        return len(self.samples)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str



IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif']

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)

class ImageFolder(DatasetFolder):
    """A generic data loader where the images are arranged in this way: ::

        root/dog/xxx.png
        root/dog/xxy.png
        root/dog/xxz.png

        root/cat/123.png
        root/cat/nsdf3.png
        root/cat/asd932_.png

    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.

     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """
    def __init__(self, root, transform=None, target_transform=None,
                 loader=default_loader):
        super(ImageFolder, self).__init__(root, loader, IMG_EXTENSIONS,
                                          transform=transform,
                                          target_transform=target_transform)
        self.imgs = self.samples
```

​    ImageFolder类继承了DatasetFolder类，参数有root数据根目录，图片加载方式loader，图片预处理方式为transform，相应的标签预处理为target_transform(target_transform默认为None)。在实例化ImageFolder类对象时，要先初始化DatasetFolder类。

​    DatasetFolder初始化函数__init__中，传入原始标签的路径，make_label函数将原始标签转换为一个字典，键为类别的英文名称，值为类别的数组索引；再将得到标签字典和训练集（测试集）对应的txt文件路径传入make_dataset函数，得到数据集的二元组，\__init__函数，make_label以及make_dataset函数源码如下：

```python
    def __init__(self, root, loader, extensions, transform=None, target_transform=None):
        #classes, class_to_idx = find_classes(root) # classes为列表，存储类别名字；class_to_idx为字典，存储列表名字和对应的序号
        #samples = make_dataset(root, class_to_idx, extensions) # sample是一个列表，元素都是二元组，存储图片和图片所属类别的序号
        file_path = '/home/momo/sun.zheng/multi_label_classification/GhostNet/valid-labels-107.txt'
        dic = make_label(file_path)
        samples = make_dataset(root, dic) # 传入一个list文件路径和标签字典，从该list文件中解析出所有图片和相应的标签
        if len(samples) == 0:
            raise(RuntimeError("Found 0 files in subfolders of: " + root + "\n"
                               "Supported extensions are: " + ",".join(extensions)))

        self.root = root
        self.loader = loader
        self.extensions = extensions

        #self.classes = classes
        #self.class_to_idx = class_to_idx
        self.samples = samples

        self.transform = transform
        self.target_transform = target_transform
```

```python
def make_label(file_path):
    count = 0
    with open(file_path,'r',encoding='utf-8') as f:
        dic=[]
        for line in f.readlines():
            line=line.strip('\n') #去掉换行符\n
            b=line.split(',') #将每一行以逗号为分隔符转换成列表
            #print(type(b))
            #print(b)
            del b[1] # 删除第二个元素
            b.append(count) # 增加value
            #print(b)
            dic.append(b)
            count += 1
        dic=dict(dic)
    return dic

def make_dataset(dir, dic):
    images = []
    # 读取list文件，返回images，和多元分类一样，images是一个列表，元素都是二元组，存储图片和图片所属类别的np.array数组
    for line in open(dir):
        line=line.strip('\n') #去掉换行符\n
        b=line.split(' ') #将每一行以空格为分隔符转换成列表
        length = len(b) # 获取列表b的长度
        # 获取图片路径
        path = b[0]

        # 获取图片所属类别的np.array数组
        label_array = np.zeros(107) # 全0的标签数组

        for i in range(length - 2):
            label_array[dic[b[i + 2]]] = 1 #所属的类别置1
        item = (path, label_array)
        images.append(item)

    return images

```





2.多标签分类任务训练代码解读

（和多元分类大体一样，不过需要修改一些地方）

（1）导入改写之后的ImageFolder

```python
from D_v2 import ImageFolder # 代替pytorch自带的ImageFolder来读取多标签数据
```

（2）输入的参数更改为训练集和测试集的txt文件

```python
parser.add_argument('--train_data', metavar='DIR', default='/home/momo/sun.zheng/multi_label_classification/GhostNet/train.txt', help='path to train_dataset')
parser.add_argument('--val_data', metavar='DIR', default='/home/momo/sun.zheng/multi_label_classification/GhostNet/test.txt', help='path to val_dataset')
```

（3）使用改写之后的ImageFolder来加载数据

```python
train_dataset = ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=args.batch_size,
                                               shuffle=(train_sampler is None),
                                               num_workers=2,
                                               pin_memory=True,
                                               sampler=train_sampler)
val_dataset = ImageFolder(
        valdir,
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]))
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                            batch_size = args.batch_size,
                                            shuffle=False,
                                            num_workers=2,
                                            pin_memory=True)

```

（4）衡量指标改为精确度和召回率

```python
top1 = AverageMeter('Acc', ':6.2f')
top2 = AverageMeter('Recall', ':6.2f')
# 后续的相应修改
```

（5）修改准确率函数

```python
def accuracy(output, target):
    acc = recall = 0
    pred = torch.sigmoid(output).ge(0.5).cpu()
    pred = [pred_one.nonzero().reshape(-1).numpy().tolist() for pred_one in pred]
    target = target.cpu()
    target = [target_one.nonzero().reshape(-1).numpy().tolist() for target_one in target]
    for i in range(len(target)):
        if len(pred[i]) != 0:
            acc += len(set(pred[i])&set(target[i])) / (len(pred[i])+1e-7)
            recall += len(set(pred[i])&set(target[i])) / (len(target[i])+1e-7)
    acc = acc / len(target)
    recall = recall / len(target)
    return torch.tensor(acc).cuda().float(), torch.tensor(recall).cuda().float()
```



3.训练结果

（1）GhostNet



（2）ShuffleNetV2



（3）MobileNetV2