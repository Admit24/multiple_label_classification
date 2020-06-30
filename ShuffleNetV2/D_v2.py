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
