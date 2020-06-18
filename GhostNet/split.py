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



