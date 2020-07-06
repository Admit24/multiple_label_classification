# 划分有标签和没有标签的业务图像至不同的文件夹，并在有标签的图像当中统计各个类别的数量

import subprocess

label_path = './labelsmap.txt'
business_data_result = './business_label_result.txt'

# 获取标签字典，键为类别名称，值为类别名称出现的次数，所有类别次数均初始化为0
def make_label(label_path):
    with open(label_path,'r') as f:
        dic = []
        for line in f.readlines():
            line = line.strip('\n') # 去掉换行符
            b=line.split(':') #将每一行以冒号为分隔符转换成列表
            del b[1]
            b.append(0)
            dic.append(b)
        dic=dict(dic)
    return dic

def main():
    dic = make_label(label_path)
    #print(dic)
    # 划分有标签和无标签的图像，并统计标签数量
    with open(business_data_result,'r') as f:
        for line in f.readlines():
            line = line.strip('\n') # 去掉换行符
            b=line.split(' ') #将每一行以冒号为分隔符转换成列表
            if len(b)==1:
                cmd = 'cp ./images/' + b[0] + ' ' + './image_without_label/'
            else:
                cmd = 'cp ./images/' + b[0] + ' ' + './image_with_label/'
                for i in range(1, len(b)):
                    dic[b[i]] += 1 # 相应的类别数加1
            subprocess.call(cmd, shell=True)
            #print(b[0])
    print(dic)


if __name__=='__main__':
    main()


