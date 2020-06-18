
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

