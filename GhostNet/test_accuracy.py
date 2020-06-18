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

label2 = torch.tensor([[0., 0., 0.],
                       [0., 0., 1.],
                       [0., 1., 0.]])

acc1 = accuracy1(inputs, label1, (1, 2))
print('acc:' + str(acc1))

#label2 = label2.float()
acc2, recall = accuracy2(inputs, label2)
print('acc2:' + str(acc2))
print('recall:' + str(recall))


