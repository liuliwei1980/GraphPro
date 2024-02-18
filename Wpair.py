import numpy as np
import collections
import math
import torch
def Gaussian(x):
    return math.exp(-0.5*(x*x))
def paired(x,y):
    if x == 'A' and y == 'U':
        return 2
    elif x == 'G' and y == 'C':
        return 3
    elif x == "G"and y == 'U':
        return 0.8
    elif x == 'U' and y == 'A':
        return 2
    elif x == 'C' and y == 'G':
        return 3
    elif x == "U"and y == 'G':
        return 0.8
    else:
        return 0
def creatmat(data):
    pair_data = []
    with open(r'./Datasets/'+protein+'/positive') as f:
        num = 0
        for data in f:
            if '>' not in data:
                data = data[:-1]
                mat = np.zeros([len(data),len(data)])
                for i in range(len(data)):
                    for j in range(len(data)):
                        coefficient = 0
                        for add in range(30):
                            if i - add >= 0 and j + add <len(data):
                                score = paired(data[i - add].replace('T', 'U'),data[j + add].replace('T', 'U'))
                                if score == 0:
                                    break
                                else:
                                    coefficient = coefficient + score * Gaussian(add)
                            else:
                                break
                        if coefficient > 0:
                            for add in range(1,30):
                                if i + add < len(data) and j - add >= 0:
                                    score = paired(data[i + add],data[j - add])
                                    if score == 0:
                                        break
                                    else:
                                        coefficient = coefficient + score * Gaussian(add)
                                else:
                                    break
                        mat[[i],[j]] = coefficient
                if len(pair_data)==0:
                    pair_data = torch.from_numpy(mat).unsqueeze(0)
                else:
                    matt = torch.from_numpy(mat).unsqueeze(0)
                    pair_data = torch.cat((pair_data,matt),0)
                num=num+1
                print(num)
    with open(r'./Datasets/'+protein+'/negative') as f:
        for data in f:
            if '>' not in data:
                data = data[:-1]
                mat = np.zeros([len(data), len(data)])
                for i in range(len(data)):
                    for j in range(len(data)):
                        coefficient = 0
                        for add in range(30):
                            if i - add >= 0 and j + add < len(data):
                                score = paired(data[i - add], data[j + add])
                                if score == 0:
                                    break
                                else:
                                    coefficient = coefficient + score * Gaussian(add)
                            else:
                                break
                        if coefficient > 0:
                            for add in range(1, 30):
                                if i + add < len(data) and j - add >= 0:
                                    score = paired(data[i + add], data[j - add])
                                    if score == 0:
                                        break
                                    else:
                                        coefficient = coefficient + score * Gaussian(add)
                                else:
                                    break
                        mat[[i], [j]] = coefficient

                matt = torch.from_numpy(mat).unsqueeze(0)
                pair_data = torch.cat((pair_data, matt), 0)
                num = num + 1
                print(num)
    return pair_data

protein="mouse"
pair = creatmat(protein)#到时候用npy文件存一下
np.save('./Datasets/'+protein+'/pair.npy',pair)

