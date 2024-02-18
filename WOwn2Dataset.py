import torch
from torch.utils.data import Dataset,DataLoader
import numpy as np
import wyx2PreData


class WOwnDataset(Dataset):
    def __init__(self,protein):
        super(WOwnDataset, self).__init__()
        Kmer,DPCP, Y,vec,name = wyx2PreData.all_data(protein)  # 这里取pair太慢，还是要用npy文件读
        pair = np.load('./Datasets/'+protein+'/pair.npy', allow_pickle=True)
        self.Y = Y
        dict_data = {'Kmer': Kmer,'DPCP':DPCP,'vec': vec,'pair': pair, 'Y': Y,'name':name}

        self.dict_data = dict_data


    def __getitem__(self, idx):
        item = {key: torch.tensor(value[idx]) for key, value in self.dict_data.items()}
        return item

    def __len__(self):
        return len(self.Y)





