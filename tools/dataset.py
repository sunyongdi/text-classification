import os
import sys
import torch

from torch.utils.data import Dataset

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

from utils import load_pkl

def collate_fn(cfg):
    
    def collate_fn_intra(batch):
        """
    Arg : 
        batch () : 数据集
    Returna : 
        inputs (dict) : key为词，value为长度
        labels (List) : 关系对应值的集合
    """

        input_ids = []
        token_type_ids = []
        attention_mask = []
        labels = []
        for data in batch:
            inputs = data['inputs']
            label = int(data['label'])
            input_ids.append(inputs['input_ids'])
            token_type_ids.append(inputs['token_type_ids'])
            attention_mask.append(inputs['attention_mask'])
            labels.append(label)
        labels = torch.tensor(labels)  
        inputs = {'input_ids': torch.tensor(input_ids), 'token_type_ids': torch.tensor(token_type_ids), 'attention_mask': torch.tensor(attention_mask)} 
        return inputs, labels

    return collate_fn_intra


class TextClassifyDataset(Dataset):
    """
    默认使用 List 存储数据
    """
    def __init__(self, fp):
        self.file = load_pkl(fp)

    def __getitem__(self, item):
        sample = self.file[item]
        return sample

    def __len__(self):
        return len(self.file)
    

if __name__ == '__main__':
    train_data_path = '/root/sunyd/NoteBook/base-demo/text_classification/data/dataset/train.csv'
    dataset = TextClassifyDataset(train_data_path)
    print(dataset[2])