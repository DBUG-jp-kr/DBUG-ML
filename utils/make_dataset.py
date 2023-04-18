import torch
from torchvision import transforms, datasets

import random


random_seed = 777
random.seed(random_seed)
torch.manual_seed(random_seed)

data_path = 'insect_data/'  
insect_dataset = datasets.ImageFolder(
                                data_path,
                                transforms.Compose([
                                    transforms.Resize((224, 224)),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                ]))
# data split
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
train_idx, tmp_idx = train_test_split(list(range(len(insect_dataset))), test_size=0.2, random_state=random_seed)
datasets = {}
datasets['train'] = Subset(insect_dataset, train_idx)
tmp_dataset       = Subset(insect_dataset, tmp_idx)

val_idx, test_idx = train_test_split(list(range(len(tmp_dataset))), test_size=0.5, random_state=random_seed)
datasets['valid'] = Subset(tmp_dataset, val_idx)
datasets['test']  = Subset(tmp_dataset, test_idx)