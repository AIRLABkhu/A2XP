import torch
from torch.utils.data import Dataset, Subset, ConcatDataset


def class_uniform_split(dataset: Dataset, split: float, shuffle: bool=True):
    num_classes = len(dataset.classes)
    targets = torch.tensor(dataset.targets)
    first_split_list, second_split_list = [], []
    for class_idx in range(num_classes):
        indices = (targets == class_idx).nonzero().flatten()
        num_samples = len(indices)
        
        if shuffle:
            indices = indices[torch.randperm(num_samples)]
        first_split_size = int(num_samples * split)
        
        first_split_list.append(Subset(dataset, indices[:first_split_size]))
        second_split_list.append(Subset(dataset, indices[first_split_size:]))
    
    return ConcatDataset(first_split_list), ConcatDataset(second_split_list)
