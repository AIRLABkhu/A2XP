# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import os
if __name__ == '__main__':
    import sys
    __root = __file__
    for _ in range(2): 
        __root = os.path.dirname(__root)
    sys.path.append(__root)
from torch.utils.data import ConcatDataset

__DOMAINBED_ROOT = os.path.expanduser('~/.cache/domainbed')


def get_domains_and_classes(dataset_name: str, root: str=None):
    from domainbed import datasets as db_datasets
    from domainbed import hparams_registry as db_hparams
    
    if root is None:
        root = __DOMAINBED_ROOT
    
    for dset in db_datasets.DATASETS:
        if dset.lower() == dataset_name:
            dataset_name = dset

    hparams = db_hparams.default_hparams('ERM', dataset_name)
    dataset_type = db_datasets.get_dataset_class(dataset_name)
    dataset = list(dataset_type(root, [], hparams))[0]
    domains = dataset_type.ENVIRONMENTS
    classes = dataset.classes
    
    return domains, classes


def get_dataset(dataset_name: str, target: str, 
                augmentation: bool=False, include_domain: bool=False, 
                return_domain_names: bool=False, root: str=None):
    from domainbed import datasets as db_datasets
    from domainbed import hparams_registry as db_hparams
    
    if root is None:
        root = __DOMAINBED_ROOT
    
    for dset in db_datasets.DATASETS:
        if dset.lower() == dataset_name:
            dataset_name = dset

    dataset_type = db_datasets.get_dataset_class(dataset_name)
    hparams = db_hparams.default_hparams('ERM', dataset_name)
    hparams['include_domain'] = include_domain

    target = target.lower()
    sources = []
    target_idx = None
    for i, env in enumerate(dataset_type.ENVIRONMENTS):
        if env.lower() == target:
            target = env
            target_idx = i
        else:
            sources.append(env)
    
    hparams['data_augmentation'] = augmentation
    datasets = list(dataset_type(root, [target_idx], hparams))

    tar_dataset = datasets.pop(target_idx)
    src_dataset = ConcatDataset(datasets)
    
    if return_domain_names:
        return src_dataset, tar_dataset, sources, target
    return src_dataset, tar_dataset
