from __future__ import print_function
import os.path as osp
from torch.utils.data import DataLoader
from reid.dataset import get_sequence
from reid.data import seqtransforms as T
from reid.data import SeqTrainPreprocessor
from reid.data import SeqTestPreprocessor
from reid.data import RandomPairSampler


def get_data(dataset_name, split_id, data_dir, batch_size, seq_len, seq_srd, workers, train_mode):

    root = osp.join(data_dir, dataset_name)
    dataset = get_sequence(dataset_name, root, split_id=split_id,
                           seq_len=seq_len, seq_srd=seq_srd, num_val=1, download=True)
    train_set = dataset.trainval
    num_classes = dataset.num_trainval_ids
    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    train_processor = SeqTrainPreprocessor(train_set, dataset, seq_len,
                                           transform=T.Compose([T.RectScale(256, 128),
                                                                T.RandomHorizontalFlip(),
                                                                T.RandomSizedEarser(),
                                                                T.ToTensor(), normalizer]))

    query_processor = SeqTestPreprocessor(dataset.query, dataset, seq_len,
                                          transform=T.Compose([T.RectScale(256, 128),
                                                               T.ToTensor(), normalizer]))

    gallery_processor = SeqTestPreprocessor(dataset.gallery, dataset, seq_len,
                                            transform=T.Compose([T.RectScale(256, 128),
                                                                 T.ToTensor(), normalizer]))

    if train_mode == 'cnn_rnn':
        train_loader = DataLoader(train_processor, batch_size=batch_size, num_workers=workers,
                                  sampler=RandomPairSampler(train_set), pin_memory=True)
    elif train_mode == 'cnn':
        train_loader = DataLoader(train_processor, batch_size=batch_size, num_workers=workers,
                                  shuffle=True, pin_memory=True)
    else:
        raise ValueError('no such train mode')

    query_loader = DataLoader(query_processor, batch_size=8, num_workers=workers, shuffle=False, pin_memory=True)

    gallery_loader = DataLoader(gallery_processor, batch_size=8, num_workers=workers, shuffle=False, pin_memory=True)

    return dataset, num_classes, train_loader, query_loader, gallery_loader
