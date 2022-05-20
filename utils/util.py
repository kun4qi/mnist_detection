import torch
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader

import json
import collections

from dataio import image_data_set

def fetch_scheduler(optimizer, scheduler, T_max, min_lr):
    if scheduler == 'CosineAnnealingLR':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer,T_max=T_max, 
                                                   eta_min=min_lr)
    elif scheduler == None:
        return None
        
    return scheduler

def load_json(path):
    def _json_object_hook(d):
        return collections.namedtuple('X', d.keys())(*d.values())
    def _json_to_obj(data):
        return json.loads(data, object_hook=_json_object_hook)
    return _json_to_obj(open(path).read())

def anomaly_score(input_image, fake_image, D):
  # Residual loss の計算
  residual_loss = torch.sum(torch.abs(input_image - fake_image), (1, 2, 3))

  # Discrimination loss の計算
  _, real_feature = D(input_image)
  _, fake_feature = D(fake_image)
  discrimination_loss = torch.sum(torch.abs(real_feature - fake_feature), (1))

  # 二つのlossを一定の割合で足し合わせる
  total_loss_by_image = 0.9 * residual_loss + 0.1 * discrimination_loss
  total_loss = total_loss_by_image.sum()

  return total_loss, total_loss_by_image, residual_loss

def prepare_loaders(fold, df, config):
    train_df = df.query("fold!=@fold").reset_index(drop=True)
    valid_df = df.query("fold==@fold").reset_index(drop=True)
    # ラベル(1列目)を削除
    train = train_df.iloc[:,1:-1].values.astype('float32')
    valid = valid_df.iloc[:,1:-1].values.astype('float32')

    # 28×28 の行列に変換
    train = train.reshape(train.shape[0], 28, 28)
    valid = valid.reshape(valid.shape[0], 28, 28)

    train_dataset = image_data_set(train, image_size=config.dataset.img_size)
    valid_dataset = image_data_set(valid, image_size=config.dataset.img_size)

    train_loader = DataLoader(train_dataset, batch_size=config.dataset.train_batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=config.dataset.valid_batch_size, shuffle=True)
    
    return train_loader, valid_loader