import numpy as np
import pandas as pd
import argparse
from collections import defaultdict
from sklearn.model_selection import KFold

import torch
from torch import nn

from models import Generator
from models import Discriminator
from utils import fetch_scheduler
from utils import load_json
from utils import prepare_loaders
from train_function import run_training

def main(config):
    print('read data')
    df_train = pd.read_csv(config.dataset.root_dir_path+"mnist_train_small.csv",dtype = np.float32)
    df_train.rename(columns={'6': 'label'}, inplace=True)
    # 学習データとして、1を2000枚使用する
    df_train = df_train.query("label in [1.0]").head(2000)

    #trainとvalidの分割
    kf = KFold(n_splits=config.training.n_fold, shuffle=True, random_state=config.training.seed)
    df_train['fold']=1
    for fold, (train_idx, val_idx) in enumerate(kf.split(df_train)):
      df_train.iloc[val_idx,-1] = fold
    df_train=df_train.reset_index(drop=True)

    for fold in range(1):
      print(f'#'*15)
      print(f'### Fold: {fold}')
      print(f'#'*15)

      # GPU or CPU の指定
      device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
      # モデルの読み込み
      G = Generator(z_dim=config.model.z_dim, gen_filters=config.model.gen_filters).to(device)
      D = Discriminator(input_dim=config.model.input_dim, dis_filters=config.model.dis_filters).to(device)
      train_loader, valid_loader = prepare_loaders(fold=fold, df=df_train, config=config)
      # オプティマイザの設定
      optimizerG = torch.optim.AdamW(G.parameters(), lr=config.optimizer.learning_rate, weight_decay=config.optimizer.weight_decay)
      optimizerD = torch.optim.AdamW(D.parameters(), lr=config.optimizer.learning_rate, weight_decay=config.optimizer.weight_decay)
      schedulerG = fetch_scheduler(optimizer=optimizerG, scheduler=config.optimizer.scheduler, T_max=config.optimizer.T_max, min_lr=config.optimizer.min_lr)
      schedulerD = fetch_scheduler(optimizer=optimizerD, scheduler=config.optimizer.scheduler, T_max=config.optimizer.T_max, min_lr=config.optimizer.min_lr)
      # 学習時の損失関数の定義
      criterion = nn.BCEWithLogitsLoss(reduction='mean')
      G, D, history = run_training(G, D, train_loader, valid_loader, optimizerG, optimizerD, schedulerG, schedulerD, criterion, device, config.training.epochs, fold, config)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='mnist detection')
    parser.add_argument('-c', '--config', help='training config file', required=True)
    args = parser.parse_args()

    config = load_json(args.config)

    main(config)
