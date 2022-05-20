from tqdm import tqdm
import gc
import time
from collections import defaultdict
import numpy as np
import copy

import torch


def train_discriminator_one_epoch(G, D, optimizerD, schedulerD, train_loader, device, epoch, criterion):
    G.train()
    D.train()

    
    dataset_size = 0
    running_loss = 0.0
    
    pbar = tqdm(enumerate(train_loader), total=len(train_loader))
    for step, images in pbar:         
        images = images.to(device)

        batch_size = images.size(0)
        
        # 真偽のラベルを定義
        label_real = torch.full((images.size(0),), 1.0).to(device)
        label_fake = torch.full((images.size(0),), 0.0).to(device)
        
        # Generator を用いて潜在変数から偽の画像を生成
        z = torch.randn(images.size(0), 20).to(device).view(images.size(0), 20, 1, 1).to(device)
        fake_images = G(z)

        # Discriminator で偽の画像と本物の画像を判定
        d_out_real, _ = D(images)
        d_out_fake, _ = D(fake_images)

        # 損失の計算
        d_loss_real = criterion(d_out_real.view(-1), label_real)
        d_loss_fake = criterion(d_out_fake.view(-1), label_fake)
        d_loss = d_loss_real + d_loss_fake
                
        optimizerD.zero_grad()
        d_loss.backward()
        optimizerD.step()
        schedulerD.step()

        running_loss += (d_loss.item() * batch_size)
        dataset_size += batch_size

        epoch_loss = running_loss / dataset_size
        
        mem = torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0
        current_lr = optimizerD.param_groups[0]['lr']
        pbar.set_postfix(train_loss=f'{epoch_loss:0.4f}',
                        lr=f'{current_lr:0.5f}',
                        gpu_mem=f'{mem:0.2f} GB')
    torch.cuda.empty_cache()
    gc.collect()
    
    return epoch_loss

@torch.no_grad()
def valid_discriminator_one_epoch(G, D, optimizerD, valid_loader, device, epoch, criterion):
    G.eval()
    D.eval()
    
    dataset_size = 0
    running_loss = 0.0

    
    pbar = tqdm(enumerate(valid_loader), total=len(valid_loader))
    for step, images in pbar:        
        images  = images.to(device, dtype=torch.float)
        
        batch_size = images.size(0)
        
        # 真偽のラベルを定義
        label_real = torch.full((images.size(0),), 1.0).to(device)
        label_fake = torch.full((images.size(0),), 0.0).to(device)
        
        # Generator を用いて潜在変数から偽の画像を生成
        z = torch.randn(images.size(0), 20).to(device).view(images.size(0), 20, 1, 1).to(device)
        fake_images = G(z)

        # Discriminator で偽の画像と本物の画像を判定
        d_out_real, _ = D(images)
        d_out_fake, _ = D(fake_images)

        # 損失の計算
        d_loss_real = criterion(d_out_real.view(-1), label_real)
        d_loss_fake = criterion(d_out_fake.view(-1), label_fake)
        d_loss = d_loss_real + d_loss_fake

        running_loss += (d_loss.item() * batch_size)
        dataset_size += batch_size
        
        epoch_loss = running_loss / dataset_size
        
        mem = torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0
        current_lr = optimizerD.param_groups[0]['lr']
        pbar.set_postfix(Epoch=epoch, valid_loss=f'{epoch_loss:0.4f}',lr=f'{current_lr:0.5f}',gpu_memory=f'{mem:0.2f} GB')
    torch.cuda.empty_cache()
    gc.collect()
    
    return epoch_loss

def train_generator_one_epoch(G, D, optimizerG, schedulerG, train_loader, device, epoch, criterion):
    G.train()
    D.train()
    
    dataset_size = 0
    running_loss = 0.0
    
    pbar = tqdm(enumerate(train_loader), total=len(train_loader))
    for step, images in pbar:         
        images = images.to(device)

        batch_size = images.size(0)

        # 真偽のラベルを定義
        label_real = torch.full((images.size(0),), 1.0).to(device)
        label_fake = torch.full((images.size(0),), 0.0).to(device)
        
         # 潜在変数から偽の画像を生成
        z = torch.randn(images.size(0), 20).to(device).view(images.size(0), 20, 1, 1).to(device)
        fake_images = G(z)

        # Discriminator によって真偽判定
        d_out_fake, _ = D(fake_images)

        # 損失の計算
        g_loss = criterion(d_out_fake.view(-1), label_real)
    
        # 誤差逆伝播法で勾配の計算、重みの更新
        optimizerG.zero_grad()
        g_loss.backward()
        optimizerG.step()
        schedulerG.step()

        running_loss += (g_loss.item() * batch_size)
        dataset_size += batch_size

        epoch_loss = running_loss / dataset_size
        
        mem = torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0
        current_lr = optimizerG.param_groups[0]['lr']
        pbar.set_postfix(train_loss=f'{epoch_loss:0.4f}',
                        lr=f'{current_lr:0.5f}',
                        gpu_mem=f'{mem:0.2f} GB')
    torch.cuda.empty_cache()
    gc.collect()
    
    return epoch_loss

@torch.no_grad()
def valid_generator_one_epoch(G, D, optimizerG, valid_loader, device, epoch, criterion):
    G.eval()
    D.eval()
    
    dataset_size = 0
    running_loss = 0.0

    
    pbar = tqdm(enumerate(valid_loader), total=len(valid_loader))
    for step, images in pbar:        
        images  = images.to(device, dtype=torch.float)
        
        batch_size = images.size(0)
        
        # 真偽のラベルを定義
        label_real = torch.full((images.size(0),), 1.0).to(device)

        # 潜在変数から偽の画像を生成
        z = torch.randn(images.size(0), 20).to(device).view(images.size(0), 20, 1, 1).to(device)
        fake_images = G(z)

        # Discriminator によって真偽判定
        d_out_fake, _ = D(fake_images)

        # 損失の計算
        g_loss = criterion(d_out_fake.view(-1), label_real)
    

        running_loss += (g_loss.item() * batch_size)
        dataset_size += batch_size
        
        epoch_loss = running_loss / dataset_size
        
        mem = torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0
        current_lr = optimizerG.param_groups[0]['lr']
        pbar.set_postfix(Epoch=epoch, valid_loss=f'{epoch_loss:0.4f}',lr=f'{current_lr:0.5f}',gpu_memory=f'{mem:0.2f} GB')
    torch.cuda.empty_cache()
    gc.collect()
    
    return epoch_loss

def run_training(G, D, train_loader, valid_loader, optimizerG, optimizerD, schedulerG, schedulerD, criterion, device, num_epochs, fold, config):
    # To automatically log gradients
    
    if torch.cuda.is_available():
        print("cuda: {}\n".format(torch.cuda.get_device_name()))
    
    start = time.time()
    best_model_wts_G = copy.deepcopy(G.state_dict())
    best_epoch_loss_G = np.inf
    best_model_wts_D = copy.deepcopy(D.state_dict())
    best_epoch_loss_D = np.inf
    best_epoch     = -1
    history = defaultdict(list)
    
    for epoch in range(config.training.epochs): 
        gc.collect()
        print(f'Epoch {epoch}/{num_epochs}', end='')
        train_dis_loss = train_discriminator_one_epoch(G, D, optimizerD, schedulerD, train_loader=train_loader, device=device, epoch=epoch, criterion=criterion)
        train_gen_loss = train_generator_one_epoch(G, D, optimizerG, schedulerG, train_loader=train_loader, device=device, epoch=epoch, criterion=criterion)

        valid_dis_loss = valid_discriminator_one_epoch(G, D, optimizerD, valid_loader=valid_loader, device=device, epoch=epoch, criterion=criterion)
        valid_gen_loss = valid_generator_one_epoch(G, D, optimizerG, valid_loader=valid_loader, device=device, epoch=epoch, criterion=criterion)

        

        history['Train Dis Loss'].append(train_dis_loss)
        history['Train Gen Loss'].append(train_gen_loss)
        history['Valid Dis Loss'].append(valid_dis_loss)
        history['Valid Gen Loss'].append(valid_gen_loss)
        
        
        print('Valid Disctiminator Loss')
        print(valid_dis_loss)
        print('Valid Generator Loss')
        print(valid_gen_loss)

        # deep copy the model
        if valid_dis_loss <= best_epoch_loss_D:
          print(f"d Loss Improved ({best_epoch_loss_D} ---> {valid_dis_loss})")
          best_epoch_loss_D = valid_dis_loss
          best_model_wts_D = copy.deepcopy(D.state_dict())
          DPATH = config.save.output_root_dir+f"D_model_fold{fold}.bin"
          torch.save(D.state_dict(), DPATH)
          # Save a model file from the current directory
          print(f"D_Model Saved")

        if valid_gen_loss <= best_epoch_loss_G:
          print(f"g Loss Improved ({best_epoch_loss_G} ---> {valid_gen_loss})")
          best_epoch_loss_G = valid_gen_loss
          best_model_wts_G = copy.deepcopy(G.state_dict())
          GPATH = config.save.output_root_dir+f"G_model_fold{fold}.bin"
          torch.save(G.state_dict(), GPATH)
          # Save a model file from the current directory
          print(f"G_Model Saved")

        
        #if you need save last model
        last_model_wts_D = copy.deepcopy(D.state_dict())
        DLPATH = config.save.output_root_dir+f"D_last_model_fold{fold}.bin"
        torch.save(D.state_dict(), DLPATH)

        last_model_wts_G = copy.deepcopy(G.state_dict())
        GLPATH = config.save.output_root_dir+f"G_last_model_fold{fold}.bin"
        torch.save(G.state_dict(), GLPATH)
            
        print(); print()
    
    end = time.time()
    time_elapsed = end - start
    print('Training complete in {:.0f}h {:.0f}m {:.0f}s'.format(
        time_elapsed // 3600, (time_elapsed % 3600) // 60, (time_elapsed % 3600) % 60))
    print('Best Discriminator Loss')
    print(best_epoch_loss_D)
    print('Best Generator Loss')
    print(best_epoch_loss_G)
    
    # load best model weights
    G.load_state_dict(best_model_wts_G)
    D.load_state_dict(best_model_wts_D)
    
    return G, D, history


