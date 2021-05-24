import time
import os.path as osp
from tqdm import tqdm
import numpy as np

import torch
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss

from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, CosineAnnealingLR, ReduceLROnPlateau, StepLR, MultiStepLR
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast, GradScaler

from .criterions import *
from .mixing import mixing_fn
from utils.metrics import calc_auc




def run_training(model, trainloader, validloader, epochs, optimizer, optimizer_params, scheduler, \
    scheduler_params, loss_tr, loss_fn, early_stopping_steps, verbose, device, seed, fold, weight_path, mixing, log_path, grad_accum_step):
    Writer = SummaryWriter(log_dir=log_path)

    if loss_tr == "ClassWeightedCrossEntropyLoss":
        print("ClassWeighted")
        _, weights =  np.unique(trainloader.dataset.labels, return_counts=True)
        weights = torch.Tensor(1/weights).to(device)
        loss_tr = CrossEntropyLoss(weight=weights)
    else:
        loss_tr = eval(loss_tr)()
    loss_fn = eval(loss_fn)()
    optimizer = eval(optimizer)(model.parameters(), **optimizer_params)
    scheduler = eval(scheduler)(optimizer, **scheduler_params)

    early_step = 0
    best_loss = np.inf
    best_auc = -np.inf
    best_recall = -np.inf
    best_epoch = 0
    start = time.time()

    scaler = GradScaler() ## 

    for epoch in range(epochs):
        train_loss = train_fn(model, optimizer, scheduler, loss_tr, trainloader, device, scaler, epoch, mixing, grad_accum_step)
        valid_auc = valid_fn(model, loss_fn, validloader, device, epoch)
        torch.cuda.empty_cache()
        # Writer
        Writer.add_scalar(f"{seed}_{fold}_train_losses", train_loss, epoch)
        Writer.add_scalar(f"{seed}_{fold}_valid_auc", valid_auc, epoch)        

        # scheduler step
        if isinstance(scheduler, ReduceLROnPlateau):
            scheduler.step(valid_loss)
        elif isinstance(scheduler, CosineAnnealingLR):
            scheduler.step()
        elif isinstance(scheduler, CosineAnnealingWarmRestarts):
            scheduler.step()
        elif isinstance(scheduler, StepLR):
            scheduler.step()
                
        # if valid_loss < best_loss:
        if best_auc < valid_auc:
            best_auc = valid_auc
            torch.save(model.model.state_dict(), osp.join( weight_path,  f"{seed}_{fold}.pt") )
            #torch.save(model.model.state_dict(), osp.join( weight_path,  f"{seed}_{fold}.pt") ) # networkを保存
            early_step = 0
            best_epoch = epoch
        
        elif early_stopping_steps != 0:
            early_step += 1
            if (early_step >= early_stopping_steps):
                t = time.time() - start
                print(f"early stopping in iteration {epoch},  : best itaration is {best_epoch} | valid auc {best_auc:.4f}")
                Writer.close()
                return 0

    t = time.time() - start   
    torch.save(model.model.state_dict(), osp.join( weight_path,  f"{seed}_{fold}_final.pt") )    
    print(f"training until max epoch {epochs},  : best itaration is {best_epoch} | valid auc {best_auc:.4f}")
    Writer.close()
    return 0


def train_fn(model, optimizer, scheduler, loss_fn, dataloader, device, scaler, epoch, mixing, grad_accum_step):
    model.train()
    final_loss = 0
    s = time.time()
    pbar = tqdm(enumerate(dataloader), total=len(dataloader))
    for i, (images, labels) in pbar:
        # mix up 
        images, labels = mixing_fn(images, labels, mixing)
        
        images = images.to(device).float()
        labels = labels.to(device)

        with autocast():
            outputs = model(images, labels)
            loss = loss_fn(outputs, labels)
            scaler.scale(loss).backward()
            final_loss += loss.item() 
            del loss; torch.cuda.empty_cache()
            if (i+1) % grad_accum_step == 0 or ((i + 1) == len(dataloader)):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

        if i % 5 == 0 or (i+1) == len(dataloader): 
            description = f"[train] epoch {epoch} | iteration {i} | time {time.time() - s:.4f} | avg loss {final_loss / (i+1):.6f}"
            pbar.set_description(description)

        torch.cuda.empty_cache()
    final_loss /= len(dataloader)
    
    return final_loss


def valid_fn(model, loss_fn, dataloader, device, epoch):
    """
    1. calculate embedding
    2. for each pair, calculate similarity
    3. using all pair-wise similarities, calculate auc and return it
    """
    model.eval()
    pbar = tqdm(enumerate(dataloader), total=len(dataloader))
    embs = []

    persons = dataloader.dataset.labels
    
    with torch.no_grad():
        for i, (images, labels) in pbar:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model.model(images).detach().cpu()
            embs.append(outputs)
    embs = torch.cat(embs)
    embs = F.normalize(embs)
    embs = embs.numpy()
    valid_auc = calc_auc(embs, persons)    
    print(f"[valid] epoch {epoch} | mean auc {valid_auc:.4f}")
    return valid_auc



def inference_fn(model, dataloader, device):
    
    model.eval()
    pbar = tqdm(enumerate(dataloader), total=len(dataloader))
    embs = []
    with torch.no_grad():
        for i, (images, _) in pbar: # valdloaderでも使えるようにダミー変数
            images = images.to(device)
            outputs = model.model(images).detach().cpu()

            preds = outputs.softmax(1).detach().cpu().numpy()
            preds = preds.argmax(axis=1)
            embs.append(outputs)

    embs = torch.cat(embs)
    embs = F.normalize(embs)
    embs = embs.numpy()
    #valid_auc = calc_auc(embs, persons)    
    #print(f"[valid] epoch {epoch} | mean auc {valid_auc:.4f}")
    return embs
        