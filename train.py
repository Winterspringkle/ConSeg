# py imports
import os
import glob
import sys
import random
import time
import torch
import cv2
import numpy as np
from argparse import ArgumentParser
from lifelines.utils import concordance_index
import torch.utils.data as DataSet
# project imports
from data_generator_modify import *
from model import ConSeg
import losses
from tqdm import tqdm
from sklearn.manifold import TSNE
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from evaluation import *
def get_DC(SR, GT, threshold=0.5):
    # DC : Dice Coefficient
    SR = SR > threshold
    GT = GT == torch.max(GT)

    SR = SR.int()
    GT = GT.int()
    Inter = torch.sum((SR + GT) == 2)
    DC = float(2 * Inter) / (float(torch.sum(SR) + torch.sum(GT)) + 1e-6)

    return DC

def lr_scheduler(epoch):
    lr = 1e-3
    if epoch < 80:
        lr = 1e-3
    # elif epoch < 40:
    #     lr = 5e-4
    # elif epoch < 60:
    #     lr = 1e-4
    else:
        lr = 1e-4
    return lr


def train(data_dir,
          train_samples,
          valid_samples,
          model_dir,
          load_model,
          device,
          initial_epoch,
          epochs,
          steps_per_epoch,
          batch_size):
    image_dir = ""
    label_dir = ""
    mode = "train"

    BATCH_SIZE = 4
    # prepare model folder
    model_name="our"
    if not os.path.isdir(model_dir):
        os.mkdir(model_dir)

    # device handling
    if 'gpu' in device:
        os.environ['CUDA_VISIBLE_DEVICES'] = device[-1]
        device = 'cuda'
        torch.backends.cudnn.deterministic = True
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        device = 'cpu'
    loss_list=[]
    total_dice_list=[]
    # prepare the model
    model = ConSeg()
    train_npy_dir="./cross_datalist/train_data2.npy"
    test_npy_dir = "./cross_datalist/test_data2.npy"

    if load_model != './':
        print('loading', load_model)
        state_dict = torch.load(load_model, map_location=device)
        model.load_state_dict(state_dict)
    model.to(device)
    # set optimizer
    optimizer = torch.optim.Adam(model.parameters())
    # prepare losses
    criterion3 = losses.structure_loss()
    # data generator
    train_dataset = Hecktor_train(image_dir, label_dir,train_npy_dir)
    train_loader = DataSet.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    test_dataset = Hecktor(image_dir, label_dir,test_npy_dir)
    test_loader = DataSet.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    print("train ",len(train_dataset),"test ",len(test_dataset))
    init_DC=0
    # training/validation loops
    consine_loss = torch.nn.CosineEmbeddingLoss(margin=0.0, reduction='mean')
    if mode=="train":
        for epoch in range(initial_epoch, epochs):
            start_time = time.time()

            # adjust lr
            lr = lr_scheduler(epoch)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            # training
            model.train()
            train_DC = []
            train_total_loss = []
            for PET, CT, label in tqdm(train_loader):
                PET = PET.to(device=device, dtype=torch.float32)
                CT = CT.to(device=device, dtype=torch.float32)
                label = label.to(device=device, dtype=torch.float32)
                # generate inputs (and true outputs) and convert them to tensors
                # run inputs through the model to produce a warped image and flow field
                pred, a_map, cat_pred, p1, p2, p3, p4,pet_emb,ct_emb = model(PET, CT)
                c_loss = consine_loss(pet_emb, ct_emb, torch.tensor([1.]).cuda())
                curr_loss = criterion3(pred, label)
                curr_loss1 = criterion3(a_map, label)
                loss_cat = criterion3(cat_pred, label)
                loss1 = criterion3(p1, label)
                loss2 = criterion3(p2, label)
                loss3 = criterion3(p3, label)
                loss4 = criterion3(p4, label)
                loss = curr_loss + curr_loss1 + c_loss + 0.5 * (loss_cat+loss1 + loss2 + loss3 + loss4)
                train_total_loss.append(curr_loss.item())
                # backpropagate and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                DC=get_DC(pred,label)
                train_DC.append(DC)

            # validation
            model.eval()
            test_DC=[]
            for PET, CT, label in tqdm(test_loader):
                # generate inputs (and true outputs) and convert them to tensors
                PET = PET.to(device=device, dtype=torch.float32)
                CT = CT.to(device=device, dtype=torch.float32)
                label = label.to(device=device, dtype=torch.float32)

                # run inputs through the model to produce a warped image and flow field
                with torch.no_grad():
                    pred,a_map,cat_pred,p1,p2,p3,p4,pet_emb,ct_emb = model(PET, CT)
                DC = get_DC(pred, label)
                test_DC.append(DC)
            print("TRAIN LOSS ", np.mean(np.array(train_total_loss)))
            print("TRAIN DC ",np.mean(np.array(train_DC)))
            print("TEST DC ", np.mean(np.array(test_DC)))
            if np.mean(np.array(test_DC))>init_DC:
                init_DC=np.mean(np.array(test_DC))
                # save model checkpoint
                torch.save(model.state_dict(), os.path.join(model_dir, "our.pt"))
        print("DC", total_dice_list, np.mean(np.array(total_dice_list)))
    if mode =="test":
        model.load_state_dict(
            torch.load("./models/" + model_name + ".pt"))
        # 测试模式
        test_DC = []
        test_SE = []
        # test_HD = []
        test_JS = []
        PET_list=[]
        CT_list = []
        fuse_list=[]
        for PET, CT, label in tqdm(test_loader):
            # generate inputs (and true outputs) and convert them to tensors
            PET = PET.to(device=device, dtype=torch.float32)
            CT = CT.to(device=device, dtype=torch.float32)
            label = label.to(device=device, dtype=torch.float32)
            # run inputs through the model to produce a warped image and flow field
            with torch.no_grad():
                pred, a_map, cat_pred, p1, p2, p3, p4,pet_emb,ct_emb,PET,CT,fuse = model(PET, CT)
            DC = get_DC(pred, label)
            # HD = get_HD(pred, label)
            SE = get_sensitivity(pred, label)
            JS = get_JS(pred, label)
            test_DC.append(DC)
            # test_HD.append(HD)
            test_SE.append(SE)
            test_JS.append(JS)
            for i in PET:
                PET_list.append(i.detach().cpu().numpy())
            for i in CT:
                CT_list.append(i.detach().cpu().numpy())
            for i in fuse:
                fuse_list.append(i.detach().cpu().numpy())
        final_DC=np.mean(test_DC)
        final_SE = np.mean(test_SE)
        final_JS = np.mean(test_JS)
        print("DC", final_DC,"JS", final_JS,"SE", final_SE)

if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--data_dir", type=str,
                        dest="data_dir", default='',
                        help="data folder")
    parser.add_argument("--train_samples", type=str,
                        dest="train_samples", default='./train.npy',
                        help="training samples")
    parser.add_argument("--valid_samples", type=str,
                        dest="valid_samples", default='./train.npy',
                        help="validation samples")
    parser.add_argument("--model_dir", type=str,
                        dest="model_dir", default='./models/',
                        help="models folder")
    parser.add_argument("--load_model", type=str,
                        dest="load_model", default='./',
                        help="load model file to initialize with")
    parser.add_argument("--device", type=str, default='gpu0',
                        dest="device", help="gpuN or multi-gpu")
    parser.add_argument("--initial_epoch", type=int,
                        dest="initial_epoch", default=0,
                        help="initial_epoch")
    parser.add_argument("--epochs", type=int,
                        dest="epochs", default=100,
                        help="number of epoch")
    parser.add_argument("--steps_per_epoch", type=int,
                        dest="steps_per_epoch", default=200,
                        help="iterations of each epoch")
    parser.add_argument("--batch_size", type=int,
                        dest="batch_size", default=2,
                        help="batch_size")

    args = parser.parse_args()
    train(**vars(args))
