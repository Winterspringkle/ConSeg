import os
import volumentations as V
import numpy as np
import torch
from PIL import Image
import torchvision.transforms as transforms
from torch.utils import data
import nibabel as nib
class Hecktor_train(data.Dataset):
    def __init__(self, image_dir, label_dir,npy_dir):
        """Initialize and preprocess the CelebA dataset."""
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.add_part=np.load("./cross_datalist/test_data2.npy",allow_pickle=True)
        self.id_list=np.concatenate((np.load(npy_dir,allow_pickle=True),self.add_part[:int(0.4*len(self.add_part))]),axis=0)
        # self.id_list =np.load(npy_dir,allow_pickle=True)
        self.num_images=len(self.id_list)
        print(self.num_images)
        self.PET_data_list = [os.path.join(image_dir,i+"__PT.nii.gz") for i in self.id_list]
        self.CT_data_list = [os.path.join(image_dir,i+"__CT.nii.gz") for i in self.id_list]
        self.label_list = [os.path.join(label_dir,i+".nii.gz") for i in self.id_list]
        self.aug = get_augmentation_train((112,112,112))


    def __getitem__(self, index):
        """Return one image and its corresponding attribute label."""
        PET_data_list = self.PET_data_list
        CT_data_list = self.CT_data_list
        label_list = self.label_list
        PET_filename= PET_data_list[index]
        CT_filename = CT_data_list[index]
        label_filename = label_list[index]
        PET_X = np.expand_dims(nib.load(PET_filename).get_data(),0)
        # PET_X=(PET_X-np.min(PET_X))/(np.max(PET_X)-np.min(PET_X))
        PET_X = (PET_X - np.mean(PET_X)) / (np.std(PET_X))
        CT_X = np.expand_dims(nib.load(CT_filename).get_data(),0)
        CT_X[CT_X>1024]=1024
        CT_X[CT_X<-1024]=-1024
        CT_X = (CT_X - np.mean(CT_X)) / ((np.max(CT_X) - np.min(CT_X))/2)
        label_X = np.expand_dims(nib.load(label_filename).get_data(),0)
        data={"image":np.concatenate((PET_X,CT_X),axis=0)}
        aug_data=self.aug(**data)
        PET_X=np.expand_dims(aug_data["image"][0],0)
        CT_X = np.expand_dims(aug_data["image"][1],0)
        return PET_X,CT_X,label_X
    def __len__(self):
        """Return the number of images."""
        return self.num_images
class Hecktor(data.Dataset):
    def __init__(self, image_dir, label_dir,npy_dir):
        """Initialize and preprocess the CelebA dataset."""
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.id_list=np.load(npy_dir,allow_pickle=True)
        self.num_images=len(self.id_list)
        self.PET_data_list = [os.path.join(image_dir,i+"__PT.nii.gz") for i in self.id_list]
        self.CT_data_list = [os.path.join(image_dir,i+"__CT.nii.gz") for i in self.id_list]
        self.label_list = [os.path.join(label_dir,i+".nii.gz") for i in self.id_list]


    def __getitem__(self, index):
        """Return one image and its corresponding attribute label."""
        PET_data_list = self.PET_data_list
        CT_data_list = self.CT_data_list
        label_list = self.label_list
        PET_filename= PET_data_list[index]
        CT_filename = CT_data_list[index]
        label_filename = label_list[index]
        PET_X = np.expand_dims(nib.load(PET_filename).get_data(),0)
        # PET_X=(PET_X-np.min(PET_X))/(np.max(PET_X)-np.min(PET_X))
        PET_X = (PET_X - np.mean(PET_X)) / (np.std(PET_X))
        CT_X = np.expand_dims(nib.load(CT_filename).get_data(),0)
        # CT_X[CT_X>1024]=1024
        # CT_X[CT_X<-1024]=-1024
        CT_X = (CT_X - np.mean(CT_X)) / ((np.max(CT_X) - np.min(CT_X))/2)
        label_X = np.expand_dims(nib.load(label_filename).get_data(),0)
        # print(PET_X.shape)
        # PET_X = torch.Tensor(PET_X)
        # CT_X = torch.Tensor(CT_X)
        # label_X = torch.Tensor(label_X)
        # PET_noise = torch.normal(mean=torch.zeros(PET_X.size()), std=1.)
        # CT_noise = torch.normal(mean=torch.zeros(CT_X.size()), std=1.)
        # label_noise = torch.normal(mean=torch.zeros(label_X.size()), std=1.)
        # PET_X=PET_X+PET_noise
        # CT_X = CT_X + CT_noise
        # label_X = label_X + label_noise
        # data={"image":np.concatenate((PET_X,CT_X),axis=0)}
        # aug_data=self.aug(**data)
        # PET_X=np.expand_dims(aug_data["image"][0],0)
        # CT_X = np.expand_dims(aug_data["image"][1],0)
        return PET_X,CT_X,label_X
    def __len__(self):
        """Return the number of images."""
        return self.num_images
def get_augmentation_train(patch_size):
    return V.Compose([
        # V.Rotate((-8, 8), (-8, 8), (-8, 8), p=0.5),
        # V.RandomCrop(shape = (160, 160, 64), p = 1.0),
        # V.ElasticTransform((0, 0.25), interpolation=2, p=0.1),
        V.GaussianNoise(var_limit=(0, 5), p=0.2),
        # V.RandomGamma(gamma_limit=(0.5, 1.5), p=0.2),
    ], p=1.0)