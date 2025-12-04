import os
import torch
import numpy as np
from torch.utils.data import Dataset
import pandas as pd
import itertools
from torch.utils.data.sampler import Sampler
import SimpleITK as sitk
from PIL import Image
from torchvision import transforms
from scipy import ndimage
import utils.my_transforms as tr

def load_image_as_nd_array(image_name,mask = True):
    if (image_name.endswith(".nii.gz") or image_name.endswith(".nii") or
        image_name.endswith(".mha")):
        img_obj    = sitk.ReadImage(image_name)
        spacing = img_obj.GetSpacing()
        data_array = sitk.GetArrayFromImage(img_obj)
 
    elif(image_name.endswith(".jpg") or image_name.endswith(".jpeg") or
         image_name.endswith(".tif") or image_name.endswith(".png") or image_name.endswith(".bmp")):
        image = Image.open(image_name)
        data_array = np.asarray(image).copy()
        if mask:
            data_array = np.transpose(data_array, (2, 0, 1))  
            data_array[data_array == 0] = 2
            data_array[data_array == 255] = 0
            data_array[data_array == 128] = 1
            data_array = data_array[0:1]

        else:
            data_array = np.transpose(data_array, (2, 0, 1))  
            data_array = (data_array / 255.0) * 2 - 1 
    else:
        raise ValueError("unsupported image format")
    spacing = [1,1,1]
    return data_array,spacing

class NiftyDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.csv_items  = pd.read_csv(csv_file)
        self.transform = transform
    def __len__(self):
        return len(self.csv_items)

    def __getlabel__(self, idx):
        csv_keys = list(self.csv_items.keys())
        label_idx = csv_keys.index('label')
        label_name = self.csv_items.iloc[idx, label_idx]

        label,_ = load_image_as_nd_array(label_name,mask=True)
        label = np.asarray(label, np.int32)
        return label

    def __getitem__(self, idx):
        names_list, image_list = [], []

        image_name = self.csv_items.iloc[idx, 0]
        image_data,spacing = load_image_as_nd_array(image_name,mask=False)
        names_list.append(image_name)
        image_list.append(image_data)
        image = np.concatenate(image_list, axis = 0)
        image = np.asarray(image, np.float32)    
        sample = {'image': image, 'names' : names_list[0]}
        sample['label'] = self.__getlabel__(idx) 

        if self.transform:
            sample = self.transform(sample)
        sample['names'] = names_list[0] 
        sample['spacing'] = (spacing[2], spacing[1], spacing[0])
        assert(sample['image'].shape[1:] == sample['label'].shape[1:])
        return sample
    
def get_dataset(dataset, domain, online = False):
    if 'mms' in dataset or 'prostate' in dataset:
        transform_train = transforms.Compose([
                            ToTensor(),
                            ])
        transform_valid = transforms.Compose([
                            ToTensor(),
                            ]),
        transform_test = transforms.Compose([
                            ToTensor(),
                            ])
    elif 'Fundus' in dataset or 'polyp' in dataset:
        transform_train = transforms.Compose([
                        tr.Scale_imglab([1,320,320]),
                            ToTensor(),
                            ])
        transform_valid = transforms.Compose([
                        tr.Scale_imglab([1,320,320]),
                            ToTensor(),
                            ]),
        transform_test = transforms.Compose([
                        tr.Scale_imglab([1,320,320]),
                            ToTensor(),
                            ])
    elif '3d' in dataset:
        transform_train = transforms.Compose([
                            ToTensor(),
                            ])
        transform_valid = transforms.Compose([
                            ToTensor(),
                            ]),
        transform_test = transforms.Compose([
                            ToTensor(),
                            ])
    db_train,db_valid,db_test = dataset_all(
        base_dir='./data',
        dataset=dataset,
        target=domain,
        transform_train = transform_train,
        transform_valid = transform_valid,
        transform_test = transform_test,
        online = online)
    return db_train,db_valid,db_test

def dataset_all(base_dir=None, dataset='fb',target='A',transform_train=None,transform_valid=None,transform_test=None,online=False):
    _base_dir = base_dir
    if online:
        all_file = os.path.join(_base_dir,dataset,target,'all.csv')
        all_dataset  = NiftyDataset(csv_file = all_file,transform=transform_train)
        return all_dataset,None,None
    else:
        train_file = os.path.join(_base_dir,dataset,target,'train.csv')
        valid_file = os.path.join(_base_dir,dataset,target,'valid.csv')
        test_file = os.path.join(_base_dir,dataset,target,'test.csv')
        train_dataset  = NiftyDataset(csv_file  = train_file,transform=transform_train)
        valid_dataset  = NiftyDataset(csv_file  = valid_file,transform=transform_valid)
        test_dataset  = NiftyDataset(csv_file  = test_file,transform=transform_test)
        return train_dataset,valid_dataset,test_dataset

class CenterCrop(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        # pad the sample if necessary
        if label.shape[0] <= self.output_size[0] or label.shape[1] <= self.output_size[1] or label.shape[2] <= \
                self.output_size[2]:
            pw = max((self.output_size[0] - label.shape[0]) // 2 + 3, 0)
            ph = max((self.output_size[1] - label.shape[1]) // 2 + 3, 0)
            pd = max((self.output_size[2] - label.shape[2]) // 2 + 3, 0)
            image = np.pad(image, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
            label = np.pad(label, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)

        (w, h, d) = image.shape

        w1 = int(round((w - self.output_size[0]) / 2.))
        h1 = int(round((h - self.output_size[1]) / 2.))
        d1 = int(round((d - self.output_size[2]) / 2.))

        label = label[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
        image = image[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]

        return {'image': image, 'label': label}


class RandomCrop(object):
    """
    Crop randomly the image in a sample
    Args:
    output_size (int): Desired output size
    """

    def __init__(self, output_size, depth = False):
        self.output_size = output_size
        self.depth = depth

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        # pad the sample if necessary
        if label.shape[0] <= self.output_size[0] or label.shape[1] <= self.output_size[1] or label.shape[2] <= \
                self.output_size[2]:
            pw = max((self.output_size[0] - label.shape[0]) // 2 + 3, 0)
            ph = max((self.output_size[1] - label.shape[1]) // 2 + 3, 0)
            pd = max((self.output_size[2] - label.shape[2]) // 2 + 3, 0)
            image = np.pad(image, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
            label = np.pad(label, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)

        (w, h, d) = image.shape
        
        w1 = np.random.randint(0, w - self.output_size[0])
        h1 = np.random.randint(0, h - self.output_size[1])
        d1 = np.random.randint(0, d - self.output_size[2])
        if self.depth:
            label = label[w1:w1 + self.output_size[0], :, :]
            image = image[w1:w1 + self.output_size[0], :, :]
        else:
            label = label[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
            image = image[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
        return {'image': image, 'label': label}

class Scale_img(object):
    """
    Crop randomly the image in a sample
    Args:
    output_size (int): Desired output size
    """

    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        c,h,w = image.shape
        cc,hh,ww = self.output_size
        zoom = [1,hh/h,ww/w]
        image = ndimage.zoom(image,zoom,order=2)

        return {'image': image, 'label': label}

class RandomNoise(object):
    def __init__(self, mu=0, sigma=0.1):
        self.mu = mu
        self.sigma = sigma

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        type_i = image.dtype
        noise = np.clip(self.sigma * np.random.randn(image.shape[0], image.shape[1], image.shape[2]), -2*self.sigma, 2*self.sigma)
        noise = (noise + self.mu)
        image = image + noise
        image = image.astype(type_i)

        return {'image': image, 'label': label}


class CreateOnehotLabel(object):
    def __init__(self, num_classes):
        self.num_classes = num_classes

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        onehot_label = np.zeros((self.num_classes, label.shape[0], label.shape[1], label.shape[2]), dtype=np.float32)
        for i in range(self.num_classes):
            onehot_label[i, :, :, :] = (label == i).astype(np.float32)
        return {'image': image, 'label': label,'onehot_label':onehot_label}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        if len(sample['image'].shape) == 2:
            image = (torch.from_numpy(sample['image'])).unsqueeze(0)
            label = (torch.from_numpy(sample['label'])).unsqueeze(0).long()
        elif len(sample['image'].shape) == 3:
            image = (torch.from_numpy(sample['image']))
            label = (torch.from_numpy(sample['label'])).long()
        if 'onehot_label' in sample:
            return {'image': image, 'label': label,
                    'onehot_label': torch.from_numpy(sample['onehot_label']).long()}
        else:
            return {'image': image, 'label': label}


class TwoStreamBatchSampler(Sampler):
    """Iterate two sets of indices

    An 'epoch' is one iteration through the primary indices.
    During the epoch, the secondary indices are iterated through
    as many times as needed.
    """
    def __init__(self, primary_indices, secondary_indices, batch_size, secondary_batch_size):
        self.primary_indices = primary_indices
        self.secondary_indices = secondary_indices
        self.secondary_batch_size = secondary_batch_size
        self.primary_batch_size = batch_size - secondary_batch_size

        assert len(self.primary_indices) >= self.primary_batch_size > 0
        assert len(self.secondary_indices) >= self.secondary_batch_size > 0

    def __iter__(self):
        primary_iter = iterate_once(self.primary_indices)
        secondary_iter = iterate_eternally(self.secondary_indices)
        return (
            primary_batch + secondary_batch
            for (primary_batch, secondary_batch)
            in zip(grouper(primary_iter, self.primary_batch_size),
                    grouper(secondary_iter, self.secondary_batch_size))
        )

    def __len__(self):
        return len(self.primary_indices) // self.primary_batch_size

def iterate_once(iterable):
    return np.random.permutation(iterable)


def iterate_eternally(indices):
    def infinite_shuffles():
        while True:
            yield np.random.permutation(indices)
    return itertools.chain.from_iterable(infinite_shuffles())


def grouper(iterable, n):
    "Collect data into fixed-length chunks or blocks"
    args = [iter(iterable)] * n
    return zip(*args)

def scale(img,target_size=[1,1,320,416]):
    if len(img.shape) == 3:
        b,h,w = img.shape
    elif len(img.shape) == 4:
        b,c,h,w = img.shape
    bb,cc,hh,ww = target_size
    zoom = [1,hh/h,ww/w]
    img = ndimage.zoom(img,zoom,order=0)
    return img

def convert_2d(img = None,lab = None):
    x_shape = list(img.shape)
    if(len(x_shape) == 5):
        [N, C, D, H, W] = x_shape
        new_shape = [N*D, C, H, W]
        img = torch.transpose(img, 1, 2)
        img = torch.reshape(img, new_shape)
        if lab.shape == img.shape:
            lab = torch.transpose(lab, 1, 2)
            lab = torch.reshape(lab, new_shape)
        else:
            [N, C, D, H, W] = list(lab.shape)
            new_shape = [N*D, C, H, W]
            lab = torch.transpose(lab, 1, 2)
            lab = torch.reshape(lab, new_shape)
            lab = torch.transpose(lab, 1, 2)
            lab = torch.reshape(lab, new_shape)
            
    return img,lab