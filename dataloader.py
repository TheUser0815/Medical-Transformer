import os
from PIL import Image, ImageOps
import torch.utils.data as data
import torchvision.transforms as transforms
import numpy as np
import random
import torch
import os


def get_loader(dataset, batch_size, shuffle=True, num_workers=4, pin_memory=True):

    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=shuffle,
                                  num_workers=num_workers,
                                  pin_memory=pin_memory)
    return data_loader



class FileCollector:
    def __init__(self, base_dir):
        self.base_dir = base_dir
        self.images = []

    def add_data(self, img_dir, mask_dir, subdirs = True):
        self.__collect_image_names(img_dir, mask_dir, subdirs = subdirs)
        self.size = len(self.images)

    def __collect_image_names(self, img_path, mask_path, current = "", subdirs = True):
        path, folders, files=next(os.walk(os.path.join(self.base_dir, img_path, current)))

        #collect images in style (img_path, mask_path, file_path)
        for image in files:
            self.images.append((img_path, mask_path, os.path.join(current, image)))

        #traverse subfolders if requested
        if subdirs:
            for folder in folders:
                self.__collect_image_names(img_path, mask_path, os.path.join(current, folder), subdirs)



class ImageLoader:
    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')



class DSet(data.Dataset, FileCollector, ImageLoader):
    def __init__(self, base_dir, trainsize, augmentations, class_channels):
        FileCollector.__init__(self, base_dir)
        ImageLoader.__init__(self)
        self.trainsize = trainsize
        self.augmentations = augmentations
        self.class_channels = class_channels

        if self.augmentations == 'True':
            print('Using RandomRotation, RandomFlip')
            self.transform = transforms.Compose([
                transforms.RandomRotation(90, resample=False, expand=False, center=None, fill=None),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.Resize((self.trainsize, self.trainsize)),
                transforms.ToTensor()])
            
        else:
            print('no augmentation')
            self.transform = transforms.Compose([
                transforms.Resize((self.trainsize, self.trainsize)),
                transforms.ToTensor()])

    def resize(self, img, gt):
        assert img.size == gt.size
        w, h = img.size
        if h < self.trainsize or w < self.trainsize:
            h = max(h, self.trainsize)
            w = max(w, self.trainsize)
            return img.resize((w, h), Image.BILINEAR), gt.resize((w, h), Image.NEAREST)
        else:
            return img, gt

    def getitem(self, index, img_loader):
        img_data = self.images[index]

        seed = np.random.randint(2147483647) # make a seed with numpy generator 
        
        image = img_loader(os.path.join(self.base_dir, img_data[0], img_data[2]))
        
        random.seed(seed) # apply this seed to img tranfsorms
        torch.manual_seed(seed) # needed for torchvision 0.7
        image = self.transform(image)

        if self.class_channels > 2:
            gt = torch.zeros(image.shape[1:])
            for i in range(self.class_channels):
                gti = self.binary_loader(os.path.join(self.base_dir, img_data[0], str(i), img_data[2]))
                random.seed(seed) # apply this seed to img tranfsorms
                torch.manual_seed(seed) # needed for torchvision 0.7
                gti = self.transform(gti)[0]
                gti *= i
                gt += gti
        else:
            gt0 = self.binary_loader(os.path.join(self.base_dir, img_data[1], img_data[2]))

            random.seed(seed) # apply this seed to img tranfsorms
            torch.manual_seed(seed) # needed for torchvision 0.7
            gt = self.transform(gt0)[0]

            
        gt = torch.round(gt)
        gt = gt.type(torch.LongTensor)

        return image, gt, img_data[2].replace("/", "_")


class MonochromeDataset(DSet):
    """
    dataloader for polyp segmentation tasks
    """
    def __init__(self, base_dir, trainsize, augmentations, class_channels=1):
        super().__init__(base_dir, trainsize, augmentations, class_channels)

    def __getitem__(self, index):
        return super().getitem(index, super().binary_loader)

    def __len__(self):
        return self.size
    

class PolychromeDataset(DSet):
    """
    dataloader for polyp segmentation tasks
    """
    def __init__(self, base_dir, trainsize, augmentations, class_channels=1):
        super().__init__(base_dir, trainsize, augmentations, class_channels)

    def __getitem__(self, index):
        return super().getitem(index, super().rgb_loader)

    def __len__(self):
        return self.size



class TestDataset(FileCollector, ImageLoader):
    def __init__(self, base_dir):
        FileCollector.__init__(self, base_dir)
        ImageLoader.__init__(self)
        self.class_channels = class_channels
        self.testsize = testsize
        self.transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor()])
        self.gt_transform = transforms.ToTensor()
        self.re_init()

    def re_init(self):
        self.index = 0
        np.random.shuffle(self.images)

    def next(self, img_loader):
        img_data = self.images[self.index]

        name = img_data[2].split('/')[-1]
        self.index += 1
        
        image = img_loader(os.path.join(self.base_dir, img_data[0], img_data[2]))
        image = self.transform(image).unsqueeze(0)

        if self.class_channels > 2:
            gt = []
            for i in range(self.class_channels):
                gti = self.binary_loader(os.path.join(self.base_dir, img_data[0], str(i), img_data[2]))
                gti = self.gt_transform(gti)
                gt.append(gti)
            gt = torch.stack(gt, 1)
        else:
            gt0 = self.binary_loader(os.path.join(self.base_dir, img_data[1], img_data[2]))

            gt = self.gt_transform(gt0)

            if self.class_channels == 2:
                gt1 = ImageOps.invert(gt0)

                gt1 = self.gt_transform(gt1)

                gt = torch.stack([gt,gt1], 1)

        return image, gt, name


class MonochromeTestDataset(TestDataset):
    def __init__(self, base_dir, testsize, class_channels):
        super().__init__(base_dir, testsize, class_channels)

    def load_data(self):
        super().next(super().binary_loader)

    def __len__(self):
        return self.size


class PolychromeTestDataset(TestDataset):
    def __init__(self, base_dir, testsize, class_channels):
        super().__init__(base_dir, testsize, class_channels)

    def load_data(self):
        super().next(super().rgb_loader)

    def __len__(self):
        return self.size