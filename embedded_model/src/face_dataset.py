import os
from torch.utils.data import Dataset, Subset
from PIL import Image


class FaceDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = sorted(os.listdir(root_dir))
        self.class2idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.images, self.labels = self.get_imgs_labels()

    def __len__(self):
        return len(self.images)

    def get_imgs_labels(self):
        images, labels = [], []
        folders = os.listdir(self.root_dir)
        for fold in folders:
            fold_path = os.path.join(self.root_dir, fold)
            for img in os.listdir(fold_path):
                if img.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
                    img_path = os.path.join(fold_path, img)
                    images.append(img_path)
                    labels.append(self.class2idx[fold])
        return images, labels

    def __getitem__(self, idx):
        img = self.images[idx]
        label = self.labels[idx]
        img = Image.open(img).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


class SubsetFaceDataset(Dataset):
    def __init__(self, subset: Subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        img, cls = self.subset[idx]
        if self.transform:
            img = self.transform(img)
        return img, cls
