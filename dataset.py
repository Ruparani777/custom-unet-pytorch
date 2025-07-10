import os
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

class CustomDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.image_files = [f for f in os.listdir(data_dir) if 'image' in f]
        self.mask_files = [f for f in os.listdir(data_dir) if 'mask' in f]
        self.transform = transforms.ToTensor()

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image = Image.open(os.path.join(self.data_dir, self.image_files[idx])).convert('RGB')
        mask = Image.open(os.path.join(self.data_dir, self.mask_files[idx])).convert('L')
        return self.transform(image), self.transform(mask)
