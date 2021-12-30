import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import MNIST



class CachedMNIST(Dataset):
    def __init__(self, data_path, train, cuda, testing_mode=False):
        img_transform = transforms.Compose([transforms.Lambda(self._transformation)])
        self.ds = MNIST(data_path, download=True, train=train, transform=img_transform)
        self.cuda = cuda
        self.testing_mode = testing_mode
        self._cache = dict()

    @staticmethod
    def _transformation(img):
        return (torch.ByteTensor(torch.ByteStorage.from_buffer(img.tobytes())).float() * 0.02)

    def __getitem__(self, index):
        if index not in self._cache:
            self._cache[index] = list(self.ds[index])
            if self.cuda:
                self._cache[index][0] = self._cache[index][0].cuda(non_blocking=True)
                self._cache[index][1] = torch.tensor(self._cache[index][1], 
                                                     dtype=torch.long).cuda(non_blocking=True)
        return self._cache[index]

    def __len__(self) -> int:
        return 128 if self.testing_mode else len(self.ds)
