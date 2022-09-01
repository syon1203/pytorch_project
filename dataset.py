import os
import matplotlib.image as img
from torch.utils.data import Dataset
import torch
from PIL import Image
import numpy as np

def get_labels(root):
    label = os.walk(root).__next__()[1]
    #labels = {string for i, string in enumerate(label)}
    return label


def match_image(root):
    images = []
    labels = {}

    a = os.listdir(root)
    if '.DS_Store' in a:
        a.remove('.DS_Store')

    for i, label in enumerate(a):
        #print(labels)
        labels[i] = i
        #print(labels[i])
        try:
            #print(label)
            #print(i)
            for j in os.listdir(os.path.join(root, label)):
                image = img.imread(os.path.join(root, label,j))
                images.append((i, image))
        except:
            pass
    print("finished")
    return images


class CIFAR10(Dataset):

    def __init__(self, root, train=True, transform=None):
        super(CIFAR10, self).__init__()
        self.root = root
        self.labels = get_labels(root)
        self.images = match_image(root)
        self.transform = transform

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()

        imgarr = np.random.random_sample(self.images[index][1].shape) * 255
        imgarr = imgarr.astype(np.uint8)
        image = Image.fromarray(imgarr).convert('RGB')
        label = self.images[index][0]
        #image = Image.fromarray(np.uint8(img))

        if self.transform:
            image = self.transform(image)


        return image, label

    def __len__(self):
        return len(self.images)

