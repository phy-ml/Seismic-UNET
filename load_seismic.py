import numpy as np
from torch.utils.data import Dataset
import os

# path for train data
train_seis = r'C:\Users\Syahrir Ridha\PycharmProjects\DiffEncoder\diffencoder\data\train-20210921T145844Z-001\train\seis'
train_fault = r'C:\Users\Syahrir Ridha\PycharmProjects\DiffEncoder\diffencoder\data\train-20210921T145844Z-001\train\fault\fault'

# path for test data
test_seis = r'C:\Users\Syahrir Ridha\PycharmProjects\DiffEncoder\diffencoder\data\validation-20210921T145852Z-001\validation\seis'
test_fault = r'C:\Users\Syahrir Ridha\PycharmProjects\DiffEncoder\diffencoder\data\validation-20210921T145852Z-001\validation\fault'

# # path for pred data
# pred_sei =
# pred_fault

class Seismic_Dataset(Dataset):
    def __init__(self, data=None, label=None, transform=None):
        #         if data or label is None:
        #             raise ValueError('Enter the file path')

        self.x_dir = data
        self.y_dir = label
        self.transform = transform
        self.x_total = os.listdir(self.x_dir)
        self.y_total = os.listdir(self.y_dir)

    def __len__(self):
        return len(self.x_total)

    def __getitem__(self, idx):
        #         x_path = os.path.join(self.x_dir, self.x_total[idx])
        #         y_path = os.path.join(self.y_dir, self.x_total[idx])
        #         image = np.fromfile(x_path,dtype=np.single).reshape(128,128,128)
        #         mask = np.fromfile(y_path,dtype=np.single).reshape(128,128,128)
        image, mask = self.process_data(idx)

        if self.transform is not None:
            augmentation = self.transform(image=image, mask=mask)
            image = augmentation['image']
            mask = augmentation['mask']

        return image, mask

    def process_data(self, idx):
        x_path = os.path.join(self.x_dir, self.x_total[idx])
        y_path = os.path.join(self.y_dir, self.x_total[idx])
        image = np.fromfile(x_path, dtype=np.single)  # .reshape(-1,128,128,128)
        mask = np.fromfile(y_path, dtype=np.single)  # .reshape(-1,128,128,128)

        # reshape
        image = np.reshape(image, (128, 128, 128))
        mask = np.reshape(mask, (128, 128, 128))

        # find mean, std dev of image
        xm = np.mean(image)
        xs = np.std(image)
        image = (image - xm) / xs
        image = np.transpose(image)
        mask = np.transpose(mask)

        # other processings
        # X = np.zeros((1, 2, 128, 128, 128), dtype=np.single)
        # Y = np.zeros((1, 2, 128, 128, 128), dtype=np.single)
        # X[:, 0, :, :, :] = np.reshape(image, (-1, 128, 128, 128))
        # Y[:, 0, :, :, :] = np.reshape(mask, (-1, 128, 128, 128))
        # X[:, 1, :, :, :] = np.reshape(np.flipud(image), (-1,128, 128, 128))
        # Y[:, 1, :, :, :] = np.reshape(np.flipud(mask), (-1,128, 128, 128))
        X = np.zeros((2, 128, 128, 128), dtype=np.single)
        Y = np.zeros((2, 128, 128, 128), dtype=np.single)
        X[ 0, :, :, :] = np.reshape(image, ( 128, 128, 128))
        Y[ 0, :, :, :] = np.reshape(mask, ( 128, 128, 128))
        X[ 1, :, :, :] = np.reshape(np.flipud(image), ( 128, 128, 128))
        Y[ 1, :, :, :] = np.reshape(np.flipud(mask), ( 128, 128, 128))

        return X, Y

if __name__ == '__main__':
    dataset = Seismic_Dataset(data=train_seis, label=train_fault)
    print(dataset.__len__())
    print(dataset.__getitem__(10)[0].shape)
