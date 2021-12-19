import cv2
import paddle
import numpy as np
import pandas as pd

from utils import read_image, get_all_paths, rgb2gray
from configs import INPUT_IMAGE_SHAPE, OUTPUT_RANGE

def post_processing(img, pos):
    img_gray = rgb2gray(img)
    ksize = 11
    sigma = 0.3*((ksize-1)*0.5-1)+0.8
    img_gray = cv2.GaussianBlur(img_gray, ksize=(ksize, ksize), sigmaX=sigma, sigmaY=sigma)
    x0, y0 = int(pos[0]), int(pos[1])
    wsize = 100
    best_v = 255
    for x in range(x0-wsize, x0+wsize):
        for y in range(y0-wsize, y0+wsize):
            if best_v>img_gray[y,x]:
                best_v = img_gray[y,x]
                pos = (x, y)
    return pos

def crop_center(img, loc=None):
    h, w, d = img.shape
    dx = (w-h)//2
    img = img[:, dx:(dx+h), :]
    img = cv2.resize(img, (INPUT_IMAGE_SHAPE, INPUT_IMAGE_SHAPE))
    scale = h/INPUT_IMAGE_SHAPE
    if loc is not None:
        x, y = loc
        loc = ((x-dx)/h, y/h)
    return img, loc, (scale, dx)

def restore_xy(loc, param):
    x, y = loc
    scale, dx = param
    loc = (x*INPUT_IMAGE_SHAPE*scale+dx, y*INPUT_IMAGE_SHAPE*scale)
    return loc

def from_pred(reg, param):
    # cls, reg = cls.numpy(), reg.numpy()
    r0, r1 = OUTPUT_RANGE
    # s_x, s_y, n = 0, 0, 0
    # for hi in range(OUTPUT_SIZE):
    #     for wi in range(OUTPUT_SIZE):
    #         if cls[0,hi,wi]>0:
    #             h = hi+reg[0,hi,wi].clip(0, 1)
    #             w = wi+reg[1,hi,wi].clip(0, 1)
    #             h = h*(r1-r0)/OUTPUT_SIZE+r0
    #             w = w*(r1-r0)/OUTPUT_SIZE+r0
    #             x, y = restore_xy((w,h), param)
    #             s_x += x
    #             s_y += y
    #             n += 1
    # if n==0:
    h = reg[0].clip(0, 1)
    w = reg[1].clip(0, 1)
    h = h*(r1-r0)+r0
    w = w*(r1-r0)+r0
    s_x, s_y = restore_xy((w,h), param)
    return s_x, s_y


class MyDataset(paddle.io.Dataset):
    def __init__(self, img_folder, label_file=None, idx=None, argument=False):
        img_paths = get_all_paths(img_folder, '.jpg')
        if idx is None:
            idx = np.arange(len(img_paths))
        imgs = [read_image(img_paths[i]) for i in idx]
        if label_file:
            label_df = pd.read_excel(label_file)
            label_X = label_df['Fovea_X'].values
            label_Y = label_df['Fovea_Y'].values
            labels = [(label_X[i], label_Y[i]) for i in idx]
        else:
            labels = [None,]*len(idx)
        datas = [(img, label) for img, label in zip(imgs, labels)]
        self.datas = datas
        self.argument = argument
        self.color_jitter = paddle.vision.transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)

    def __getitem__(self, idx):
        img, loc = self.datas[idx]
        img = img.copy()
        if self.argument:
            img, loc = self.do_arg(img, loc)
        img, label, param = self.pre_process(img, loc)
        return img, label, param

    def __len__(self):
        return len(self.datas)

    def do_arg(self, img, loc):
        h, w, _ = img.shape
        img = self.color_jitter(img)
        if np.random.uniform(0,1)>0.5:
            # horizon flip
            img[:,::-1,:] = img
            loc = (w-loc[0], loc[1])
        if np.random.uniform(0,1)>0.5:
            # vertical flip
            img[::-1,:,:] = img
            loc = (loc[0], h-loc[1])
        return img, loc

    def pre_process(self, img, loc):
        img, loc, param = crop_center(img, loc)
        img = img.transpose([2,0,1]).astype(np.float32)/255-0.5
        label = np.zeros([2], dtype=np.float32)
        if loc is not None:
            fx, fy = loc
            r0, r1 = OUTPUT_RANGE
            w = (fx-r0)/(r1-r0)
            h = (fy-r0)/(r1-r0)
            label[:] = (h, w)
        return img, label, param


def collate_fn(datas):
    imgs = []
    labels = []
    params = []
    for img, label, param in datas:
        imgs.append(img)
        if label is not None:
            labels.append(label)
        params.append(param)
    return paddle.to_tensor(imgs), paddle.to_tensor(labels), params


if __name__=='__main__':
    import warnings
    warnings.filterwarnings('ignore')

    from configs import TRAIN_IMAGE_FOLDER, TRAIN_LABEL_PATH, TEST_IMAGE_FOLDER, BATCH_SIZE

    train_dataset = MyDataset(TRAIN_IMAGE_FOLDER, TRAIN_LABEL_PATH, idx=np.arange(10), argument=False)
    img, reg, param = train_dataset[0]
    print(img.shape, reg.shape, param)
    print(reg)
    test_dataset = MyDataset(TEST_IMAGE_FOLDER, idx=np.arange(10), argument=False)
    img, _, param = test_dataset[0]
    print(img.shape, param)

    label_df = pd.read_excel(TRAIN_LABEL_PATH)
    label_X = label_df['Fovea_X'].values
    label_Y = label_df['Fovea_Y'].values
    train_loader = paddle.io.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, collate_fn=collate_fn)
    test_loader = paddle.io.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, collate_fn=collate_fn)
    i = 0
    for img, label, params in train_loader:
        for reg, param in zip(label, params):
            print([label_X[i], label_Y[i]], reg.numpy(), param)
            x, y = from_pred(reg.numpy(), param)
            xt, yt = label_X[i], label_Y[i]
            i += 1
            print(i, x, xt, y, yt)

    # for img, (cls, reg), params in test_loader:
    #     print(img.shape, cls.shape, reg.shape, params)



