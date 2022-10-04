import os
import glob
import numpy as np
from data import common
import pickle
import torch.utils.data as data
import random

class SRData(data.Dataset):
    def __init__(self, args, name='', train=True):
        self.args = args
        self.name = name
        self.train = train
        self.split = 'train' if train else 'test'
        self.do_eval = True
        self.scale = args.scale
        self.idx_scale = 0
        data_range = [r.split('-') for r in args.data_range.split('/')]
        if train:
            data_range = data_range[0]
        else:
            if args.test_only and len(data_range) == 1:
                data_range = data_range[0]
            else:
                data_range = data_range[1]
        self.begin, self.end = list(map(lambda x: int(x), data_range))
        self._set_filesystem(args.dir_data)

        path_bin = self.apath
        os.makedirs(path_bin, exist_ok=True)

        list_hr = self._scan()
        os.makedirs(
            self.dir_hr.replace(self.apath, path_bin),
            exist_ok=True
        )

        self.images_hr = []
        for h in list_hr:
            if not self.train:
                            b = h.replace(self.apath, path_bin)
                            self.images_hr.append(b)
            else:
                    b = h.replace(self.apath, path_bin)
                    self.images_hr.append(b)

        if train:
            self.repeat = 1

    # Below functions as used to prepare images
    def _scan(self):
        names_hr = sorted(
            glob.glob(os.path.join(self.dir_hr, '*.pt'))
        )
        return names_hr

    def _set_filesystem(self, dir_data):
        self.apath = os.path.join(dir_data, self.name)
        self.dir_hr = os.path.join(self.apath, 'HR')
        self.ext = ('.png', '.png')

    def __getitem__(self, idx):
        lr, hr, dist, filename = self._load_file(idx)
        if self.train:
            lr, hr = self.get_patch(lr, hr)
        lr_tensor, hr_tensor, dist_tensor = common.np2Tensor(
            lr, hr, dist
        )
        return lr_tensor, hr_tensor, dist_tensor, filename

    def __len__(self):
        if self.train:
            return len(self.images_hr) * self.repeat
        else:
            return len(self.images_hr)

    def _get_index(self, idx):
        if self.train:
            return idx % len(self.images_hr)
        else:
            return idx

    def _load_file(self, idx):
        idx = self._get_index(idx)
        f_hr = self.images_hr[idx]
        filename, _ = os.path.splitext(os.path.basename(f_hr))
        with open(f_hr, 'rb') as _f: hr = pickle.load(_f)
        if self.train:
            factor = random.choice([4,6])
            hr, spacing = self.gen_diff_thickness(hr)
            if not hr.shape[-1]%factor == 0:
                hr = hr[...,hr.shape[-1]%factor:]
            dist = self.get_dist(spacing, factor)[0]
            if self.train:
                return hr[...,::factor], self.get_hr(hr[1],factor), dist, filename
        else:
            factor = int(self.scale[0])
            hr, spacing = hr['image'].astype('int32'), hr['spacing']
            if not hr.shape[-1]%factor == 0:
                hr = hr[...,hr.shape[-1]%factor:]       
            sag = hr
            cor = np.transpose(sag, (1,0,2))
            out = np.zeros((2, 3, 512, 512, sag.shape[-1]))
            out[0,0,1:] = sag[:sag.shape[0]-1] 
            out[0,1] = sag
            out[0,2,:-1] = sag[1:]

            out[1,0,1:] = cor[:cor.shape[0]-1]
            out[1,1] = cor
            out[1,2,:-1] = cor[1:]
            out = out[...,::factor].transpose(0,2,1,3,4)
            dist = self.get_dist(spacing, factor)[0]
            return out, hr, dist, filename

    def gen_diff_thickness(self, data):
        ## this will decide the downsampling rate and the random starting point, while making sure that
        ## the input is no smaller than 60 in slice dimension, thickness generates up to 5mm
        spacing = list(data['spacing'])
        image = data['image']
        upper = 5//spacing[2]

        # print(image.shape[2],upper)
        if upper < 1:
            upper = 1
        while image.shape[2]//upper < 60:
            upper = upper - 1
        dwn_rate = random.randint(1,upper)
        spacing[2] = spacing[2]*dwn_rate
        starting = random.randint(0, dwn_rate-1)
        image = image[...,starting:][...,::dwn_rate]
        return image.astype('int32'), spacing


    def get_hr(self, data, factor):
        output = np.zeros((6, data.shape[0], data.shape[1]//factor)).astype('int32')
        for i in range(factor):
            output[i] = data[...,i::factor]
        return output

    def get_dist(self, spacing, factor):
        x, y = spacing[1], spacing[2]
        kernel = 3
        FDM = np.zeros((factor,kernel,kernel,2))

        for k in range(factor):
            for i in range(kernel):
                for j in range(kernel):
                    FDM[k][i][j] = [k+factor*i,j]


        dist = np.zeros((6,kernel,kernel))
        center = FDM[0][kernel//2][kernel//2]
        for k in range(factor):
            for i in range(kernel):
                for j in range(kernel):
                    ind_dist = abs(FDM[k][i][j] - center)
                    ind_dist[0]*=y
                    ind_dist[1]*=x
                    dist[k][i][j] = np.linalg.norm(ind_dist, ord=2)
        dist = dist[:,::-1] 
        return np.expand_dims(dist,0)



    def get_patch(self, lr, hr):
        scale = self.scale[self.idx_scale]
        # note that lr should be of smaller size than hr, unless scale is 1
        if self.train:
            lr, hr = common.get_patch_x_side(
                lr,
                hr,
                patch_size=self.args.patch_size,
                scale=scale
            )
        else:
            ih, iw = lr.shape[:2]
            hr = hr[0:ih * scale, 0:iw * scale]

        return lr, hr

    def set_scale(self, idx_scale):
        self.idx_scale = idx_scale

