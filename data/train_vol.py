import os
from data import srdata


class TRAIN_VOL(srdata.SRData):
    def __init__(self, args, name='TRAIN_VOL', train=True):
        super(TRAIN_VOL, self).__init__(
            args, name=name, train=train
        )

    def _scan(self):
        names_hr = super(TRAIN_VOL, self)._scan()
        print('Training Set Size (Slices):',len(names_hr))
        names_hr = names_hr[self.begin - 1:self.end]
        return names_hr

    def _set_filesystem(self, dir_data):
        super(TRAIN_VOL, self)._set_filesystem(dir_data)
        self.dir_hr = os.path.join(self.apath, 'HR')
