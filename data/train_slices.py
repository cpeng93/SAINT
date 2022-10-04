import os
from data import srdata


class TRAIN_SLICES(srdata.SRData):
    def __init__(self, args, name='TRAIN_SLICES', train=True):
        super(TRAIN_SLICES, self).__init__(
            args, name=name, train=train
        )

    def _scan(self):
        names_hr = super(TRAIN_SLICES, self)._scan()
        print('Training Set Size (Slices):',len(names_hr))
        names_hr = names_hr[self.begin - 1:self.end]
        return names_hr

    def _set_filesystem(self, dir_data):
        super(TRAIN_SLICES, self)._set_filesystem(dir_data)
        self.dir_hr = os.path.join(self.apath, 'HR')
