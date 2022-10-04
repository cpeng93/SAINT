import os
from data import srdata


class TEST(srdata.SRData):
    def __init__(self, args, name='TEST', train=True,):
        super(TEST, self).__init__(
            args, name=name, train=train
        )

    def _scan(self):
        names_hr = super(TEST, self)._scan()
        print('Testing Set Size (Volumes):',len(names_hr))
        names_hr = names_hr[self.begin - 1:self.end]
        return names_hr

    def _set_filesystem(self, dir_data):
        super(TEST, self)._set_filesystem(dir_data)
        self.dir_hr = os.path.join(self.apath, 'HR')

