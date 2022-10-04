# arg 1 input dir, arg2 output dir
from medpy.io import load
import sys, os, pickle
import numpy as np

test_set = pickle.load(open('test_set.pt','rb'))
# print(test_set.keys())
os.makedirs(sys.argv[2],exist_ok=True)
os.makedirs(os.path.join(sys.argv[2],'TEST'),exist_ok=True)
os.makedirs(os.path.join(sys.argv[2],'TEST/HR'),exist_ok=True)
os.makedirs(os.path.join(sys.argv[2],'TRAIN_VOL'),exist_ok=True)
os.makedirs(os.path.join(sys.argv[2],'TRAIN_VOL/HR'),exist_ok=True)
os.makedirs(os.path.join(sys.argv[2],'TRAIN_SLICES'),exist_ok=True)
os.makedirs(os.path.join(sys.argv[2],'TRAIN_SLICES/HR'),exist_ok=True)

for inst in ['imagesTr','imagesTs']:
    file_dir = os.path.join(sys.argv[1], inst)
    patients = os.listdir(file_dir)
    for patient in patients:
        if not '._' in patient:
            img, header = load(os.path.join(file_dir, patient))
            if img.max().astype(float) - img.min().astype(float) > 9000:
                print("possible metal artifact: " + patient + " , HU: range" + str(img.max().astype(float) - img.min().astype(float)))
                continue
            spacing = header.get_voxel_spacing()
            img = np.clip(img,-1024,img.max())
            img = img - img.min()
            img = img.astype("uint16")
            data = {'image': img, 'spacing': spacing}
            #generate slices
            if not patient.split('.')[0] in test_set:
                pickle.dump(data, open(os.path.join(sys.argv[2],'TRAIN_VOL/HR', patient.replace('.nii.gz','.pt')), 'wb'))
                print("volume finished, " + patient, img.shape)
                output_train_path_HR = os.path.join(sys.argv[2], 'TRAIN_SLICES/HR')
                output = {}
                output['spacing'] = spacing
                for j in range(len(img)):
                    name = os.path.join(output_train_path_HR, patient.replace('.nii.gz','_cor_slice_'+str(j) +'.pt'))
                    hr = np.zeros((3, img.shape[1], img.shape[2])).astype('uint16')
                    if j == 0:
                        hr[1] = img[0]
                        hr[2] = img[1]
                    elif j == 511:
                        hr[0] = img[510]
                        hr[1] = img[511]
                    else:
                        hr[0] = img[j-1]
                        hr[1] = img[j]
                        hr[2] = img[j+1]
                    output['image'] = np.clip(hr,0,4000)
                    pickle.dump(output, open(name,'wb'))

                img = np.transpose(img, (1,0,2))
                for j in range(len(img)):
                    name = os.path.join(output_train_path_HR, patient.replace('.nii.gz','_sag_slice_'+str(j) +'.pt'))
                    hr = np.zeros((3, img.shape[1], img.shape[2])).astype('uint16')
                    if j == 0:
                        hr[1] = img[0]
                        hr[2] = img[1]
                    elif j == 511:
                        hr[0] = img[510]
                        hr[1] = img[511]
                    else:
                        hr[0] = img[j-1]
                        hr[1] = img[j]
                        hr[2] = img[j+1]
                    output['image'] = np.clip(hr,0,4000)
                    pickle.dump(output, open(name,'wb'))
                print("slide finished, " + patient, img.shape)
            else:
                pickle.dump(data, open(os.path.join(sys.argv[2],'TEST/HR', patient.replace('.nii.gz','.pt')), 'wb'))
                print("volume finished, " + patient, img.shape)

