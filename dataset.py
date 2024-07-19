import numpy as np
from torch.utils.data import Dataset
from option import args
import torchvision.transforms as transforms
import SimpleITK as sitk
from PIL import Image
import glob

def read_img(path):
    img = sitk.ReadImage(path)
    data = sitk.GetArrayFromImage(img)
    return data

def normalize(img):

    if np.max(img) != np.min(img):
        img=(img-np.min(img))/(np.max(img)-np.min(img))
    return img
 

def reshape(img1, img2, label):
    output_vol_size = [160, 160]
    tempL = np.nonzero(img2)

    min_h, max_h = np.min(tempL[1]), np.max(tempL[1])
    min_w, max_w = np.min(tempL[2]), np.max(tempL[2])

    # 所需要的边缘的值
    p_h = max(output_vol_size[0] - (max_h - min_h), 0) // 2
    p_w = max(output_vol_size[1] - (max_w - min_w), 0) // 2
    min_h = max(min_h - p_h, 0)
    max_h = min_h + 160
    min_w = max(min_w - p_w, 0)
    max_w = min_w + 160
    # crop the row_data
    img1 = img1[:, min_h:max_h, min_w:max_w]
    img2 = img2[:, min_h:max_h, min_w:max_w]
    label = label[:, min_h:max_h, min_w:max_w]

    return img1, img2, label


def reshape_isles(img1, img2, label):
    output_vol_size = [96, 96]
    tempL = np.nonzero(img2)

    min_h, max_h = np.min(tempL[0]), np.max(tempL[0])
    min_w, max_w = np.min(tempL[1]), np.max(tempL[1])

    # 所需要的边缘的值
    p_h = max(output_vol_size[0] - (max_h - min_h), 0) // 2
    p_w = max(output_vol_size[1] - (max_w - min_w), 0) // 2
    min_h = max(min_h - p_h, 0)
    max_h = min_h + 96
    min_w = max(min_w - p_w, 0)
    max_w = min_w + 96
    # crop the row_data
    img1 = img1[min_h:max_h, min_w:max_w]
    img2 = img2[min_h:max_h, min_w:max_w]
    label = label[min_h:max_h, min_w:max_w]

    return img1, img2, label

class BraTS19(Dataset):
    def __init__(self, root=None, mode = 'Training', transform=None):
        self.root = args.dir_root 
        self.mode = mode

        self.examples = []
        self.root_path = self.root
        if mode == 'Training':
            with open(self.root_path + "/Training_list.txt", 'r') as f:
                self.image_list = f.readlines()
                image_list = [item.replace('\n','') for item in self.image_list]
                
                for niidir in image_list:
                    nii_t1_path = self.root_path +'/' + niidir + '/' + niidir + '_t1.nii.gz'
                    nii_t1ce_path = self.root_path +'/' + niidir + '/' + niidir + '_t1ce.nii.gz'
                    nii_flair_path = self.root_path +'/' + niidir + '/' + niidir + '_flair.nii.gz'
                    nii_t2_path = self.root_path +'/' + niidir + '/' + niidir + '_t2.nii.gz'
                    nii_seg_path = self.root_path +'/' + niidir + '/' + niidir + '_seg.nii.gz'

                    t2 = read_img(nii_t2_path)
                    d, h, w = t2.shape

                    for slice in range(40,55):
                        seg_data = read_img(nii_seg_path)[slice*2, :, :].astype(np.float32)
                        seg_data[seg_data > 0] = 1
                        if seg_data.sum() > 200:
                            self.examples.append((nii_flair_path, nii_t2_path, nii_seg_path, slice*2))

        elif mode == 'Testing':
            with open(self.root_path + "/Testing_list.txt",'r') as f:
                self.image_list = f.readlines()
                image_list = [item.replace('\n','') for item in self.image_list]

                for niidir in image_list:
                    nii_t1_path = self.root_path +'/' + niidir + '/' + niidir + '_t1.nii.gz'
                    nii_t1ce_path = self.root_path +'/' + niidir + '/' + niidir + '_t1ce.nii.gz'
                    nii_flair_path = self.root_path +'/' + niidir + '/' + niidir + '_flair.nii.gz'
                    nii_t2_path = self.root_path +'/' + niidir + '/' + niidir + '_t2.nii.gz'
                    nii_seg_path = self.root_path +'/' + niidir + '/' + niidir + '_seg.nii.gz'

                    t2 = read_img(nii_t2_path)
                    d, h, w = t2.shape
                    for slice in range(40,55):
                        seg_data = read_img(nii_seg_path)[slice*2, :, :].astype(np.float32)
                        seg_data[seg_data > 0] = 1
                        if seg_data.sum() > 200:
                            self.examples.append((nii_flair_path, nii_t2_path, nii_seg_path, slice*2))
                        

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        inf = self.examples[i]
        # t1_data = read_img(inf[0])[inf[3]][np.newaxis, :, :].astype(np.float32)
        seg_data = read_img(inf[2])[inf[3]][np.newaxis, :, :].astype(np.float32)
        t2_data = read_img(inf[1])[inf[3]][np.newaxis, :, :].astype(np.float32)
        flair_data = read_img(inf[0])[inf[3]][np.newaxis, :, :].astype(np.float32)        
        seg_data = np.where(seg_data == 4, 3, seg_data)

        flair_data, t2_data, seg_data = reshape(flair_data, t2_data, seg_data)
        t2_data = normalize(t2_data)
        flair_data = normalize(flair_data)

        return t2_data, flair_data, seg_data

class ISLES(Dataset):
    def __init__(self,mode = 'Training', transform=None):
        
        self.mode = mode
        self.examples = []
        self.transform = transform

        root_list = glob.glob(r'D:\data\ISLES2022\rawdata\*')

  
        for niidir in root_list:
            nii_adc_path = niidir +'/ses-0001/' + niidir[-18:] + '_ses-0001_adc.nii'
            nii_dwi_path = niidir +'/ses-0001/' + niidir[-18:] + '_ses-0001_dwi.nii'
            nii_flair_path = niidir +'/ses-0001/' + niidir[-18:] + '_ses-0001_flair.nii'

            nii_lab_path = 'D:\data\ISLES2022\derivatives/' + niidir[-18:] + '/ses-0001/'+ niidir[-18:] + '_ses-0001_msk.nii'


            fat = read_img(nii_adc_path)
            h, w, d = fat.shape
            if h >60 :
                for slice in range(15,25):
                    seg_data = read_img(nii_lab_path)[slice*2, :, :].astype(np.float32)
                    adc_data = read_img(nii_adc_path)[slice*2, :, :].astype(np.float32)
                    seg_data[seg_data > 0] = 1
                    if seg_data.sum() > 50 and np.max(adc_data)!=0:
                        self.examples.append((nii_adc_path, nii_dwi_path, nii_flair_path, nii_lab_path, slice*2))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        inf = self.examples[i]
        # t1_data = read_img(inf[0])[inf[3]][np.newaxis, :, :].astype(np.float32)
        adc_data = read_img(inf[0])
        adc_data = read_img(inf[0])[inf[4]].astype(np.float32)
        dwi_data = read_img(inf[1])[inf[4]][ :, :np.newaxis].astype(np.float32)
        # flair_data = read_img(inf[2]).transpose(0,1,2)[inf[4]][np.newaxis, :, :].astype(np.float32)
        lab_data = read_img(inf[3])[inf[4]][ :, :np.newaxis].astype(np.float32)
        # dwi_data = np.ndarray(dwi_data)
        # adc_data = np.ndarray(adc_data)
        # lab_data = np.ndarray(lab_data)
        adc_data, dwi_data, lab_data = reshape_isles(adc_data, dwi_data, lab_data)
        adc_data = np.expand_dims(adc_data, axis=0)
        dwi_data = np.expand_dims(dwi_data, axis=0)
        lab_data = np.expand_dims(lab_data, axis=0)
        if self.transform:
            adc_data = self.transform(adc_data)
            dwi_data = self.transform(dwi_data)
            lab_data = self.transform(lab_data)
        adc_data = normalize(adc_data)
        dwi_data = normalize(dwi_data)
        lab_data = normalize(lab_data)

        return adc_data, dwi_data, lab_data

