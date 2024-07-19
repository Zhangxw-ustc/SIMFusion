from model import Fusion, Segmentation
import os
import numpy as np
from utils.Util import tensor2img, save_img
from utils.Metrics import *
import torch
import warnings
import logging
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.CRITICAL)
import cv2
from torch.utils.data import DataLoader
from dataset import BraTS19
from tqdm import tqdm

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

Fusion_type = "T2-T2Flair"

testloader = DataLoader(BraTS19(mode='Testing'), batch_size=1, shuffle=False, num_workers=0)

#---------------------------------------------#
class Validation_With_Seg(object):
    def __init__(self, model_root, testloader, save = False):
        n_feat = 64
        self.model_root = model_root
        self.model_name=self.model_root.split('/')[-1].split('.')[0]

        self.testloader = testloader
        self.Fusion_model = Fusion(n_feat).cuda()
        
        self.Fusion_model.load_state_dict(torch.load(self.model_root))

        self.save = save


    def validation(self):

        seg_model = Segmentation().cuda()
        model = './PreSeg_models/BraTS/segmodel.pth'
        # model = './PreSeg_models/ISLES/seg_80.pth'
        model = torch.load(model)
        seg_model.load_state_dict(model)

        metric_result = np.zeros((12))
        ep_PSNR_list = []
        ep_SSIM_list = []
        
        ep_CC_list = []
        ep_MSE_list = []
        ep_VIFF_list = []
        ep_SCD_list = []
        ep_Qabf_list = []

        ep_MI_list = []
        ep_EN_list = []
        ep_SF_list = []
        ep_SD_list = []
        ep_AG_list = []

        self.Fusion_model.eval()

        with torch.no_grad():
            for i, (modal1, modal2, label) in tqdm(enumerate(self.testloader), total=len(self.testloader)):
                modal1, modal2, label = modal1.cuda(), modal2.cuda(), label.cuda()
                
                _, _, seg_32 = seg_model(torch.cat([modal1, modal2], dim=1))

                output, f_unique_x, f_unique_y, feature_x, feature_y = self.Fusion_model(modal1, modal2, seg_32)                
                # output = self.Fusion_model(modal1, modal2, seg_32)
                output=(output-torch.min(output))/(torch.max(output)-torch.min(output))

                modal1 = tensor2img(modal1)
                modal2 = tensor2img(modal2)
                output = tensor2img(output)

                PSNR_idx = PSNR_function(output, modal1) + PSNR_function(output, modal2)
                SSIM_idx = SSIM_function(modal1, modal2, output)
                
                CC_idx = CC_function(modal1, modal2, output)
                MSE_idx = MSE_function(modal1, modal2, output)
                VIFF_idx = VIFF_function(modal1, modal2, output)
                SCD_idx = SCD_function(modal1, modal2, output)
                Qabf_idx = Qabf_function(modal1, modal2, output)

                MI_idx = MI_function(modal1, modal2, output)
                EN_idx = EN_function(output)
                SF_idx = SF_function(output)
                SD_idx = SD_function(output)
                AG_idx = AG_function(output)


                ep_PSNR_list.append(PSNR_idx)
                ep_SSIM_list.append(SSIM_idx)

                ep_CC_list.append(CC_idx)
                ep_MSE_list.append(MSE_idx)
                ep_VIFF_list.append(VIFF_idx)
                ep_SCD_list.append(SCD_idx)
                ep_Qabf_list.append(Qabf_idx)

                ep_MI_list.append(MI_idx)
                ep_EN_list.append(EN_idx)
                ep_SF_list.append(SF_idx)
                ep_SD_list.append(SD_idx)
                ep_AG_list.append(AG_idx)


                if self.save:
                    modal1_path = os.path.join(self.model_root, 'results/{}/{}'.format(i+1, Fusion_type.split('-')[0]))
                    os.makedirs(modal1_path, exist_ok=True)
                    modal2_path = os.path.join(self.model_root, 'results/{}/{}'.format(i+1, Fusion_type.split('-')[1]))
                    os.makedirs(modal2_path, exist_ok=True)
                    result_path = os.path.join(self.model_root, 'results/{}/fusion'.format(i+1))
                    os.makedirs(result_path, exist_ok=True)
                    
                    save_img(output, '{}/{}.png'.format(result_path,i))
                    save_img(modal1, '{}/'.format(modal1_path) + '{}.png'.format(i+1))
                    save_img(modal2, '{}/'.format(modal2_path) + '{}.png'.format(i+1))
        
        metric_result = np.array([np.mean(ep_PSNR_list), np.mean(ep_SSIM_list), np.mean(ep_CC_list), np.mean(ep_MSE_list), np.mean(ep_VIFF_list), np.mean(ep_SCD_list), np.mean(ep_Qabf_list), np.mean(ep_MI_list), np.mean(ep_EN_list), np.mean(ep_SF_list), np.mean(ep_SD_list), np.mean(ep_AG_list)])        
        metric_result = metric_result.round(3)

        print("PSNR\t SSIM\t CC\t MSE\t VIFF\t SCD\t Qabf\t MI\t EN\t SF\t SD\t AG\t")
        print(str(metric_result[0]) + '\t'
            + str(metric_result[1]) + '\t'
            + str(metric_result[2]) + '\t'
            + str(metric_result[3]) + '\t'
            + str(metric_result[4]) + '\t'
            + str(metric_result[5]) + '\t'
            + str(metric_result[6]) + '\t'
            + str(metric_result[7]) + '\t'
            + str(metric_result[8]) + '\t'
            + str(metric_result[9]) + '\t'
            + str(metric_result[10]) + '\t'
            + str(metric_result[11]) + '\t')
            
        return metric_result


if __name__ == '__main__':

    for kflod in range(1,6):
        save = False
        path1 = r'E://dataset/adc-dwi/' + str(kflod) + '/adc/'
        path2 = r'E://dataset/adc-dwi/' + str(kflod) + '/dwi/'
        output_path = r'E://Hi-Net/adc-dwi/' + str(kflod) + '/'

        metric_result = np.zeros((12))
        ep_PSNR_list = []
        ep_SSIM_list = []
        
        ep_CC_list = []
        ep_MSE_list = []
        ep_VIFF_list = []
        ep_SCD_list = []
        ep_Qabf_list = []

        ep_MI_list = []
        ep_EN_list = []
        ep_SF_list = []
        ep_SD_list = []
        ep_AG_list = []

        file_names = os.listdir(path1)

        for i, file_name in tqdm(enumerate(file_names), total=len(file_names)):

            file_path1 = os.path.join(path1, file_name)
            file_path2 = os.path.join(path2, file_name)
            fusion_path = os.path.join(output_path, file_name)

            modal1 = cv2.imread(file_path1, cv2.IMREAD_GRAYSCALE)
            modal2 = cv2.imread(file_path2, cv2.IMREAD_GRAYSCALE)
            output = cv2.imread(fusion_path, cv2.IMREAD_GRAYSCALE)

            PSNR_idx = PSNR_function(output, modal1) + PSNR_function(output, modal2)
            SSIM_idx = SSIM_function(modal1, modal2, output)
            
            CC_idx = CC_function(modal1, modal2, output)
            MSE_idx = MSE_function(modal1, modal2, output)
            VIFF_idx = VIFF_function(modal1, modal2, output)
            SCD_idx = SCD_function(modal1, modal2, output)
            Qabf_idx = Qabf_function(modal1, modal2, output)

            MI_idx = MI_function(modal1, modal2, output)
            EN_idx = EN_function(output)
            SF_idx = SF_function(output)
            SD_idx = SD_function(output)
            AG_idx = AG_function(output)

            ep_PSNR_list.append(PSNR_idx)
            ep_SSIM_list.append(SSIM_idx)

            ep_CC_list.append(CC_idx)
            ep_MSE_list.append(MSE_idx)
            ep_VIFF_list.append(VIFF_idx)
            ep_SCD_list.append(SCD_idx)
            ep_Qabf_list.append(Qabf_idx)

            ep_MI_list.append(MI_idx)
            ep_EN_list.append(EN_idx)
            ep_SF_list.append(SF_idx)
            ep_SD_list.append(SD_idx)
            ep_AG_list.append(AG_idx)
        
        metric_result = np.array([np.mean(ep_PSNR_list), np.mean(ep_SSIM_list), np.mean(ep_CC_list), np.mean(ep_MSE_list), np.mean(ep_VIFF_list), np.mean(ep_SCD_list), np.mean(ep_Qabf_list), np.mean(ep_MI_list), np.mean(ep_EN_list), np.mean(ep_SF_list), np.mean(ep_SD_list), np.mean(ep_AG_list)])        
        metric_result = metric_result.round(3)

        print("PSNR\t SSIM\t CC\t MSE\t VIFF\t SCD\t Qabf\t MI\t EN\t SF\t SD\t AG\t")
        print(str(metric_result[0]) + '\t'
            + str(metric_result[1]) + '\t'
            + str(metric_result[2]) + '\t'
            + str(metric_result[3]) + '\t'
            + str(metric_result[4]) + '\t'
            + str(metric_result[5]) + '\t'
            + str(metric_result[6]) + '\t'
            + str(metric_result[7]) + '\t'
            + str(metric_result[8]) + '\t'
            + str(metric_result[9]) + '\t'
            + str(metric_result[10]) + '\t'
            + str(metric_result[11]) + '\t')