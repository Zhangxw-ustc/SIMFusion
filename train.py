import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'  
import sys
import time
import datetime
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import BraTS19, ISLES
from tqdm import tqdm
import torch.nn.functional as F
from sklearn.model_selection import KFold
from model import Fusion,Segmentation , Mutual_info_reg

def cc(img1, img2):
    eps = torch.finfo(torch.float32).eps
    """Correlation coefficient for (N, C, H, W) image; torch.float32 [0.,1.]."""
    N, C, _, _ = img1.shape
    img1 = img1.reshape(N, C, -1)
    img2 = img2.reshape(N, C, -1)
    img1 = img1 - img1.mean(dim=-1, keepdim=True)
    img2 = img2 - img2.mean(dim=-1, keepdim=True)
    cc = torch.sum(img1 * img2, dim=-1) / (eps + torch.sqrt(torch.sum(img1 **
                                                                      2, dim=-1)) * torch.sqrt(torch.sum(img2**2, dim=-1)))
    cc = torch.clamp(cc, -1., 1.)
    return cc.mean()

class BCEDiceLoss(nn.Module):
    def __init__(self):
        super(BCEDiceLoss, self).__init__()

    def forward(self, input, target):
        bce = F.binary_cross_entropy_with_logits(input, target)
        smooth = 1e-5
        input = torch.sigmoid(input)
        num = target.size(0)
        input = input.view(num, -1)
        target = target.view(num, -1)
        intersection = (input * target)
        dice = (2. * intersection.sum(1) + smooth) / (input.sum(1) + target.sum(1) + smooth)
        dice = 1 - dice.sum() / num
        return 0.5 * bce + dice

class Fusionloss(nn.Module):
    def __init__(self):
        super(Fusionloss, self).__init__()
        self.sobelconv=Sobelxy()

    def forward(self,image_vis,image_ir,generate_img):
        image_y=image_vis[:,:1,:,:]
        x_in_max=torch.max(image_y,image_ir)
        # loss_in=F.l1_loss(x_in_max,generate_img)
        loss_in = torch.mean(torch.abs(x_in_max - generate_img))
        y_grad=self.sobelconv(image_y)
        ir_grad=self.sobelconv(image_ir)
        generate_img_grad=self.sobelconv(generate_img)
        x_grad_joint=torch.max(y_grad,ir_grad)
        # loss_grad=F.l1_loss(x_grad_joint,generate_img_grad)
        loss_grad = torch.mean(torch.abs(x_grad_joint - generate_img_grad))
        loss_total=loss_in+10*loss_grad
        return loss_total,loss_in,loss_grad

class Sobelxy(nn.Module):
    def __init__(self):
        super(Sobelxy, self).__init__()
        kernelx = [[-1, 0, 1],
                  [-2,0 , 2],
                  [-1, 0, 1]]
        kernely = [[1, 2, 1],
                  [0,0 , 0],
                  [-1, -2, -1]]
        kernelx = torch.FloatTensor(kernelx).unsqueeze(0).unsqueeze(0)
        kernely = torch.FloatTensor(kernely).unsqueeze(0).unsqueeze(0)
        self.weightx = nn.Parameter(data=kernelx, requires_grad=False).cuda()
        self.weighty = nn.Parameter(data=kernely, requires_grad=False).cuda()
    def forward(self,x):
        sobelx=F.conv2d(x, self.weightx, padding=1)
        sobely=F.conv2d(x, self.weighty, padding=1)
        return torch.abs(sobelx)+torch.abs(sobely)


# data loader
class Training(object):
    def __init__(self, trainloader, testloader, timestamp, kfold):
        self.trainloader = trainloader
        self.testloader = testloader
        self.timestamp = timestamp
        self.kfold = kfold

        self.best_cc = 0
        self.best_psnr = 0

    def training(self):

        os.environ['CUDA_VISIBLE_DEVICES'] = '0'

        # . Set the hyper-parameters for training
        num_epochs = 100 # total epoch

        lr = 2e-4
        weight_decay = 0

        optim_step = 20
        optim_gamma = 0.5

        n_feat = 64

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        Fusion_model = Fusion(n_feat).to(device)
        Fusion_model.train()

        # optimizer, scheduler and loss function
        optimizer = torch.optim.Adam(
            Fusion_model.parameters(), lr=lr, weight_decay=weight_decay)

        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=optim_step, gamma=optim_gamma)

        seg_model = Segmentation().to(device)
        model = './PreSeg_models/BraTS/segmodel.pth'
        # model = './PreSeg_models/segmodel.pth'
        # model = './PreSeg_models/ISLES/segmodel.pth'
        model = torch.load(model)
        seg_model.load_state_dict(model)


        fusion_loss = Fusionloss()  
        mutual = Mutual_info_reg(n_feat//2,n_feat//2).to(device)

        '''
        ------------------------------------------------------------------------------
        Train
        ------------------------------------------------------------------------------
        '''

        torch.backends.cudnn.benchmark = True
        prev_time = time.time()

        for epoch in range(num_epochs):
            ''' train '''
            epoch = epoch + 1
            loop = tqdm(enumerate(self.trainloader), total=len(self.trainloader))

            for i, (model1, model2, label) in loop:
                model1, model2, label = model1.cuda(), model2.cuda(), label.cuda()
                _, _, seg_32 = seg_model(torch.cat([model1, model2], dim=1))
                output, f_unique_x, f_unique_y, feature_x, feature_y = Fusion_model(model1, model2, seg_32)
                output=(output-torch.min(output))/(torch.max(output)-torch.min(output))

                fus_loss, _, _ = fusion_loss(model1, model2, output)
                latentloss = cc(f_unique_x, f_unique_y)**2/(torch.clip(mutual(feature_x,feature_y),-1,1)+1.01)
                loss = fus_loss + 0.1 * latentloss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()  
                loop.set_description("Fold:[%d/5] Epoch:[%d/%d] Loss: %.4f" %(self.kfold, epoch, num_epochs, loss.item()))
           
            batches_done = epoch * len(self.trainloader) + i
            batches_left = num_epochs * len(self.trainloader) - batches_done
            time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
            prev_time = time.time()
            sys.stdout.write(
                "\r[Epoch %d/%d] [Batch %d/%d] [loss: %f] ETA: %.10s"
                % (
                    epoch,
                    num_epochs,
                    i,
                    len(self.trainloader),
                    loss.item(),
                    time_left,
                )
            )

        # adjust the learning rate
            scheduler.step()  
            if optimizer.param_groups[0]['lr'] <= 1e-6:
                optimizer.param_groups[0]['lr'] = 1e-6

if __name__ == '__main__':
    #-------   dataset  -------#
    dataset = BraTS19()
    # dataset = ISLES()
    #------- five-fold -------#
    data_induce = np.arange(0, len(dataset))
    kf = KFold(n_splits= 5)
    kfold = 0

    timestamp = datetime.datetime.now().strftime("%m-%d-%H-%M")

    for train_index, val_index in kf.split(data_induce):
        kfold = kfold + 1
        if True:
            train_subset = torch.utils.data.dataset.Subset(dataset, train_index)
            test_subset = torch.utils.data.dataset.Subset(dataset, val_index)
            train_loader = DataLoader(
                    train_subset, 
                    batch_size=4, 
                    num_workers=0, 
                    pin_memory=True,
                    shuffle=True,
                    drop_last=True
                    )  

            test_loader = DataLoader(
                    test_subset, 
                    batch_size=1, 
                    num_workers=0, 
                    pin_memory=True,
                    shuffle=False,
                    drop_last= False
                    )  
            trainer = Training(train_loader, test_loader, timestamp, kfold)
            trainer.training()