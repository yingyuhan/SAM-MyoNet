import numpy as np
import os
import random
from PIL import Image
import torch
from tqdm import tqdm
from torchvision import transforms
import segmentation_models_pytorch as smp
from torch.utils.data import Dataset ,DataLoader
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import argparse
import segmentation_models_pytorch as smp
from utils import *
from feature_utils import *

from SoftShapeLoss import SoftShapeLoss


'''The following parameters are currently replaced with empty values or 0,
please supplement them according to specific requirements.'''

parser = argparse.ArgumentParser()

parser.add_argument("--train_imgs_path", type=str, default="")
parser.add_argument("--train_gts_path", type=str, default="")
parser.add_argument("--test_imgs_path", type=str, default="")
parser.add_argument("--test_gts_path", type=str, default="")
parser.add_argument("--extra_train_imgs_path1", type=str, default="")
parser.add_argument("--extra_train_gts_path1", type=str, default="")
parser.add_argument("--extra_train_imgs_path2", type=str, default="")
parser.add_argument("--extra_train_gts_path2", type=str, default="")

parser.add_argument("--train_imgs_interpred", type=str, default="")
parser.add_argument("--extra_train_imgs_interpred1", type=str, default="")
parser.add_argument("--extra_train_imgs_interpred2", type=str, default="")
parser.add_argument("--test_imgs_interpred", type=str, default="")

parser.add_argument("--ratio", type=list, default=[0,0,0], help=" e.g., [1,1,1]")
parser.add_argument("--Unet_path", type=str, default="", help="Path to Unet++ training weights") 
parser.add_argument("--MyoNet_path", type=str, default="")
parser.add_argument("--img_size", type=int, default=256) # same to pretrained

#train
parser.add_argument("-batch_size", type=int, default=0)
parser.add_argument("-num_epochs", type=int, default=0)
parser.add_argument("--test_perbatch", type=int, default=0)
parser.add_argument("--alpha", type=float, default=0)
parser.add_argument("--kernlen", type=int, default=0)
parser.add_argument("--nsig", type=int, default=0)
parser.add_argument("--threshold", type=float, default=0)

parser.add_argument("--alpha2", type=float, default=0)
parser.add_argument("--beta", type=float, default=0)
parser.add_argument("--gamma", type=float, default=0)

args = parser.parse_args()

os.makedirs(args.MyoNet_path, exist_ok=True)

shape_prior = get_shape_prior(args.train_gts_path)


class Soft(nn.Module):  

    def __init__(self,kernlen, nsig):
        super(Soft, self).__init__()
        gaussian_kernel = np.float32(gkern(kernlen, nsig))
        gaussian_kernel = gaussian_kernel[np.newaxis, np.newaxis, ...] 
        self.gaussian_kernel = Parameter(torch.from_numpy(gaussian_kernel)) 
        self.padding = (kernlen-1)//2

    def forward(self, attention, x):
        soft_attention = F.conv2d(attention , self.gaussian_kernel.to('cuda') , padding=self.padding)
        soft_attention = min_max_norm(soft_attention)
        x = torch.mul(x, soft_attention.max(attention))  
        return x


class SoftAtt(nn.Module):
    def __init__(self, kernlen, nsig, alpha):
        super(SoftAtt, self).__init__()
        self.Soft = Soft(kernlen,nsig)
        self._initialize_weights()
        self.alpha = alpha

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight.data, std=0.01)
                if m.bias is not None:
                    m.bias.data.zero_()
    def forward(self,x7,SAM_attentionmap):  
        x7 = x7.unsqueeze(0)
        SAM_attentionmap = SAM_attentionmap.unsqueeze(0).unsqueeze(0)
        soft_u_heart_fea = self.Soft(SAM_attentionmap.to(torch.float32), x7)*self.alpha + x7*(1-self.alpha)   
        return soft_u_heart_fea
       
def fm_merge(x7,SAM_attentionmap, kernlen, nsig, alpha):
    output_list = []
    for i in range(x7.shape[0]):
        soft_att_model = SoftAtt(kernlen, nsig, alpha)
        output = soft_att_model(x7[i,:,:,:],SAM_attentionmap[i,:,:])
        output_list.append(output)
    output = torch.cat(output_list,dim=0)
    return output


train_imgs = dataset(args.train_imgs_path)*args.ratio[0] + dataset(args.extra_train_imgs_path1)*args.ratio[1] + dataset(args.extra_train_imgs_path2)*args.ratio[2]
train_labels =  dataset(args.train_gts_path)*args.ratio[0] + dataset(args.extra_train_gts_path1)*args.ratio[1] + dataset(args.extra_train_gts_path2)*args.ratio[2]
test_imgs = dataset(args.test_imgs_path)
test_labels = dataset(args.test_gts_path)

train_sam_probmasks = dataset(args.train_imgs_interpred) + dataset(args.extra_train_imgs_interpred1) + dataset(args.extra_train_imgs_interpred2)
test_sam_probmasks = dataset(args.test_imgs_interpred)  



class HeartDataset(Dataset): 
    def __init__(self, img, mask,sam_prob_mask, transformer):
        self.img = img
        self.mask = mask
        self.sam_prob_mask = sam_prob_mask
        self.transformer = transformer
    def __getitem__(self, index):
        img = self.img[index]
        mask = self.mask[index]
        sam_prob_mask = self.sam_prob_mask[index]
        img_open = Image.open(img)  
        img_tensor = self.transformer(img_open)
        mask_open = Image.open(mask)
        mask_tensor = self.transformer(mask_open)
        mask_tensor = torch.squeeze(mask_tensor).type(torch.long) 
        
        sam_prob_mask_open = Image.open(sam_prob_mask)
        sam_prob_mask_tensor = self.transformer(sam_prob_mask_open)
        sam_prob_mask_tensor = torch.squeeze(sam_prob_mask_tensor).type(torch.long)        
        
        return img_tensor, mask_tensor, sam_prob_mask_tensor
    def __len__(self):
        return len(self.img)


train_transformer = transforms.Compose([transforms.Resize((args.img_size, args.img_size)),transforms.ToTensor()])  
test_transformer = transforms.Compose([transforms.Resize((args.img_size, args.img_size)), transforms.ToTensor()])  

train_data = HeartDataset(train_imgs, train_labels,train_sam_probmasks, train_transformer)
test_data = HeartDataset(test_imgs, test_labels,test_sam_probmasks, test_transformer)
dl_train = DataLoader(train_data, batch_size=args.batch_size, shuffle=True) 
dl_test = DataLoader(test_data, batch_size=args.batch_size, shuffle=True)



class CustomModel(nn.Module):
    def __init__(self, base_model, kernlen, nsig, alpha):
        super(CustomModel, self).__init__()
        self.kernlen= kernlen
        self.nsig= nsig
        self.alpha = alpha
        self.base_model = base_model
        self.conv = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=1)

    def forward(self, x, sam_prob_mask):
        x1, x2, x3, x4, x5, x6 = self.base_model.encoder(x) 
        x7 = self.base_model.decoder(x1, x2, x3, x4, x5, x6) 
        x_merge = fm_merge(x7, sam_prob_mask, self.kernlen, self.nsig, self.alpha)
        x_merge = torch.cat((x_merge, sam_prob_mask.unsqueeze(1)), dim=1)
        y_pred = self.base_model.segmentation_head(x_merge)    
        return y_pred




base_model = smp.UnetPlusPlus(encoder_name="resnet34", in_channels=3, classes=2)
base_model.load_state_dict(torch.load(args.Unet_path))
base_model.segmentation_head[0] = nn.Conv2d(17,2,kernel_size=(3,3),stride=(1,1),padding=(1,1))
model = CustomModel(base_model, args.kernlen, args.nsig, args.alpha) 


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model=model.to(device)
loss_fn=SoftShapeLoss(args.alpha2, args.beta, args.gamma, shape_prior=torch.tensor(shape_prior).to(device)) 
optimizer=torch.optim.Adam(model.parameters(),lr=0.0001)

def train_batches(batch_count,model, dl_train, dl_test):
    correct = 0
    total = 0
    running_loss = 0
    epoch_iou = []  
    train_dice = []
    epoch_test_iou = [] 

    model.train()
    for x, y, sam_prob_mask in tqdm(dl_train):
        batch_count += 1
        x, y, sam_prob_mask = x.to(device), y.to(device), sam_prob_mask.to(device)
        y_pred = model(x, sam_prob_mask)
        loss = loss_fn(y_pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            y_pred = torch.argmax(y_pred, dim=1)
            correct += (y_pred == y).sum().item()
            total += y.size(0)
            running_loss += loss.item()

            intersection = torch.logical_and(y, y_pred)  
            union = torch.logical_or(y, y_pred)  
            batch_iou = torch.sum(intersection) / torch.sum(union)  
            epoch_iou.append(batch_iou.item())

            y_pred_dice = y_pred 
            y_pred_dice = y_pred_dice.cpu().numpy()
            ys = y.cpu().numpy()  
            dice = Dice(y_pred_dice, ys)  
            train_dice.append(dice)

        if batch_count % args.test_perbatch == 0:  #!######## 原始代码是100
            print("Batch prediction results:")
            test_correct = 0
            test_total = 0
            test_running_loss = 0 
            epoch_test_iou = []
            test_dice = []  
            test_precision = [] 
            test_recall = []

            model.eval()
            with torch.no_grad():
                for x, y, sam_prob_mask in tqdm(dl_test):
                    x, y, sam_prob_mask = x.to(device), y.to(device), sam_prob_mask.to(device)
                    y_pred = model(x, sam_prob_mask)
                    loss = loss_fn(y_pred, y)
                    #loss = enhanced_loss_function_try(y_pred, y)
                    y_pred = torch.argmax(y_pred, dim=1)
                    test_correct += (y_pred == y).sum().item()
                    test_total += y.size(0)
                    test_running_loss += loss.item()

                    intersection = torch.logical_and(y, y_pred)
                    union = torch.logical_or(y, y_pred)
                    batch_iou = torch.sum(intersection) / torch.sum(union)
                    epoch_test_iou.append(batch_iou.item())

                    y_pred_dice = y_pred  
                    y_pred_dice = y_pred_dice.cpu().numpy()  
                    ys = y.cpu().numpy()
                    dice = Dice(y_pred_dice, ys)
                    test_dice.append(dice)

                    batch_precision = precision(ys, y_pred_dice)
                    batch_recall = recall(ys, y_pred_dice)
                    test_precision.append(batch_precision)
                    test_recall.append(batch_recall)

            static_dict = model.state_dict() 
            if np.mean(test_dice)> args.threshold:
                torch.save(static_dict,'{}/epo{}_batch{}_dice_{}_testdice_{}.pth'.format(
                    args.MyoNet_path,
                    epoch+1,batch_count, 
                    round(np.mean(train_dice), 5), 
                    round(np.mean(test_dice), 5),
                    ))
            print(  'epoch:', epoch+1,
                    'batch_count:', batch_count,
                    'train_Dice:', round(np.mean(train_dice), 5),
                    'test_iou:', round(np.mean(epoch_test_iou), 5),
                    'testDice:', round(np.mean(test_dice), 5),)   
    epoch_loss = running_loss / len(dl_train.dataset)
    epoch_acc = correct / (total * 256 * 256) 

    return batch_count, epoch_loss, epoch_acc, epoch_iou, epoch_test_iou


batch_count = 0
for epoch in range(args.num_epochs):
    batch_count, epoch_loss, epoch_acc, epoch_iou, epoch_test_iou = train_batches(batch_count,model,dl_train,dl_test)
    print('EPO{} fin --------------'.format(epoch+1))

