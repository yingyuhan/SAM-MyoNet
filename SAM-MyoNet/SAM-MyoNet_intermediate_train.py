import numpy as np
import os
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import monai
from segment_anything import sam_model_registry
import torch.nn.functional as F
import argparse
import random
from datetime import datetime
import shutil
import glob
from PIL import Image
import segmentation_models_pytorch as smp

from WeightPredictionNet import WeightPredictionNet
from feature_utils import *
from utils import *
join = os.path.join

torch.cuda.empty_cache()

os.environ["OMP_NUM_THREADS"] = "4"  
os.environ["OPENBLAS_NUM_THREADS"] = "4"  
os.environ["MKL_NUM_THREADS"] = "6"  
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"  
os.environ["NUMEXPR_NUM_THREADS"] = "6"  


'''The following parameters are currently replaced with empty values or 0,
please supplement them according to specific requirements.'''

parser = argparse.ArgumentParser()

parser.add_argument("-task_name", type=str, default="") 
parser.add_argument("-model_type", type=str, default="", help=" e.g., 'vit_l', 'vit_h', 'vit_b'") 
parser.add_argument("-checkpoint", type=str, default="") # SAM checkpoint path
parser.add_argument("--load_pretrain", type=bool, default=True, help="use wandb to monitor training")
parser.add_argument("-pretrain_model_path", type=str, default="")
parser.add_argument("-work_dir", type=str, default="")
# train
parser.add_argument("-num_epochs", type=int, default=0)
parser.add_argument("-batch_size", type=int, default=0)
parser.add_argument("-num_workers", type=int, default=0)
# Optimizer parameters
parser.add_argument("-weight_decay", type=float, default=0)
parser.add_argument("-lr", type=float, default=0, metavar="LR")
parser.add_argument("-use_wandb", type=bool, default=False)
parser.add_argument("-use_amp", action="store_true", default=False)
parser.add_argument("--resume", type=str, default="")
parser.add_argument("--device", type=str, default="cuda:0")

# Other Path
parser.add_argument("--train_imgs_path", type=str, default="")
parser.add_argument("--train_gts_path", type=str, default="")
parser.add_argument("--test_imgs_path", type=str, default="")
parser.add_argument("--test_gts_path", type=str, default="")

parser.add_argument("--extra_train_imgs_path1", type=str, default="")
parser.add_argument("--extra_train_gts_path1", type=str, default="")
parser.add_argument("--extra_train_imgs_path2", type=str, default="")
parser.add_argument("--extra_train_gts_path2", type=str, default="")

parser.add_argument("--Unet_path", type=str, default="", help="Path to Unet++ training weights")

parser.add_argument("--train_path", type=str, default="")
parser.add_argument("--test_path", type=str, default="")
parser.add_argument("--extra_train_path1", type=str, default="")
parser.add_argument("--extra_train_path2", type=str, default="")

parser.add_argument("--test_perbatch", type=int, default=0)
parser.add_argument("--extra_test_perbatch", type=int, default=0)
parser.add_argument("--all_test_perbatch", type=int, default=0)
parser.add_argument("--testall_perepo", type=int, default=0)

args = parser.parse_args()



class PngDataset(Dataset):
    def __init__(self, data_root,bbox_list, extra):
        self.data_root = data_root
        self.bbox_list = bbox_list
        self.extra = extra
        self.gt_path = join(data_root, "gts")
        self.img_path = join(data_root, "imgs")
        self.gt_path_files = sorted(
            glob.glob(join(self.gt_path, "**/*.png"), recursive=True)
        )
        self.gt_path_files = [
            file
            for file in self.gt_path_files
            if os.path.isfile(join(self.img_path, os.path.basename(file)))
        ]
        print(f"number of images: {len(self.gt_path_files)}")

    def __len__(self):
        return len(self.gt_path_files)

    def __getitem__(self, index):
        img_name = os.path.basename(self.gt_path_files[index])
        img_pil = Image.open(join(self.img_path, img_name)).convert('RGB') 
        img_pil = img_pil.resize((1024, 1024), Image.Resampling.LANCZOS)
        img_1024 = np.array(img_pil, dtype=np.float32) / 255.0  
        img_1024 = np.transpose(img_1024, (2, 0, 1)) 
        
        assert np.max(img_1024) <= 1.0 and np.min(img_1024) >= 0.0, "image should be normalized to [0, 1]"
        
        gt_pil = Image.open(self.gt_path_files[index]).convert('L')
        gt_pil = gt_pil.resize((1024, 1024), Image.Resampling.NEAREST)
        gt = np.array(gt_pil, dtype=np.uint8)  

        assert img_name == os.path.basename(self.gt_path_files[index]), "img gt name error"
        
        label_ids = np.unique(gt)[1:]
        if len(label_ids) == 0:
            gt2D = np.zeros_like(gt, dtype=np.uint8)
        else:
            gt2D = np.uint8(gt == random.choice(label_ids.tolist()))

        if self.extra == True:
            bbox_sim,area_sim,scaled_snr = 1,1,1
        else:
            bbox_sim = calculate_bbox_sim(torch.tensor(img_1024).float(),bbox_feature_array,cluster_centers,labels)
            current_area = calculate_label_area(torch.tensor(img_1024).float())
            area_sim = calculate_area_sim(current_area, avg_area)
            scaled_snr = calculate_snr(torch.tensor(img_1024).float())
        
        return (
            torch.tensor(img_1024).float(),
            torch.tensor(gt2D[None, :, :]).long(),
            self.bbox_list[index],
            img_name,
            [bbox_sim,area_sim,scaled_snr]
        )



bbox_feature_array,cluster_centers,labels = feature_cluster(args.train_gts_path,n_clusters=3) 
avg_area = calculate_avg_area_in_dataset(args.train_gts_path)

train_mask_list = [args.train_gts_path + x for x in os.listdir(args.train_gts_path)]
extra_train_mask_list1 = [args.extra_train_gts_path1 + x for x in os.listdir(args.extra_train_gts_path1)]
extra_train_mask_list2 = [args.extra_train_gts_path2 + x for x in os.listdir(args.extra_train_gts_path2)]
   
train_bbox_list = make_train_bbox(train_mask_list)
extra_train_bbox_list1 = make_train_bbox(extra_train_mask_list1)
extra_train_bbox_list2 = make_train_bbox(extra_train_mask_list2)

cnn_model = smp.UnetPlusPlus(encoder_name="resnet34", in_channels=3,  classes=2) 
test_mask_pre_p_list = make_fm_and_pro(args.test_imgs_path,args.test_gts_path,cnn_model,args.Unet_path) 
test_bbox_pre_list = prob_to_mask(test_mask_pre_p_list)



train_dataset = PngDataset( args.train_path , train_bbox_list, extra=False )  
train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
extra_train_dataset = PngDataset(args.extra_train_path1, extra_train_bbox_list1, extra=True) + PngDataset(args.extra_train_path2, extra_train_bbox_list2, extra=True)
extra_train_dataloader = DataLoader(extra_train_dataset, batch_size=args.batch_size, shuffle=True)
all_train_dataloader = DataLoader(train_dataset + extra_train_dataset, batch_size=args.batch_size, shuffle=True)
test_dataset = PngDataset(args.test_path, test_bbox_pre_list,extra=False)
test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
print("Number of training samples: ", len(train_dataset))


run_id = datetime.now().strftime("%Y%m%d-%H%M")
model_save_path = join(args.work_dir, args.task_name + "-" + run_id)
device = torch.device(args.device)

class SAM(nn.Module):
    def __init__(
        self,
        image_encoder,
        mask_decoder,
        prompt_encoder,
    ):
        super().__init__()
        self.image_encoder = image_encoder
        self.mask_decoder = mask_decoder
        self.prompt_encoder = prompt_encoder
        # freeze prompt encoder
        for param in self.prompt_encoder.parameters():
            param.requires_grad = False
        for param in self.image_encoder.parameters():
            param.requires_grad = False

    def forward(self, image, box):
        image_embedding = self.image_encoder(image)[0] 
        with torch.no_grad():
            box_torch = torch.as_tensor(box, dtype=torch.float32, device=image.device)
            sparse_embeddings, dense_embeddings = self.prompt_encoder(
                points=None,
                boxes=box_torch,
                masks=None,
            )
        low_res_masks, _ = self.mask_decoder(
            image_embeddings=image_embedding, 
            image_pe=self.prompt_encoder.get_dense_pe(),  
            sparse_prompt_embeddings=sparse_embeddings, 
            dense_prompt_embeddings=dense_embeddings,  
            multimask_output=False,
        )
        ori_res_masks = F.interpolate(
            low_res_masks,
            size=(image.shape[2], image.shape[3]),
            mode="bilinear",
            align_corners=False,
        )
        return ori_res_masks



os.makedirs(model_save_path, exist_ok=True)
shutil.copyfile(
    __file__, join(model_save_path, run_id + "_" + os.path.basename(__file__)))

sam_model = sam_model_registry[args.model_type](checkpoint=args.checkpoint)
medsam_model = SAM(
    image_encoder=sam_model.image_encoder,
    mask_decoder=sam_model.mask_decoder,
    prompt_encoder=sam_model.prompt_encoder,
).to(device)

weight_prediction_net = WeightPredictionNet()
weight_prediction_net = weight_prediction_net.to(device)


for name, param in medsam_model.named_parameters():
  if name.startswith("vision_encoder") or name.startswith("prompt_encoder"):
    param.requires_grad_(False)   # mask decoder only

parameters = list(medsam_model.mask_decoder.parameters()) + list(weight_prediction_net.parameters())
optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=args.weight_decay)

seg_loss = monai.losses.DiceLoss(sigmoid=True, squared_pred=True, reduction="mean")
ce_loss = nn.BCEWithLogitsLoss(reduction="mean")




start_epoch = 1
num_epochs = args.num_epochs
iter_num = 0
losses = []
best_loss = 1e10
batch_count = 0
bestdice = 0
for epoch in range(start_epoch, num_epochs+1):
    epoch_loss = 0
    train_dice = []
    medsam_model.train()
    for batch, (image, gt2D, boxes, img_name, num) in enumerate(tqdm(train_dataloader)): 
        batch_count += 1
        optimizer.zero_grad()
        image, gt2D = image.to(device), gt2D.to(device)
        medsam_pred = medsam_model(image,torch.tensor([[p * 4 for p in boxes]]))
        loss = seg_loss(medsam_pred, gt2D) + ce_loss(medsam_pred, gt2D.float())
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        epoch_loss += loss.item()
        iter_num += 1
        with torch.no_grad():
            y_pred = (medsam_pred.cpu().numpy() >= 0).astype(np.int64)
            gt2D = gt2D.cpu().numpy() 
            dice=Dice(y_pred,gt2D)  
            train_dice.append(dice)
        
        if batch_count % args.test_perbatch == 0 and batch_count!=0 :
            test_dice = []
            medsam_model.eval()
            with torch.no_grad():
                for step, (image, gt2D, boxes, img_name, num) in enumerate(tqdm(test_dataloader)):
                    image, gt2D = image.to(device), gt2D.to(device)

                    medsam_pred = medsam_model(image,torch.tensor([[p * 4 for p in boxes]]))
                
                    y_pred = (medsam_pred.cpu().numpy() >= 0).astype(np.int64)
                    gt2D = gt2D.cpu().numpy() 
                    dice = Dice(y_pred,gt2D)  
                    test_dice.append(dice)
            checkpoint = {"model": medsam_model.state_dict(),"optimizer": optimizer.state_dict(),"epoch": epoch,}
            if np.mean(test_dice)> bestdice:
                bestdice = np.mean(test_dice)
                torch.save(checkpoint, join(model_save_path, "Batch{}_traindice{}_testdice{}.pth".format(batch_count,         round(np.mean(train_dice),5),    round(np.mean(test_dice),5)       )))
            print('epoch {} batch {} Train： testdice_{}                   '.format(epoch , batch_count, round(np.mean(test_dice),5)))
    
    
    epoch_loss = 0
    train_dice = []
    medsam_model.train()
    for batch, (image, gt2D, boxes, img_name, num) in enumerate(tqdm(extra_train_dataloader)):
        optimizer.zero_grad()
        image, gt2D = image.to(device), gt2D.to(device)
        num1,num2,num3 = num[0].to(device),num[1].to(device),num[2].to(device)
        predicted_weights = weight_prediction_net(image,num1,num2,num3)
        if torch.rand(1).to(device) > predicted_weights:
            medsam_pred = medsam_model(image,torch.tensor([[p * 4 for p in boxes]]))
            loss = seg_loss(medsam_pred, gt2D) + ce_loss(medsam_pred, gt2D.float())
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            epoch_loss += loss.item()
            iter_num += 1
            with torch.no_grad():
                y_pred = (medsam_pred.cpu().numpy() >= 0).astype(np.int64)
                gt2D = gt2D.cpu().numpy()  
                dice=Dice(y_pred,gt2D)  
                train_dice.append(dice)
        batch_count += 1

        if batch_count % args.extra_test_perbatch == 0:
            test_dice = []
            medsam_model.eval()
            with torch.no_grad():
                for step, (image, gt2D, boxes, img_name, num) in enumerate(tqdm(test_dataloader)):
                    image, gt2D = image.to(device), gt2D.to(device)
                    medsam_pred = medsam_model(image,torch.tensor([[p * 4 for p in boxes]]))
                    y_pred = (medsam_pred.cpu().numpy() >= 0).astype(np.int64)
                    gt2D = gt2D.cpu().numpy()  
                    dice = Dice(y_pred,gt2D)  
                    test_dice.append(dice)
            checkpoint = {"model": medsam_model.state_dict(),"optimizer": optimizer.state_dict(),"epoch": epoch,}
            if np.mean(test_dice)> bestdice:
                bestdice = np.mean(test_dice)
                torch.save(checkpoint, join(model_save_path, "Batch{}_traindice{}_testdice{}.pth".format(batch_count,         round(np.mean(train_dice),5),    round(np.mean(test_dice),5)       )))
            print('epoch {} batch {} Extra Train: testdice_{}            '.format(epoch, batch_count,round(np.mean(test_dice),5)))



    if epoch % args.testall_perepo == 0:
        epoch_loss = 0
        train_dice = []
        medsam_model.train()
        for batch, (image, gt2D, boxes, img_name, num) in enumerate(tqdm(all_train_dataloader)): 
            optimizer.zero_grad()
            image, gt2D = image.to(device), gt2D.to(device)
            medsam_pred = medsam_model(image,torch.tensor([[p * 4 for p in boxes]]))
            loss = seg_loss(medsam_pred, gt2D) + ce_loss(medsam_pred, gt2D.float())
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            epoch_loss += loss.item()
            iter_num += 1
            with torch.no_grad():
                y_pred = (medsam_pred.cpu().numpy() >= 0).astype(np.int64)
                gt2D = gt2D.cpu().numpy()  
                dice=Dice(y_pred,gt2D)  
                train_dice.append(dice)
            batch_count += 1
            if batch_count % args.all_test_perbatch == 0:
                test_dice = []
                medsam_model.eval()
                with torch.no_grad():
                    for step, (image, gt2D, boxes, img_name, num) in enumerate(tqdm(test_dataloader)):
                        image, gt2D = image.to(device), gt2D.to(device)

                        medsam_pred = medsam_model(image,torch.tensor([[p * 4 for p in boxes]]))
                    
                        y_pred = (medsam_pred.cpu().numpy() >= 0).astype(np.int64)
                        gt2D = gt2D.cpu().numpy()  
                        dice = Dice(y_pred,gt2D)  
                        test_dice.append(dice)
                checkpoint = {"model": medsam_model.state_dict(),"optimizer": optimizer.state_dict(),"epoch": epoch,}
                if np.mean(test_dice)> bestdice:
                    bestdice = np.mean(test_dice)
                    torch.save(checkpoint, join(model_save_path, "Batch{}_traindice{}_testdice{}.pth".format(batch_count, round(np.mean(train_dice),5),    round(np.mean(test_dice),5)  )))
                print('epoch {} batch {} ALL Train: testdice_{}              '.format(epoch,batch_count,round(np.mean(test_dice),5)))








