import numpy as np
import os
from PIL import Image
import torch
from tqdm import tqdm
import segmentation_models_pytorch as smp
import cv2
import torch.nn as nn
import torch.nn.functional as F
from segment_anything import sam_model_registry
import argparse
from utils import *
from feature_utils import *



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

parser.add_argument("--Unet_path", type=str, default="", help="Path to Unet++ training weights")
parser.add_argument("--SAM_pth", type=str, default="")

parser.add_argument("--train_pred_output_path", type=str, default="")
parser.add_argument("--extra_train_pred_output_path1", type=str, default="")
parser.add_argument("--extra_train_pred_output_path2", type=str, default="")
parser.add_argument("--test_pred_output_path", type=str, default="")

parser.add_argument("-model_type", type=str, default="" , help=" e.g., 'vit_l', 'vit_h', 'vit_b'")  
parser.add_argument("-checkpoint", type=str, default="") # SAM checkpoint path

args = parser.parse_args()




train_mask_list = [args.train_gts_path + x for x in os.listdir(args.train_gts_path)]
extra_train_mask_list1 = [args.extra_train_gts_path1 + x for x in os.listdir(args.extra_train_gts_path1)]
extra_train_mask_list2 = [args.extra_train_gts_path2 + x for x in os.listdir(args.extra_train_gts_path2)]
   
train_bbox_list = make_train_bbox(train_mask_list)
extra_train_bbox_list1 = make_train_bbox(extra_train_mask_list1)
extra_train_bbox_list2 = make_train_bbox(extra_train_mask_list2)

cnn_model = smp.UnetPlusPlus(encoder_name="resnet34", in_channels=3,  classes=2) 
test_mask_pre_p_list = make_fm_and_pro(args.test_imgs_path,args.test_gts_path,cnn_model,args.Unet_path)
test_bbox_pre_list = prob_to_mask(test_mask_pre_p_list)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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


def process_single_image(image_path, bbox, output_dir):

    os.makedirs(output_dir, exist_ok=True)
    img_pil = Image.open(image_path).convert('RGB')
    img_pil = img_pil.resize((1024, 1024), Image.Resampling.LANCZOS)
    img_np = np.array(img_pil, dtype=np.float32) / 255.0
    img_tensor = torch.tensor(img_np, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device)
    
    with torch.no_grad():
        prediction = medsam_model(img_tensor, torch.tensor([[p * 4 for p in bbox]]))
        prediction = prediction.squeeze(0).squeeze(0).cpu().numpy()
    binary_mask = (prediction > 0.5).astype(np.uint8) * 255
    img_name = os.path.basename(image_path)
    output_path = os.path.join(output_dir, f"pred_{img_name}")
    cv2.imwrite(output_path, binary_mask)
    return output_path


def process_imgs_infolder(imgs_folder_path,bbox_list,output_dir):
    imgs_path_list = [imgs_folder_path+'/'+i   for i in os.listdir(imgs_folder_path)]
    for index,image_path in enumerate(tqdm(imgs_path_list)):
        process_single_image(image_path, bbox_list[index] ,output_dir)




sam = sam_model_registry[args.model_type](checkpoint=args.checkpoint)
medsam_model = SAM(
    image_encoder=sam.image_encoder,
    mask_decoder=sam.mask_decoder,
    prompt_encoder=sam.prompt_encoder,
).to(device)

checkpoint = torch.load(args.SAM_pth, map_location=device)
medsam_model.load_state_dict(checkpoint['model']) 
medsam_model.eval()

# image generation
process_imgs_infolder(args.train_imgs_path, train_bbox_list, args.train_pred_output_path)
process_imgs_infolder(args.extra_train_imgs_path1, extra_train_bbox_list1, args.extra_train_pred_output_path1)
process_imgs_infolder(args.extra_train_imgs_path2, extra_train_bbox_list2, args.extra_train_pred_output_path2)
process_imgs_infolder(args.test_imgs_path, test_bbox_pre_list, args.test_pred_output_path)


