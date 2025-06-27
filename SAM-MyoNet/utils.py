import numpy as np
import os
import cv2
from torchvision import transforms
import torch
from PIL import Image
from tqdm import tqdm
import scipy.stats as st



def Dice(input, target, eps=1): 
    input_flatten = input.flatten()  
    target_flatten = target.flatten()
    overlap = np.sum(input_flatten * target_flatten)  
    return np.clip(((2. * overlap) / (np.sum(target_flatten) + np.sum(input_flatten) + eps)), 1e-4, 0.9999) 

def precision(y_true, y_pred):
    true_positives = np.sum(np.logical_and(y_true == 1, y_pred == 1))
    predicted_positives = np.sum(y_pred == 1)
    if predicted_positives == 0:
        return 0
    return true_positives / predicted_positives

def recall(y_true, y_pred):
    true_positives = np.sum(np.logical_and(y_true == 1, y_pred == 1))
    actual_positives = np.sum(y_true == 1)
    if actual_positives == 0:
        return 0
    return true_positives / actual_positives

def dataset(root):
    imgs = []
    files = os.listdir(root)
    for j in files:
        if '.png' in j:
            img = os.path.join(root,j)
            imgs.append(img)
    return imgs

def normalize_array(array):
    min_val = np.min(array)
    max_val = np.max(array)
    normalized_array = 255 * (array - min_val) / (max_val - min_val)
    return normalized_array.astype(np.uint8)

def get_bounding_box(ground_truth_map):
    y_indices, x_indices = np.where(ground_truth_map > 0)
    x_min, x_max = np.min(x_indices), np.max(x_indices)
    y_min, y_max = np.min(y_indices), np.max(y_indices)
    bbox = [x_min, y_min, x_max, y_max]   
    return bbox  

def mask_pruing_soft(img):
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) 
    mask = np.zeros_like(img)  
    for contour in contours:  
        area = cv2.contourArea(contour)  
        if area < 400:    
            cv2.drawContours(mask, [contour], -1, 255, -1)  
    img[mask == 255] = 0  
    return img

def make_fm_and_pro(img_folder_path,mask_folder_path,model,pth_path): 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    test_imgs, test_labels = dataset(img_folder_path),dataset(mask_folder_path) 
    model.load_state_dict(torch.load(pth_path))
    model.eval()
    model.to(device)
    test_transformer=transforms.Compose([transforms.Resize((256,256)),transforms.ToTensor()])
    mask_pre_p_list =[] 
    for i in tqdm(range(len(test_imgs)), desc='Processing images', ncols=100): 
        test_image_path = test_imgs[i] 
        test_image = Image.open(test_image_path)
        test_image = test_transformer(test_image).unsqueeze(0).to(device)  
        x1,x2,x3,x4,x5,x6 = model.encoder(test_image)  
        x7 = model.decoder(x1,x2,x3,x4,x5,x6) 
        predictions = model.segmentation_head(x7)
        predictions = torch.nn.functional.softmax(predictions.squeeze(0), dim=0)  
        predictions = predictions.cpu().detach().numpy()
        predictions=normalize_array(predictions)
        prob_map = predictions[1,:,:]
        mask_pre_p = mask_pruing_soft(prob_map)
        mask_pre_p_list.append(mask_pre_p) 
    return mask_pre_p_list


def prob_to_mask(ceshi_mask_pre_p_list):
    bbox_pre_list = []
    for prob in ceshi_mask_pre_p_list:
        binary_predictions = (prob > 128)
        arr = binary_predictions.astype(np.uint8) 
        bbox_pre = get_bounding_box(arr)
        bbox_pre_list.append(bbox_pre)
    return bbox_pre_list

def make_train_bbox(train_mask_list):
    train_bbox_list=[]
    for i in train_mask_list:
        img = Image.open(i).convert('L') 
        img = np.array(img)
        bbox = get_bounding_box(img)
        train_bbox_list.append(bbox)
    return train_bbox_list







def load_mask_images(mask_dir):
    mask_images = []
    for filename in os.listdir(mask_dir):
        if filename.endswith('.png') or filename.endswith('.jpg'):
            mask_path = os.path.join(mask_dir, filename)
            mask = Image.open(mask_path).convert('L')  
            mask = np.array(mask)
            mask_images.append(mask)
    return mask_images

def calculate_shape_features(mask_images):
    shape_features = []
    for mask in mask_images:
        fg_mask = (mask > 0).astype(np.float32)  
        grad_x = np.gradient(fg_mask, axis=1)
        grad_y = np.gradient(fg_mask, axis=0)
        grad_mag = np.sqrt(grad_x**2 + grad_y**2)
        shape_feature = np.mean(grad_mag)  
        shape_features.append(shape_feature)
    shape_prior = np.mean(shape_features)
    return shape_prior

def get_shape_prior(mask_dir):
    mask_images = load_mask_images(mask_dir)
    shape_prior = calculate_shape_features(mask_images)
    return shape_prior

def gkern(kernlen=17, nsig=3):   # 生成一个尺寸是（16，16）的高斯核
    interval = (2*nsig+1.)/kernlen
    x = np.linspace(-nsig-interval/2., nsig+interval/2., kernlen+1)
    kern1d = np.diff(st.norm.cdf(x))
    kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
    kernel = kernel_raw/kernel_raw.sum()
    return kernel

def min_max_norm(in_):  # 将数据映射到[0,1]，归一化操作
    max_ = in_.max(3)[0].max(2)[0].unsqueeze(2).unsqueeze(3).expand_as(in_)
    min_ = in_.min(3)[0].min(2)[0].unsqueeze(2).unsqueeze(3).expand_as(in_)
    in_ = in_ - min_
    return in_.div(max_-min_+1e-8)
