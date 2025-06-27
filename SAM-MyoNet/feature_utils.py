import cv2
import numpy as np
from PIL import Image
import os
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from scipy.linalg import inv


def get_bbox_feature(label):
    img_array = np.array(label)
    if len(img_array)==3:
        img_array = img_array[0]
    y_indices, x_indices = np.where(img_array > 0)
    x_min, x_max = np.min(x_indices), np.max(x_indices)
    y_min, y_max = np.min(y_indices), np.max(y_indices) 
    center_x = (x_min + x_max) / 2
    center_y = (y_min + y_max) / 2
    aspect_ratio = (x_max-x_min)/(y_max-y_min)
    return center_x/256, center_y/256, aspect_ratio

def calculate_bbox_feature_in_dataset(dataset_folder):
    label_files = [os.path.join(dataset_folder, f) for f in os.listdir(dataset_folder) if f.endswith('.png')] 
    bbox_feature_list = []
    for label_path in label_files:
        label = Image.open(label_path)
        center_x, center_y, aspect_ratio = get_bbox_feature(label)
        bbox_feature_list.append([center_x, center_y, aspect_ratio])
    bbox_feature_array = np.array(bbox_feature_list, dtype=np.float32)
    return bbox_feature_array

def feature_cluster(dataset_folder_path,n_clusters=3):
    bbox_feature_array = calculate_bbox_feature_in_dataset(dataset_folder_path)
    kmeans = KMeans(n_clusters=n_clusters) 
    kmeans.fit(bbox_feature_array)
    labels = kmeans.labels_ 
    cluster_centers = kmeans.cluster_centers_ 
    return bbox_feature_array,cluster_centers,labels



def calculate_label_area(label_path):
    img_array = np.array(label_path)
    label_area = np.sum(img_array!= 0)
    return label_area

def calculate_avg_area_in_dataset(dataset_folder_path):
    image_files = [os.path.join(dataset_folder_path, f) for f in os.listdir(dataset_folder_path) if f.endswith('.png')]
    total_area = 0
    num_images = len(image_files)
    for image_path in image_files:
        img = Image.open(image_path)
        area = calculate_label_area(img)
        total_area += area
    avg_area = total_area / num_images
    return avg_area



def calculate_bbox_sim(label,bbox_feature_array,cluster_centers,labels):
    center_x, center_y, aspect_ratio = get_bbox_feature(label)
    distances = cdist([[center_x, center_y, aspect_ratio]], cluster_centers, metric='euclidean') 
    closest_cluster_index = np.argmin(distances) 
    cluster_center_vector = cluster_centers[closest_cluster_index]
    cluster_data = bbox_feature_array[labels == closest_cluster_index] 
    covariance_matrix = np.cov(cluster_data.T)
    inverse_covariance = inv(covariance_matrix)
    diff = [center_x, center_y, aspect_ratio] - cluster_center_vector
    mahalanobis_distance = np.sqrt(np.dot(np.dot(diff, inverse_covariance), diff.T))
    bbox_sim = 1/(1+mahalanobis_distance) # 相似性计算
    return bbox_sim

def calculate_area_sim(current_area, avg_area):
    area_sim = 1 / (1 + np.abs(current_area - avg_area)/np.maximum(current_area, avg_area))
    return area_sim



def calculate_snr(image):
    img_array = np.array(image)
    signal_power = np.mean(img_array)**2 
    noise_power = np.mean((img_array - np.mean(img_array))**2) 
    snr = signal_power / noise_power 
    scaled_snr = 1 - (1 / (1 + snr)) 
    return scaled_snr







