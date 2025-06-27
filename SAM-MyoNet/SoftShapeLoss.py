import torch
import torch.nn as nn
import torch.nn.functional as F

class SoftShapeLoss(nn.Module):  # 使用shape_prior作为约束项
    def __init__(self, alpha, beta, gamma, shape_prior=None):
        super(SoftShapeLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.ce_loss = nn.CrossEntropyLoss()
        self.shape_prior = shape_prior  # shape_prior仍然是固定的

    def gradient_map(self, img):
        # Sobel算子，提取梯度边缘（可导）
        sobel_x = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=torch.float32, device=img.device).view(1, 1, 3, 3) / 8.0
        sobel_y = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32, device=img.device).view(1, 1, 3, 3) / 8.0

        img = img.unsqueeze(1)  # [B, 1, H, W]
        grad_x = F.conv2d(img, sobel_x, padding=1)
        grad_y = F.conv2d(img, sobel_y, padding=1)
        grad = torch.sqrt(grad_x**2 + grad_y**2 + 1e-6)  # [B, 1, H, W]
        return grad

    def forward(self, y_pred, y_true):
        loss_ce = self.ce_loss(y_pred, y_true)  # 标准交叉熵损失

        # --- Shape loss ---
        y_pred_soft = F.softmax(y_pred, dim=1)  # [B, C, H, W]
        fg_pred = y_pred_soft[:, 1, :, :]  # [B, H, W]
        fg_true = (y_true > 0).float()  # [B, H, W]

        # 计算边缘
        pred_edge = self.gradient_map(fg_pred)  # 使用self来调用实例方法
        true_edge = self.gradient_map(fg_true)  # 使用self来调用实例方法

        # 计算边缘损失
        shape_loss = F.mse_loss(pred_edge, true_edge)

        # 如果提供了shape_prior，计算其与训练结果的差异
        shape_prior_loss = 0.0
        if self.shape_prior is not None:
            
            fg_pred_shape = self.calculate_shape(fg_pred)
            shape_prior_loss = F.mse_loss(fg_pred_shape, self.shape_prior)

        total_loss = self.alpha * loss_ce + 100 * self.beta * shape_loss + 1000 * self.gamma * shape_prior_loss
        return total_loss

    def calculate_shape(self, fg_pred):
        # 使用平均边缘梯度作为形状特征
        return self.gradient_map(fg_pred).mean(dim=[2, 3])