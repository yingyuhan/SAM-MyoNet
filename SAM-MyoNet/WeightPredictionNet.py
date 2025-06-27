
import torch
import torch.nn as nn



class WeightPredictionNet(nn.Module):  
    def __init__(self):
        super(WeightPredictionNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc_num1 = nn.Linear(1, 8)
        self.relu_num1 = nn.ReLU()
        self.fc_num2 = nn.Linear(1, 8)
        self.relu_num2 = nn.ReLU()
        self.fc_num3 = nn.Linear(1, 8)
        self.relu_num3 = nn.ReLU()

        self.fc_img_compress = nn.Linear(32 * (1024 // 4) * (1024 // 4), 32)
        self.fc_combined = nn.Linear(32 + 8 * 3, 64)
        self.relu_combined = nn.ReLU()
        self.fc_output = nn.Linear(64, 1)

    def forward(self, image, num1, num2, num3):
        img_features = self.pool2(self.relu2(self.conv2(self.pool1(self.relu1(self.conv1(image))))))
        img_features = img_features.view(-1, 32 * (1024 // 4) * (1024 // 4))
        img_features = self.fc_img_compress(img_features)

        num1 = num1.to(img_features.dtype).unsqueeze(1)
        num2 = num2.to(img_features.dtype).unsqueeze(1)
        num3 = num3.to(img_features.dtype).unsqueeze(1)

        num_features1 = self.relu_num1(self.fc_num1(num1))
        num_features2 = self.relu_num2(self.fc_num2(num2))
        num_features3 = self.relu_num3(self.fc_num3(num3))
        combined_features = torch.cat((img_features, num_features1, num_features2, num_features3), dim=1)
        raw_weight = self.fc_output(self.relu_combined(self.fc_combined(combined_features)))

        return torch.sigmoid(raw_weight) + 1e-6