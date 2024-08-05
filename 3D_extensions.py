import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
import nibabel as nib

# Define a 3D CNN model
class CNN3D(nn.Module):
    def __init__(self):
        super(CNN3D, self).__init__()
        self.conv1 = nn.Conv3d(1, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv3d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * 16 * 16 * 8, 128)
        self.fc2 = nn.Linear(128, 2)  # Assuming binary classification

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 16 * 16 * 8)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Create a dataset class for 3D medical images
class Medical3DDataset(Dataset):
    def __init__(self, data_dir, labels_file):
        self.data_dir = data_dir
        self.labels = self.load_labels(labels_file)
        self.data = self.load_data()

    def load_labels(self, labels_file):
        labels = {}
        with open(labels_file, 'r') as f:
            for line in f.readlines():
                parts = line.strip().split(',')
                labels[parts[0]] = int(parts[1])
        return labels

    def load_data(self):
        data = []
        for filename in os.listdir(self.data_dir):
            image_path = os.path.join(self.data_dir, filename)
            image = nib.load(image_path).get_fdata()
            data.append((image, self.labels[filename]))
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image, label = self.data[idx]
        image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)
        return image, torch.tensor(label, dtype=torch.long)

# Train the 3D CNN model
def train_3d_cnn(data_dir, labels_file, num_epochs=10, batch_size=4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = Medical3DDataset(data_dir, labels_file)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = CNN3D().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {running_loss/len(dataloader)}")

if __name__ == "__main__":
    train_data_dir = 'MRNet/train/axial'
    train_labels_file = 'MRNet/train/train-abnormal.csv'
    train_3d_cnn(train_data_dir, train_labels_file)
