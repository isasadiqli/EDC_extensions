import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder2D(nn.Module):
    def __init__(self):
        super(Encoder2D, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(128 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, 256)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 128 * 8 * 8)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class Decoder2D(nn.Module):
    def __init__(self):
        super(Decoder2D, self).__init__()
        self.fc1 = nn.Linear(256, 512)
        self.fc2 = nn.Linear(512, 128 * 8 * 8)
        self.deconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2, padding=0)
        self.deconv2 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2, padding=0)
        self.deconv3 = nn.ConvTranspose2d(32, 1, kernel_size=2, stride=2, padding=0)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = x.view(-1, 128, 8, 8)
        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        x = torch.sigmoid(self.deconv3(x))
        return x


class EDC2D(nn.Module):
    def __init__(self):
        super(EDC2D, self).__init__()
        self.encoder = Encoder2D()
        self.decoder = Decoder2D()

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

    def compute_loss(self, x, reconstructed):
        return F.mse_loss(reconstructed, x)

    def global_cosine_distance_loss(self, x, y):
        x_flat = x.view(x.size(0), -1)
        y_flat = y.view(y.size(0), -1)
        cos_sim = F.cosine_similarity(x_flat, y_flat)
        return 1 - cos_sim.mean()

    def contrastive_loss(self, x, reconstructed):
        with torch.no_grad():
            encoded_x = self.encoder(x)
        encoded_reconstructed = self.encoder(reconstructed)
        return self.global_cosine_distance_loss(encoded_x, encoded_reconstructed)

    def edc_loss(self, x, reconstructed):
        reconstruction_loss = self.compute_loss(x, reconstructed)
        contrastive_loss = self.contrastive_loss(x, reconstructed)
        return reconstruction_loss + contrastive_loss
