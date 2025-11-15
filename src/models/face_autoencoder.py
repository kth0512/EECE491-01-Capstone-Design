import torch
import torch.nn as nn
import torchvision.models as models

class FaceAutoencoder(nn.Module):
    def __init__(self, latent_dim=256, image_size=128):
        super(FaceAutoencoder, self).__init__()

        # Encoder: Pretrained ResNet-18
        resnet = models.resnet18(weights='IMAGENET1K_V1')
        # Obtain feature map
        modules = list(resnet.children())[:-2]


        self.encoder_cnn = nn.Sequential(*modules)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc_encode = nn.Linear(resnet.fc.in_features, latent_dim)
        
        self.fc_decode = nn.Linear(latent_dim, 512 * 4 * 4)
        self.decoder_cnn = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1), nn.ReLU(), nn.BatchNorm2d(256), # -> 8x8
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1), nn.ReLU(), nn.BatchNorm2d(128), # -> 16x16
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  nn.ReLU(), nn.BatchNorm2d(64),  # -> 32x32
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),   nn.ReLU(), nn.BatchNorm2d(32),  # -> 64x64
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),    nn.Tanh()                       # -> 128x128
        )


    def forward(self, x):
        x = self.encode(x)
        x = self.decode(x)
        return x

    def encode(self, x):
        # Image -> Feature vector
        x = self.encoder_cnn(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        encoded = self.fc_encode(x)
        return encoded

    def decode(self, x):
        # Feature Vector -> Reconstructed Image
        x = self.fc_decode(x)
        x = x.view(-1, 512, 4, 4)
        decoded = self.decoder_cnn(x)
        return decoded