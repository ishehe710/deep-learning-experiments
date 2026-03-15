"""
Convolutional Autoencoder Implementation (PyTorch)

This module implements a convolutional autoencoder for learning compact
representations of images using unsupervised learning. The model is designed
for grayscale image datasets such as MNIST (28×28 handwritten digits).

The autoencoder consists of two main components:

Encoder
-------
The encoder compresses the input image into a low-dimensional latent
representation. It uses convolutional layers and max pooling to gradually
reduce the spatial dimensions while increasing feature depth.

Architecture:
    Input (1×28×28)
        ↓
    Conv2d(1 → 16, kernel=5)
        ↓
    ReLU
        ↓
    MaxPool2d(2)
        ↓
    Conv2d(16 → 32, kernel=5)
        ↓
    ReLU
        ↓
    MaxPool2d(2)
        ↓
    Flatten
        ↓
    Linear → latent vector

Decoder
-------
The decoder reconstructs the original image from the latent representation.
It mirrors the encoder using a fully connected layer followed by transposed
convolutions to restore spatial dimensions.

Architecture:
    Latent Vector
        ↓
    Linear
        ↓
    Unflatten
        ↓
    ConvTranspose2d
        ↓
    ReLU
        ↓
    ConvTranspose2d
        ↓
    Sigmoid → reconstructed image

Training
--------
The model is trained to minimize the reconstruction error between the input
image and its reconstruction using Mean Squared Error (MSE) loss.

Loss Function:
    Mean Squared Error (MSE)

Optimizer:
    Adam

Additional utilities are included for:
    - Training the model (`fit`)
    - Evaluating reconstruction loss (`evaluate`)
    - Visualizing original vs reconstructed images (`visualize_result`)

This implementation is intended for educational and experimental purposes,
demonstrating representation learning and basic generative modeling concepts.

Author: Ilagaba Shehe
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision.utils import make_grid

class AutoEncoder(nn.Module):
    def __init__(self, latent_dim=32):    
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(in_features=32 * 4 * 4, out_features=latent_dim)
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(in_features=latent_dim, out_features=32 * 4 * 4),
            nn.Unflatten(1, (32, 4, 4)),
            nn.ConvTranspose2d(32, 16, kernel_size=5, stride=2, padding=0, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, kernel_size=5, stride=2, padding=0, output_padding=1),
            nn.Sigmoid()        
        )

    
    def forward(self, x):

        z = self.encoder(x)

        reconstruction = self.decoder(z)

        return reconstruction
    
    def training_step(self, batch):
        """Returns the loss for a batch of training data"""
        images, _ = batch
        out = self(images)                  # Generate predictions
        loss = F.mse_loss(out, images) # Calculate loss
        return loss
        
    def epoch_end(self, outputs):
        batch_losses = outputs
        epoch_loss = torch.stack(batch_losses).mean()   # Combine losses
        return {'loss': epoch_loss.item()}

    def fit(self, train_loader, val_loader=None, epochs=10, lr=0.001):

        history = []
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        for epoch in range(epochs):

            self.train()  # set model to training mode
            batch_losses = []

            for batch in train_loader:

                loss = self.training_step(batch)

                loss.backward()
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

                batch_losses.append(loss.detach())

            train_loss = torch.stack(batch_losses).mean().item()

            if val_loader:
                val_result = self.evaluate(val_loader)
                print(f"Epoch [{epoch+1}/{epochs}] train_loss: {train_loss:.4f}, val_loss: {val_result['loss']:.4f}")
                history.append((train_loss, val_result['loss']))
            else:
                print(f"Epoch [{epoch+1}/{epochs}] train_loss: {train_loss:.4f}")
                history.append(train_loss)

        return history
    
    def evaluate(self, loader):

        self.eval()  # set evaluation mode

        with torch.no_grad():
            outputs = [self.training_step(batch) for batch in loader]

        return self.epoch_end(outputs)
    
    # one image
    def visualize_result(self, output, original):
        # Create a 1x2 layout
        fig, axs = plt.subplots(1, 2, figsize=(8, 4))
        axs[0].imshow(output.detach().cpu().squeeze().numpy(), cmap='gray')
        axs[0].set_title("Generated Image")
        axs[0].axis('off')  
        
        axs[1].imshow(original.detach().cpu().squeeze().numpy(), cmap='gray')
        axs[1].set_title("Original Image")
        axs[1].axis('off')  
        
        plt.tight_layout()
        plt.show()
        