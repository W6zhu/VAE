import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import nibabel as nib
import torch.nn.functional as F
from glob import glob
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_nii_file(filepath):
    nii = nib.load(filepath)
    data = nii.get_fdata()
    return np.asarray(data, dtype=np.float32)


class PairedNIIDataset(Dataset):
    def __init__(self, anatomical_dir, fatfraction_dir, transform=None):
        self.anatomical_paths = sorted(glob(os.path.join(anatomical_dir, "*.nii")))
        self.fatfraction_paths = sorted(glob(os.path.join(fatfraction_dir, "*.nii")))
        self.transform = transform

        # Verify files are found and matched
        if len(self.anatomical_paths) == 0 or len(self.fatfraction_paths) == 0:
            raise ValueError("No NII files found in the specified directories.")
        if len(self.anatomical_paths) != len(self.fatfraction_paths):
            raise ValueError("Mismatch between anatomical and fat fraction NII files.")
        
        # Prepare slice indexing
        self.slices = []
        for anatomical_path, fatfraction_path in zip(self.anatomical_paths, self.fatfraction_paths):
            anatomical_image = load_nii_file(anatomical_path)
            num_slices = anatomical_image.shape[2]
            for slice_idx in range(num_slices):
                self.slices.append((anatomical_path, fatfraction_path, slice_idx))

    def __len__(self):
        return len(self.slices)

    def __getitem__(self, idx):
        anatomical_path, fatfraction_path, slice_idx = self.slices[idx]
        
        # Load the entire 3D images
        anatomical_image = load_nii_file(anatomical_path)
        fatfraction_image = load_nii_file(fatfraction_path)
        
        # Select the specific slice
        anatomical_slice = anatomical_image[:, :, slice_idx]
        fatfraction_slice = fatfraction_image[:, :, slice_idx]

        # Normalize each slice
        anatomical_slice = (anatomical_slice - anatomical_slice.min()) / (anatomical_slice.max() - anatomical_slice.min())
        fatfraction_slice = (fatfraction_slice - fatfraction_slice.min()) / (fatfraction_slice.max() - fatfraction_slice.min())

        # Convert to tensor, add channel dimension, and pad to 256x256
        anatomical_slice = torch.from_numpy(anatomical_slice).unsqueeze(0)  # [1, H, W]
        fatfraction_slice = torch.from_numpy(fatfraction_slice).unsqueeze(0)  # [1, H, W]

        # Pad to 256x256 if necessary
        anatomical_slice = F.pad(anatomical_slice, (0, 256 - anatomical_slice.shape[2], 0, 256 - anatomical_slice.shape[1]), "constant", 0)
        fatfraction_slice = F.pad(fatfraction_slice, (0, 256 - fatfraction_slice.shape[2], 0, 256 - fatfraction_slice.shape[1]), "constant", 0)

        return anatomical_slice, fatfraction_slice


class Encoder(nn.Module):
    def __init__(self, input_channels=1, latent_dim=100):
        super(Encoder, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.fc_mu = nn.Linear(256 * 16 * 16, latent_dim)
        self.fc_logvar = nn.Linear(256 * 16 * 16, latent_dim)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar


# class Generator(nn.Module):
#     def __init__(self, input_dim=100, output_channels=1):
#         super(Generator, self).__init__()
#         self.main = nn.Sequential(
#             # Input is a latent vector of shape (input_dim, 1, 1)
#             nn.ConvTranspose2d(input_dim, 512, kernel_size=4, stride=1, padding=0),  # [batch, 512, 4, 4]
#             nn.BatchNorm2d(512),
#             nn.ReLU(True),
            
#             nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),  # [batch, 256, 8, 8]
#             nn.BatchNorm2d(256),
#             nn.ReLU(True),
            
#             nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # [batch, 128, 16, 16]
#             nn.BatchNorm2d(128),
#             nn.ReLU(True),
            
#             nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # [batch, 64, 32, 32]
#             nn.BatchNorm2d(64),
#             nn.ReLU(True),
            
#             nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # [batch, 32, 64, 64]
#             nn.BatchNorm2d(32),
#             nn.ReLU(True),
            
#             nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),  # [batch, 16, 128, 128]
#             nn.BatchNorm2d(16),
#             nn.ReLU(True),
            
#             # Final layer to produce 256x256 output
#             nn.ConvTranspose2d(16, output_channels, kernel_size=4, stride=2, padding=1),  # [batch, output_channels, 256, 256]
#             nn.Tanh()  # Output range [-1, 1]
#         )

#     def forward(self, x):
#         return self.main(x)


# Generator (Decoder) Network

class Generator(nn.Module):
    def __init__(self, latent_dim=100, output_channels=1):
        super(Generator, self).__init__()
        self.fc = nn.Linear(latent_dim, 256 * 16 * 16)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, output_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh()  # Output range [-1, 1]
        )

    def forward(self, z):
        x = self.fc(z)
        x = x.view(z.size(0), 256, 16, 16)
        return self.deconv(x)



class Discriminator(nn.Module):
    def __init__(self, input_channels=1):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Flatten(),
            nn.Linear(256 * 16 * 16, 1),
            nn.Sigmoid()  # Output range [0, 1]
        )

    def forward(self, x):
        return self.main(x)


def reparameterize(mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std

# Initialize models
generator = Generator()
discriminator = Discriminator()

# Loss and Optimizers
criterion_gan = nn.BCELoss()  # GAN loss for real vs fake discrimination
criterion_l1 = nn.L1Loss()    # L1 loss for anatomical-to-fatfraction mapping

optimizer_G = optim.Adam(generator.parameters(), lr=0.0001)
optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0001)

# Training loop parameters
num_epochs = 250
batch_size = 4
latent_dim = 100  # Dimension of the latent space for the generator
noise = torch.randn(batch_size, latent_dim, 1, 1)

# Directories for anatomical and fatfraction MRIs
anatomical_dir = "C:/Users/Wilson Zhu/Desktop/UCSD Health/UNIT/datasets/10_anatomical/"  # Replace with your anatomical MRI directory
fatfraction_dir = "C:/Users/Wilson Zhu/Desktop/UCSD Health/UNIT/datasets/10fatf/"  # Replace with your fat fraction MRI directory

# Print file paths to verify contents
print("Anatomical Files:", sorted(glob(os.path.join(anatomical_dir, "*.nii"))))
print("Fat Fraction Files:", sorted(glob(os.path.join(fatfraction_dir, "*.nii"))))


# Initialize Dataset and DataLoader
dataset = PairedNIIDataset(anatomical_dir, fatfraction_dir)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

def display_images(anatomical, fat_fraction, reconstructed, generated, epoch, step, device):
    fig, axes = plt.subplots(1, 4, figsize=(15, 5))
    images = [anatomical, fat_fraction, reconstructed, generated]
    titles = ["Original Anatomical", "Original Fat Fraction", "Reconstructed", "Generated"]

    for ax, img, title in zip(axes, images, titles):
        img = img.to(device)
        ax.imshow(img.detach().squeeze().cpu().numpy(), cmap="gray")
        ax.axis("off")
        ax.set_title(f"{title}")
    
    plt.tight_layout()
    plt.savefig("results.png")  # Save the image
    plt.close()

    plt.show()



# Loss functions
criterion_gan = nn.BCELoss()
criterion_recon = nn.MSELoss()  # Reconstruction loss

# Initialize models
encoder = Encoder().to(device)
generator = Generator().to(device)
discriminator = Discriminator().to(device)

# Optimizers with adjusted learning rates
optimizer_E = optim.Adam(encoder.parameters(), lr=0.0001, betas=(0.5, 0.999))
optimizer_G = optim.Adam(generator.parameters(), lr=0.0001, betas=(0.5, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=0.00005, betas=(0.5, 0.999))  # Reduced learning rate

d_losses = []
g_losses = []
recon_losses = []
kl_losses = []


# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # Move models to device
# generator.to(device)
# discriminator.to(device)

# # Define optimizers with adjusted learning rates
# optimizer_G = optim.Adam(generator.parameters(), lr=0.0001, betas=(0.5, 0.999))
# optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0001, betas=(0.5, 0.999))

# # Loss function
# criterion_gan = nn.BCELoss()

# # Function to display images
# def display_images(anatomical, fat_fraction, reconstructed, generated, epoch, step, device):
#     fig, axes = plt.subplots(1, 4, figsize=(15, 5))
#     images = [anatomical, fat_fraction, reconstructed, generated]
#     titles = ["Original Anatomical", "Original Fat Fraction", "Reconstructed", "Generated"]

#     for ax, img, title in zip(axes, images, titles):
#         img = img.to(device)  # Ensure image is on the GPU
#         ax.imshow(img.detach().squeeze().cpu().numpy(), cmap="gray")  # Detach, move to CPU, and convert to NumPy
#         ax.axis("off")
#         ax.set_title(f"{title} (Epoch {epoch}, Step {step})")

#     plt.show()

# # Training Loop
# for epoch in range(num_epochs):
#     for i, (anatomical_image, real_fatfraction_image) in enumerate(dataloader):
        
#         # Move images to the same device as the model
#         anatomical_image = anatomical_image.to(device)
#         real_fatfraction_image = real_fatfraction_image.to(device)

#         # -------- Train Discriminator (more frequently if needed) --------
#         for _ in range(2):  # Train the discriminator twice per generator update
#             optimizer_D.zero_grad()
            
#             # Real images with label smoothing
#             real_labels = torch.ones((real_fatfraction_image.size(0), 1), device=device) * 0.9
#             output_real = discriminator(real_fatfraction_image)
#             loss_real = criterion_gan(output_real, real_labels)

#             # Fake images generated from random noise
#             noise = torch.randn(real_fatfraction_image.size(0), 100, 1, 1).to(device)
#             fake_fatfraction_image = generator(noise)
#             fake_labels = torch.zeros((real_fatfraction_image.size(0), 1), device=device) + 0.1
#             output_fake = discriminator(fake_fatfraction_image.detach())
#             loss_fake = criterion_gan(output_fake, fake_labels)

#             # Discriminator loss and update
#             loss_D = (loss_real + loss_fake) * 0.5
#             loss_D.backward()
#             optimizer_D.step()

#         # -------- Train Generator --------
#         optimizer_G.zero_grad()
        
#         # Generate fake images and compute GAN loss
#         output = discriminator(fake_fatfraction_image)
#         loss_G = criterion_gan(output, real_labels)  # Use real labels to trick discriminator
#         loss_G.backward()
#         optimizer_G.step()

#         # Print losses occasionally
#         if i % 10 == 0:
#             print(f"Epoch [{epoch}/{num_epochs}], Step [{i}/{len(dataloader)}], "
#                   f"D Loss: {loss_D.item():.4f}, G Loss: {loss_G.item():.4f}")

#         # Display and/or save images every few steps/epochs
#         if i % 50 == 0:  # Adjust interval as needed
#             display_images(
#                 anatomical_image[0],              # Original anatomical image
#                 real_fatfraction_image[0],        # Original fat fraction image
#                 fake_fatfraction_image[0],        # Generated fat fraction image
#                 fake_fatfraction_image[0],        # Display as reconstructed if applicable
#                 epoch,
#                 i,
#                 device
#             )


# Training Loop
for epoch in range(num_epochs):
    epoch_d_loss = 0.0
    epoch_g_loss = 0.0
    epoch_recon_loss = 0.0
    epoch_kl_loss = 0.0
    num_batches = 0

    for i, (anatomical_image, real_fatfraction_image) in enumerate(dataloader):
        # Move images to device
        anatomical_image = anatomical_image.to(device)
        real_fatfraction_image = real_fatfraction_image.to(device)

        # -------- Train Discriminator --------
        optimizer_D.zero_grad()
        
        # Real images with label smoothing
        real_labels = torch.ones((real_fatfraction_image.size(0), 1), device=device) * 0.9
        output_real = discriminator(real_fatfraction_image)
        loss_real = criterion_gan(output_real, real_labels)

        # Fake images
        mu, logvar = encoder(anatomical_image)
        z = reparameterize(mu, logvar)
        fake_fatfraction_image = generator(z)
        fake_labels = torch.zeros((real_fatfraction_image.size(0), 1), device=device) + 0.1
        output_fake = discriminator(fake_fatfraction_image.detach())
        loss_fake = criterion_gan(output_fake, fake_labels)

        # Discriminator loss and update
        loss_D = (loss_real + loss_fake) * 0.5
        loss_D.backward()
        optimizer_D.step()

        # -------- Train Generator and Encoder --------
        optimizer_E.zero_grad()
        optimizer_G.zero_grad()
        
        # Adversarial loss for generator
        output = discriminator(fake_fatfraction_image)
        gan_loss = criterion_gan(output, real_labels)

        # Reconstruction loss and KL-divergence loss
        recon_loss = criterion_recon(fake_fatfraction_image, real_fatfraction_image)
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        kl_loss /= anatomical_image.size(0) * 256 * 256  # Normalize by batch and image size

        # Total generator and encoder loss
        loss_G = gan_loss + recon_loss + kl_loss
        loss_G.backward()
        optimizer_E.step()
        optimizer_G.step()

        # Accumulate losses for the epoch
        epoch_d_loss += loss_D.item()
        epoch_g_loss += loss_G.item()
        epoch_recon_loss += recon_loss.item()
        epoch_kl_loss += kl_loss.item()
        num_batches += 1

        # Print losses occasionally
        if i % 10 == 0:
            print(f"Epoch [{epoch}/{num_epochs}], Step [{i}/{len(dataloader)}], "
                  f"D Loss: {loss_D.item():.4f}, G Loss: {loss_G.item():.4f}, Recon Loss: {recon_loss.item():.4f}, KL Loss: {kl_loss.item():.4f}")

        # Display and/or save images every few steps/epochs
        if i % 50 == 0:
            display_images(
                anatomical_image[0],
                real_fatfraction_image[0],
                fake_fatfraction_image[0],
                fake_fatfraction_image[0],
                epoch,
                i,
                device
            )

    # Calculate average losses for the epoch
    d_losses.append(epoch_d_loss / num_batches)
    g_losses.append(epoch_g_loss / num_batches)
    recon_losses.append(epoch_recon_loss / num_batches)
    kl_losses.append(epoch_kl_loss / num_batches)

    # Plot the losses
    plt.figure(figsize=(10, 5))
    plt.plot(d_losses, label="Discriminator Loss")
    plt.plot(g_losses, label="Generator Loss")
    plt.plot(recon_losses, label="Reconstruction Loss")
    plt.plot(kl_losses, label="KL Divergence Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training Losses")
    plt.savefig("training_losses.png")  # Save the plot to a file
    plt.close()  

        # Plot and save separate loss graphs in one image
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))

    # Plot each loss on a separate subplot
    axs[0, 0].plot(d_losses, label='Discriminator Loss', color='red')
    axs[0, 0].set_title('Discriminator Loss')
    axs[0, 0].set_xlabel('Epochs')
    axs[0, 0].set_ylabel('Loss')
    axs[0, 0].legend()
    axs[0, 0].grid(True)

    axs[0, 1].plot(g_losses, label='Generator Loss', color='blue')
    axs[0, 1].set_title('Generator Loss')
    axs[0, 1].set_xlabel('Epochs')
    axs[0, 1].set_ylabel('Loss')
    axs[0, 1].legend()
    axs[0, 1].grid(True)

    axs[1, 0].plot(recon_losses, label='Reconstruction Loss', color='green')
    axs[1, 0].set_title('Reconstruction Loss')
    axs[1, 0].set_xlabel('Epochs')
    axs[1, 0].set_ylabel('Loss')
    axs[1, 0].legend()
    axs[1, 0].grid(True)

    axs[1, 1].plot(kl_losses, label='KL Divergence Loss', color='purple')
    axs[1, 1].set_title('KL Divergence Loss')
    axs[1, 1].set_xlabel('Epochs')
    axs[1, 1].set_ylabel('Loss')
    axs[1, 1].legend()
    axs[1, 1].grid(True)

    plt.tight_layout()
    plt.savefig("loss_plots.png")  # Save the plot to a file
    plt.close()  






