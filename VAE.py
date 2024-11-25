import os
import time  # For timing epochs
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import nibabel as nib
import torch.nn.functional as F
from glob import glob
import pandas as pd  # For saving results to Excel
import openpyxl

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import h5py  # For saving models as .h5 files

# Import the lr_scheduler
from torch.optim import lr_scheduler

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_nii_file(filepath):
    nii = nib.load(filepath)
    data = nii.get_fdata()
    return np.asarray(data, dtype=np.float32)


class PairedNIIDataset(Dataset):
    def __init__(self, anatomical_dir, fatfraction_dir, liver_mask_dir, indices=None, transform=None):
        self.anatomical_paths = sorted(glob(os.path.join(anatomical_dir, "*.nii")))
        self.fatfraction_paths = sorted(glob(os.path.join(fatfraction_dir, "*.nii")))
        self.liver_mask_paths = sorted(glob(os.path.join(liver_mask_dir, "*.nii")))
        self.transform = transform

        # Verify files are found and matched
        if len(self.anatomical_paths) == 0 or len(self.fatfraction_paths) == 0 or len(self.liver_mask_paths) == 0:
            raise ValueError("No NII files found in the specified directories.")
        if not (len(self.anatomical_paths) == len(self.fatfraction_paths) == len(self.liver_mask_paths)):
            raise ValueError("Mismatch between anatomical, fat fraction, and liver mask NII files.")

        # Prepare slice indexing
        self.slices = []
        for anatomical_path, fatfraction_path, mask_path in zip(self.anatomical_paths, self.fatfraction_paths, self.liver_mask_paths):
            # Extract volume ID from the file name
            volume_id = os.path.basename(anatomical_path)
            volume_id = os.path.splitext(volume_id)[0]  # Remove extension

            anatomical_image = load_nii_file(anatomical_path)
            num_slices = anatomical_image.shape[2]
            for slice_idx in range(num_slices):
                self.slices.append((anatomical_path, fatfraction_path, mask_path, slice_idx, volume_id))

        # If indices are provided, select the slices accordingly
        if indices is not None:
            self.slices = [self.slices[i] for i in indices]

    def __len__(self):
        return len(self.slices)

    def __getitem__(self, idx):
        anatomical_path, fatfraction_path, mask_path, slice_idx, volume_id = self.slices[idx]

        # Load the entire 3D images
        anatomical_image = load_nii_file(anatomical_path)
        fatfraction_image = load_nii_file(fatfraction_path)
        liver_mask = load_nii_file(mask_path)

        # Select the specific slice
        anatomical_slice = anatomical_image[:, :, slice_idx]
        fatfraction_slice = fatfraction_image[:, :, slice_idx]
        mask_slice = liver_mask[:, :, slice_idx]

        # Apply the mask to the slices
        anatomical_slice = anatomical_slice * mask_slice
        fatfraction_slice = fatfraction_slice * mask_slice

        # Normalize each slice to [0, 1]
        # Handle potential division by zero when max value is zero
        max_val_anatomical = anatomical_slice.max()
        if max_val_anatomical > 0:
            anatomical_slice = anatomical_slice / max_val_anatomical
        else:
            anatomical_slice = anatomical_slice  # Slice remains zeros

        max_val_fatfraction = fatfraction_slice.max()
        if max_val_fatfraction > 0:
            fatfraction_slice = fatfraction_slice / max_val_fatfraction
        else:
            fatfraction_slice = fatfraction_slice  # Slice remains zeros

        # Convert to tensor and add channel dimension
        anatomical_slice = torch.from_numpy(anatomical_slice).unsqueeze(0)  # [1, H, W]
        fatfraction_slice = torch.from_numpy(fatfraction_slice).unsqueeze(0)  # [1, H, W]

        # Resize to 128x128
        anatomical_slice = F.interpolate(anatomical_slice.unsqueeze(0), size=(128, 128), mode='bilinear', align_corners=False).squeeze(0)
        fatfraction_slice = F.interpolate(fatfraction_slice.unsqueeze(0), size=(128, 128), mode='bilinear', align_corners=False).squeeze(0)

        return anatomical_slice, fatfraction_slice, volume_id


class Encoder(nn.Module):
    def __init__(self, input_channels=1, latent_dim=100):
        super(Encoder, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=4, stride=2, padding=1),  # Output: 64x64
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # Output: 32x32
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # Output: 16x16
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),  # Output: 8x8
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.fc_mu = nn.Linear(256 * 8 * 8, latent_dim)
        self.fc_logvar = nn.Linear(256 * 8 * 8, latent_dim)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar


class Generator(nn.Module):
    def __init__(self, latent_dim=100, output_channels=1):
        super(Generator, self).__init__()
        self.fc = nn.Linear(latent_dim, 256 * 8 * 8)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # Output: 16x16
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),   # Output: 32x32
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),    # Output: 64x64
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, output_channels, kernel_size=4, stride=2, padding=1),  # Output: 128x128
            nn.Sigmoid()  # Output range [0, 1]
        )

    def forward(self, z):
        x = self.fc(z)
        x = x.view(z.size(0), 256, 8, 8)
        return self.deconv(x)


class Discriminator(nn.Module):
    def __init__(self, input_channels=1):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=4, stride=2, padding=1),  # Output: 64x64
            nn.LeakyReLU(0.2, inplace=True),
            # Optionally reduce the capacity by commenting out a layer
            # nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # Output: 32x32
            # nn.BatchNorm2d(64),
            # nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 128, kernel_size=4, stride=2, padding=1),  # Adjusted layer
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),  # Output: 16x16
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Flatten(),
            nn.Linear(256 * 16 * 16, 1),  # Adjusted input size due to layer change
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.main(x)


def reparameterize(mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std


def display_images(anatomical, fat_fraction, reconstructed, generated, epoch, device):
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    images = [anatomical, fat_fraction, reconstructed, generated]
    titles = ["Original Anatomical", "Original Fat Fraction", "Reconstructed", "Generated"]

    for ax, img, title in zip(axes, images, titles):
        img = img.to(device)
        im = ax.imshow(img.detach().squeeze().cpu().numpy(), cmap="gray")
        ax.axis("off")
        ax.set_title(f"{title}")
        fig.colorbar(im, ax=ax)

    plt.tight_layout()
    filename = f"results.png"
    plt.savefig(filename)
    plt.close()


if __name__ == '__main__':
    # Training loop parameters
    num_epochs = 2  # Adjust as needed
    batch_size = 32  # Adjust as needed
    latent_dim = 100  # Dimension of the latent space for the generator

    # Directories for anatomical, fat fraction MRIs, and liver masks
    anatomical_dir = "C:/Users/Wilson Zhu/Desktop/UCSD Health/UNIT/datasets/10_anatomical/"
    fatfraction_dir = "C:/Users/Wilson Zhu/Desktop/UCSD Health/UNIT/datasets/10fatf/"
    liver_mask_dir = "C:/Users/Wilson Zhu/Desktop/UCSD Health/10wholeliver/"

    # Create the full dataset
    full_dataset = PairedNIIDataset(anatomical_dir, fatfraction_dir, liver_mask_dir)

    # Get the number of slices
    num_slices = len(full_dataset)

    # Create indices and split
    indices = list(range(num_slices))
    np.random.seed(42)  # For reproducibility
    np.random.shuffle(indices)
    split_idx = int(num_slices * 0.8)
    train_indices = indices[:split_idx]
    test_indices = indices[split_idx:]

    # Create training and testing datasets
    train_dataset = PairedNIIDataset(anatomical_dir, fatfraction_dir, liver_mask_dir, indices=train_indices)
    test_dataset = PairedNIIDataset(anatomical_dir, fatfraction_dir, liver_mask_dir, indices=test_indices)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)

    # Initialize models
    encoder = Encoder().to(device)
    generator = Generator().to(device)
    discriminator = Discriminator().to(device)

    # Loss functions
    criterion_gan = nn.BCELoss()
    criterion_recon = nn.MSELoss()

    # Adjusted learning rates and weight decay
    optimizer_E = optim.Adam(encoder.parameters(), lr=0.0001, betas=(0.5, 0.999), weight_decay=1e-5)
    optimizer_G = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999), weight_decay=1e-5)  # Increased LR
    optimizer_D = optim.Adam(discriminator.parameters(), lr=0.00005, betas=(0.5, 0.999), weight_decay=1e-5)  # Decreased LR

    # Learning rate schedulers
    scheduler_E = lr_scheduler.CosineAnnealingLR(optimizer_E, T_max=num_epochs)
    scheduler_G = lr_scheduler.CosineAnnealingLR(optimizer_G, T_max=num_epochs)
    scheduler_D = lr_scheduler.CosineAnnealingLR(optimizer_D, T_max=num_epochs)

    # Initialize lists to store losses
    d_losses = []
    g_losses = []
    recon_losses = []
    kl_losses = []

    # Training Loop
    for epoch in range(num_epochs):
        start_time = time.time()
        encoder.train()
        generator.train()
        discriminator.train()
        epoch_d_loss = 0.0
        epoch_g_loss = 0.0
        epoch_recon_loss = 0.0
        epoch_kl_loss = 0.0
        num_batches = 0

        for anatomical_image, real_fatfraction_image, _ in train_loader:
            # Move images to device
            anatomical_image = anatomical_image.to(device, non_blocking=True)
            real_fatfraction_image = real_fatfraction_image.to(device, non_blocking=True)

            # -------- Train Discriminator --------
            optimizer_D.zero_grad()
            
            # Real images with label smoothing
            real_labels = torch.full((real_fatfraction_image.size(0), 1), 0.9, device=device)
            output_real = discriminator(real_fatfraction_image)
            loss_real = criterion_gan(output_real, real_labels)

            # Fake images
            mu, logvar = encoder(anatomical_image)
            z = reparameterize(mu, logvar)
            fake_fatfraction_image = generator(z)
            fake_labels = torch.zeros((real_fatfraction_image.size(0), 1), device=device)
            output_fake = discriminator(fake_fatfraction_image.detach())
            loss_fake = criterion_gan(output_fake, fake_labels)

            # Discriminator loss and update
            loss_D = (loss_real + loss_fake) * 0.5
            loss_D.backward()
            torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=1.0)
            optimizer_D.step()

            # -------- Train Generator and Encoder --------
            optimizer_E.zero_grad()
            optimizer_G.zero_grad()
            
            # Adversarial loss for generator
            output = discriminator(fake_fatfraction_image)
            gan_loss = criterion_gan(output, real_labels)  # Use real_labels to encourage generator to produce real-like images
            gan_loss_weight = 1.0  # Set to 1.0
            gan_loss = gan_loss * gan_loss_weight

            # Reconstruction loss and KL-divergence loss
            recon_loss_weight = 1.0
            recon_loss = criterion_recon(fake_fatfraction_image, real_fatfraction_image) * recon_loss_weight
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            kl_loss /= anatomical_image.size(0) * 128 * 128

            # Total generator and encoder loss
            loss_G = gan_loss + recon_loss + kl_loss
            loss_G.backward()
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=1.0)
            optimizer_E.step()
            optimizer_G.step()

            # Accumulate losses
            epoch_d_loss += loss_D.item()
            epoch_g_loss += loss_G.item()
            epoch_recon_loss += recon_loss.item()
            epoch_kl_loss += kl_loss.item()
            num_batches += 1

        # Step the schedulers at the end of the epoch
        scheduler_E.step()
        scheduler_G.step()
        scheduler_D.step()

        # Optional: Print the current learning rates
        current_lr_E = optimizer_E.param_groups[0]['lr']
        current_lr_G = optimizer_G.param_groups[0]['lr']
        current_lr_D = optimizer_D.param_groups[0]['lr']

        # Calculate average losses for the epoch
        avg_d_loss = epoch_d_loss / num_batches
        avg_g_loss = epoch_g_loss / num_batches
        avg_recon_loss = epoch_recon_loss / num_batches
        avg_kl_loss = epoch_kl_loss / num_batches

        d_losses.append(avg_d_loss)
        g_losses.append(avg_g_loss)
        recon_losses.append(avg_recon_loss)
        kl_losses.append(avg_kl_loss)

        # End of epoch timing
        end_time = time.time()
        epoch_duration = end_time - start_time

        # Print losses, learning rates, and epoch duration
        print(f"Epoch [{epoch+1}/{num_epochs}], Duration: {epoch_duration:.2f}s, "
              f"D Loss: {avg_d_loss:.4f}, G Loss: {avg_g_loss:.4f}, "
              f"Recon Loss: {avg_recon_loss:.4f}, KL Loss: {avg_kl_loss:.4f}")
        print(f"Learning Rates - LR_E: {current_lr_E:.6f}, LR_G: {current_lr_G:.6f}, LR_D: {current_lr_D:.6f}")

        # Save images every few epochs
        if (epoch + 1) % 10 == 0:
            display_images(
                anatomical_image[0],
                real_fatfraction_image[0],
                fake_fatfraction_image[0],
                fake_fatfraction_image[0],
                epoch,
                device
            )

        # Plot and save loss plots every epoch
        plt.figure(figsize=(10, 5))
        plt.plot(range(1, len(d_losses) + 1), d_losses, label="Discriminator Loss")
        plt.plot(range(1, len(g_losses) + 1), g_losses, label="Generator Loss")
        plt.plot(range(1, len(recon_losses) + 1), recon_losses, label="Reconstruction Loss")
        plt.plot(range(1, len(kl_losses) + 1), kl_losses, label="KL Divergence Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.title("Training Losses")
        plt.savefig(f"training_losses.png")
        plt.close()

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

    # Save the model weights as .h5 files
    with h5py.File('encoder_weights.h5', 'w') as f:
        for key, value in encoder.state_dict().items():
            f.create_dataset(key, data=value.cpu().numpy())

    with h5py.File('generator_weights.h5', 'w') as f:
        for key, value in generator.state_dict().items():
            f.create_dataset(key, data=value.cpu().numpy())

    with h5py.File('discriminator_weights.h5', 'w') as f:
        for key, value in discriminator.state_dict().items():
            f.create_dataset(key, data=value.cpu().numpy())

    # Testing phase
    encoder.eval()
    generator.eval()

    real_values_per_volume = {}
    pred_values_per_volume = {}

    with torch.no_grad():
        for anatomical_image, real_fatfraction_image, volume_ids in test_loader:
            anatomical_image = anatomical_image.to(device)
            real_fatfraction_image = real_fatfraction_image.to(device)

            mu, logvar = encoder(anatomical_image)
            z = reparameterize(mu, logvar)
            fake_fatfraction_image = generator(z)

            volume_id = volume_ids[0]  # Since batch_size=1
            real_slice = real_fatfraction_image[0].cpu().numpy().reshape(-1)
            pred_slice = fake_fatfraction_image[0].cpu().numpy().reshape(-1)

            # Mask out zero values (non-liver regions)
            mask = real_slice > 0  # Assuming zero indicates non-liver
            real_slice = real_slice[mask]
            pred_slice = pred_slice[mask]

            # Initialize lists if not present
            if volume_id not in real_values_per_volume:
                real_values_per_volume[volume_id] = []
                pred_values_per_volume[volume_id] = []

            # Append values
            real_values_per_volume[volume_id].extend(real_slice)
            pred_values_per_volume[volume_id].extend(pred_slice)

    # Compute median and IQR per volume
    results = []
    for volume_id in real_values_per_volume.keys():
        real_values = np.array(real_values_per_volume[volume_id])
        pred_values = np.array(pred_values_per_volume[volume_id])

        real_median = np.median(real_values)
        real_iqr = np.subtract(*np.percentile(real_values, [75, 25]))
        pred_median = np.median(pred_values)
        pred_iqr = np.subtract(*np.percentile(pred_values, [75, 25]))

        results.append({
            'Volume_ID': volume_id,
            'Real_Median': real_median,
            'Real_IQR': real_iqr,
            'Pred_Median': pred_median,
            'Pred_IQR': pred_iqr
        })

    # Save the results to an Excel file
    df = pd.DataFrame(results)
    df.to_excel('test_results.xlsx', index=False)
    print("Test results saved to test_results.xlsx")
