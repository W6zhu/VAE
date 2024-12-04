import os
import torch
import torch.nn as nn
import numpy as np
import nibabel as nib
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pandas as pd
import h5py
import collections
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the Encoder model
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

# Define the Generator model
class Generator(nn.Module):
    def __init__(self, latent_dim=100, output_channels=1):
        super(Generator, self).__init__()
        self.fc = nn.Linear(latent_dim, 256 * 8 * 8)

        # Encoder (downsampling path)
        self.enc1 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),  # 8x8
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),  # 8x8 -> 4x4
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )

        # Decoder (upsampling path)
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),  # 4x4 -> 8x8
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),   # 8x8 -> 16x16
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),    # 16x16 -> 32x32
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.dec4 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),     # 32x32 -> 64x64
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.final = nn.Sequential(
            nn.ConvTranspose2d(32, output_channels, kernel_size=4, stride=2, padding=1),  # 64x64 -> 128x128
            nn.Tanh()  # Output range [-1, 1]. Consider replacing with Sigmoid() to constrain to [0,1]
        )

    def forward(self, z):
        x = self.fc(z)
        x = x.view(z.size(0), 256, 8, 8)

        enc1 = self.enc1(x)  # 8x8
        enc2 = self.enc2(enc1)  # 4x4

        dec1 = self.dec1(enc2)  # 8x8
        dec1 = dec1 + enc1  # Skip connection

        dec2 = self.dec2(dec1)  # 16x16
        dec3 = self.dec3(dec2)  # 32x32
        dec4 = self.dec4(dec3)  # 64x64
        out = self.final(dec4)  # 128x128

        return out

def reparameterize(mu, logvar):
    """
    Reparameterization trick to sample from N(mu, var) from N(0,1).
    """
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std

def load_model_weights(model, weight_file):
    """
    Load model weights from an .h5 file.
    """
    state_dict = collections.OrderedDict()
    with h5py.File(weight_file, 'r') as f:
        for key in f.keys():
            # Handle scalar datasets
            weight = f[key][()]
            if weight.ndim == 0:
                weight = np.array(weight)
            state_dict[key] = torch.from_numpy(weight)
    model.load_state_dict(state_dict)

def load_nii_file(filepath):
    """
    Load a NIfTI file and return its data, affine, and header.
    """
    nii = nib.load(filepath)
    data = nii.get_fdata()
    return np.asarray(data, dtype=np.float32), nii.affine, nii.header

def get_patient_list(anatomical_dir, fat_fraction_dir, mask_dir):
    """
    Get a list of patient IDs by matching files across directories.
    Only include patients that have corresponding files in all three directories.
    """
    anatomical_files = set(os.listdir(anatomical_dir))
    fat_fraction_files = set(os.listdir(fat_fraction_dir))
    mask_files = set(os.listdir(mask_dir))

    # Find common files across all directories
    common_files = anatomical_files & fat_fraction_files & mask_files

    # Extract patient IDs by removing file extensions
    patient_ids = [os.path.splitext(f)[0] for f in common_files]

    return sorted(patient_ids)

def process_patient(patient_id, anatomical_dir, fat_fraction_dir, mask_dir, output_dir, encoder, generator, scaling_factor=100.0):
    """
    Process a single patient: load data, perform normalization, model inference,
    compute statistics, and save results.
    """
    # Define file paths
    anatomical_path = os.path.join(anatomical_dir, f"{patient_id}.nii")
    fat_fraction_path = os.path.join(fat_fraction_dir, f"{patient_id}.nii")
    mask_path = os.path.join(mask_dir, f"{patient_id}.nii")

    # Load images
    try:
        anatomical_image, anatomical_affine, anatomical_header = load_nii_file(anatomical_path)
        fat_fraction_image, fat_fraction_affine, fat_fraction_header = load_nii_file(fat_fraction_path)
        liver_mask, mask_affine, mask_header = load_nii_file(mask_path)
        logging.info(f"Loaded NIfTI files for patient {patient_id}.")
    except Exception as e:
        logging.error(f"Error loading NIfTI files for patient {patient_id}: {e}")
        return  # Skip this patient

    # Normalize fat fraction data to [0,1]
    fat_fraction_image = np.clip(fat_fraction_image, 0, scaling_factor)  # Clip to [0, scaling_factor]
    fat_fraction_image_norm = fat_fraction_image / scaling_factor  # Normalize to [0,1]
    logging.info(f"Patient {patient_id}: Fat fraction data normalized to [0,1].")

    # Get spatial dimensions
    if anatomical_image.ndim != 3:
        logging.error(f"Patient {patient_id}: Anatomical image does not have 3 dimensions.")
        return
    original_height, original_width, num_slices = anatomical_image.shape
    logging.info(f"Patient {patient_id}: Image dimensions - Height: {original_height}, Width: {original_width}, Slices: {num_slices}.")

    # Initialize a 3D array for the generated volume
    generated_volume = np.zeros_like(fat_fraction_image_norm, dtype=np.float32)

    # Initialize lists to accumulate real and predicted fat fraction values
    real_values = []
    pred_values = []

    per_slice_stats = []  # To store per-slice statistics

    for slice_idx in range(num_slices):
        slice_number = slice_idx + 1  # Slices start at 1

        anatomical_slice = anatomical_image[:, :, slice_idx]
        fat_fraction_slice = fat_fraction_image_norm[:, :, slice_idx]  # [0,1]
        mask_slice = liver_mask[:, :, slice_idx]  # Binary mask

        # Apply the mask
        anatomical_slice_masked = anatomical_slice * mask_slice
        fat_fraction_slice_masked = fat_fraction_slice * mask_slice

        # Check if the slice is empty after masking
        if np.sum(mask_slice) == 0:
            logging.warning(f"Patient {patient_id}, Slice {slice_number}: Empty after masking. Recording NaN statistics.")
            per_slice_stats.append({
                'File_Name': patient_id,
                'Slice': slice_number,
                'Original_Median': np.nan,
                'Original_IQR': np.nan,
                'Generated_Median': np.nan,
                'Generated_IQR': np.nan
            })
            # Assign zeros to generated_volume for empty slices
            generated_volume[:, :, slice_idx] = 0
            continue  # Skip processing for empty slices

        # Normalize slices to [-1, 1] for model input
        fat_fraction_slice_norm = fat_fraction_slice_masked * 2 - 1  # [-1,1]
        max_val_anatomical = anatomical_slice_masked.max()
        if max_val_anatomical > 0:
            anatomical_slice_norm = (anatomical_slice_masked / max_val_anatomical) * 2 - 1  # [-1,1]
        else:
            anatomical_slice_norm = anatomical_slice_masked  # Remains zeros

        # Convert to tensor and add channel dimensions
        anatomical_slice_tensor = torch.from_numpy(anatomical_slice_norm).unsqueeze(0).unsqueeze(0).float().to(device)  # [1,1,H,W]
        fat_fraction_slice_tensor = torch.from_numpy(fat_fraction_slice_norm).unsqueeze(0).unsqueeze(0).float().to(device)  # [1,1,H,W]

        # Resize to 128x128 for model input
        anatomical_slice_resized = F.interpolate(anatomical_slice_tensor, size=(128, 128), mode='bilinear', align_corners=False)
        fat_fraction_slice_resized = F.interpolate(fat_fraction_slice_tensor, size=(128, 128), mode='bilinear', align_corners=False)
        mask_slice_resized = F.interpolate(
            torch.from_numpy(mask_slice).unsqueeze(0).unsqueeze(0).float(),
            size=(128, 128),
            mode='nearest'
        ).squeeze().numpy()

        # Model Inference
        with torch.no_grad():
            mu, logvar = encoder(anatomical_slice_resized)
            z = reparameterize(mu, logvar)
            generated_slice = generator(z)

        # Rescale generated slice from [-1, 1] to [0,1]
        generated_slice_np = generated_slice.cpu().squeeze().numpy()  # [128,128]
        generated_slice_rescaled = (generated_slice_np + 1) / 2  # [0,1]

        # Rescale to original fat fraction scale using fixed scaling_factor (100.0)
        generated_slice_scaled = generated_slice_rescaled * scaling_factor  # [0,100]

        # Clip the values to maintain within the original data range [0,100]
        generated_slice_scaled = np.clip(generated_slice_scaled, 0, scaling_factor)

        # Apply the mask to the generated slice
        generated_slice_masked = generated_slice_scaled * mask_slice_resized  # [0,100]

        # Resize generated_slice_masked back to original spatial dimensions (e.g., 224x224)
        generated_slice_masked_tensor = torch.from_numpy(generated_slice_masked).unsqueeze(0).unsqueeze(0).float().to(device)  # [1,1,128,128]
        generated_slice_resized_back = F.interpolate(generated_slice_masked_tensor, size=(original_height, original_width), mode='bilinear', align_corners=False)
        generated_slice_final = generated_slice_resized_back.cpu().squeeze().numpy()  # [H,W]

        # Place the generated slice into the generated_volume
        # Normalize back to [0,1] for storage consistency by dividing by scaling_factor
        generated_volume[:, :, slice_idx] = generated_slice_final / scaling_factor  # [0,1]

        # **Optional:** Print min and max for each generated slice
        slice_min = generated_slice_final.min()
        slice_max = generated_slice_final.max()
        logging.debug(f"Patient {patient_id}, Slice {slice_number}: Generated Fat Fraction Min: {slice_min}, Max: {slice_max}")

        # Prepare images for display (rescale to [0,1] for visualization)
        anatomical_display = (anatomical_slice_resized.cpu().squeeze().numpy() + 1) / 2  # [128,128]
        fat_fraction_display = (fat_fraction_slice_resized.cpu().squeeze().numpy() + 1) / 2  # [128,128]
        generated_display = generated_volume[:, :, slice_idx]  # [0,1]

        # Resize anatomical_display and fat_fraction_display back to [H,W]
        anatomical_display_resized = F.interpolate(
            torch.from_numpy(anatomical_display).unsqueeze(0).unsqueeze(0).float(),
            size=(original_height, original_width),
            mode='bilinear',
            align_corners=False
        ).squeeze().numpy() * mask_slice  # [H,W]

        fat_fraction_display_resized = F.interpolate(
            torch.from_numpy(fat_fraction_display).unsqueeze(0).unsqueeze(0).float(),
            size=(original_height, original_width),
            mode='bilinear',
            align_corners=False
        ).squeeze().numpy() * mask_slice  # [H,W]

        # Multiply generated_display by mask_slice [H,W]
        generated_display = generated_display * mask_slice  # [H,W]

        # Create a figure with 3 subplots side by side
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        images = [anatomical_display_resized, fat_fraction_display_resized, generated_display]
        titles = [f'Anatomical Slice {slice_number}', f'Fat Fraction Slice {slice_number}', f'Generated Slice {slice_number}']

        for ax, img, title in zip(axes, images, titles):
            im = ax.imshow(img, cmap='gray', vmin=0, vmax=1)
            ax.set_title(title)
            ax.axis('off')
            fig.colorbar(im, ax=ax)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout to accommodate suptitle

        # Save the figure
        image_filename = f"{patient_id}_Slice{slice_number}.png"
        plt.savefig(os.path.join(output_dir, image_filename))
        plt.close()

        # **Rescale Normalized Data Back to Original Scale for Statistics**
        # fat_fraction_slice_norm is in [-1,1], so rescale to [0,1]
        fat_fraction_slice_rescaled = (fat_fraction_slice_norm + 1) / 2  # [0,1]

        # **Compute Statistics on Normalized Data [0,1]**
        # Flatten and apply mask for original fat fraction
        real_slice_flat = fat_fraction_slice_rescaled.reshape(-1)  # [0,1]
        original_mask_flat = mask_slice.reshape(-1)  # [H,W] -> flat

        # Flatten and apply mask for generated fat fraction
        pred_slice_flat = generated_volume[:, :, slice_idx].reshape(-1)  # [0,1]

        # Keep only the liver region (mask > 0)
        real_slice_values = real_slice_flat[original_mask_flat > 0]  # [0,1]
        pred_slice_values = pred_slice_flat[original_mask_flat > 0]  # [0,1]

        # Append to the overall lists
        real_values.extend(real_slice_values.tolist())
        pred_values.extend(pred_slice_values.tolist())

        # Compute per-slice median and IQR
        if len(real_slice_values) > 0:
            original_median_slice = np.median(real_slice_values)
            original_iqr_slice = np.percentile(real_slice_values, 75) - np.percentile(real_slice_values, 25)
        else:
            original_median_slice = np.nan
            original_iqr_slice = np.nan
            logging.warning(f"Patient {patient_id}, Slice {slice_number}: No valid real values to compute median and IQR.")

        if len(pred_slice_values) > 0:
            generated_median_slice = np.median(pred_slice_values)
            generated_iqr_slice = np.percentile(pred_slice_values, 75) - np.percentile(pred_slice_values, 25)
        else:
            generated_median_slice = np.nan
            generated_iqr_slice = np.nan
            logging.warning(f"Patient {patient_id}, Slice {slice_number}: No valid generated values to compute median and IQR.")

        # Store per-slice statistics
        per_slice_stats.append({
            'File_Name': patient_id,
            'Slice': slice_number,
            'Original_Median': original_median_slice,
            'Original_IQR': original_iqr_slice,
            'Generated_Median': generated_median_slice,
            'Generated_IQR': generated_iqr_slice
        })

    # Compute overall IQR and median for the entire volume
    real_values_array = np.array(real_values)
    pred_values_array = np.array(pred_values)

    # Handle cases where arrays might be empty
    if len(real_values_array) > 0:
        original_median = np.median(real_values_array)
        original_iqr = np.percentile(real_values_array, 75) - np.percentile(real_values_array, 25)
    else:
        original_median = np.nan
        original_iqr = np.nan
        logging.warning(f"Patient {patient_id}: No valid real values to compute overall median and IQR.")

    if len(pred_values_array) > 0:
        generated_median = np.median(pred_values_array)
        generated_iqr = np.percentile(pred_values_array, 75) - np.percentile(pred_values_array, 25)
    else:
        generated_median = np.nan
        generated_iqr = np.nan
        logging.warning(f"Patient {patient_id}: No valid generated values to compute overall median and IQR.")

    # Append volume-level statistics
    per_slice_stats.append({
        'File_Name': patient_id,
        'Slice': 'Volume-Level',
        'Original_Median': original_median,
        'Original_IQR': original_iqr,
        'Generated_Median': generated_median,
        'Generated_IQR': generated_iqr
    })

    # Save the generated volume as a NIfTI file
    try:
        generated_nii = nib.Nifti1Image(generated_volume, affine=fat_fraction_affine, header=fat_fraction_header)
        generated_nii_path = os.path.join(output_dir, f"{patient_id}_generated.nii")
        nib.save(generated_nii, generated_nii_path)
        logging.info(f"Patient {patient_id}: Generated volume saved at {generated_nii_path}.")
    except Exception as e:
        logging.error(f"Patient {patient_id}: Error saving generated NIfTI file: {e}")

    # Save the per-slice statistics to a CSV file
    df_per_slice = pd.DataFrame(per_slice_stats)

    # Define CSV file path
    csv_path = os.path.join(output_dir, f"{patient_id}_per_slice_statistics.csv")

    # Save to CSV with exception handling
    try:
        df_per_slice.to_csv(csv_path, index=False)
        logging.info(f"Patient {patient_id}: Per-slice statistics saved at {csv_path}.")
    except PermissionError as pe:
        logging.error(f"Patient {patient_id}: Permission denied while saving CSV file: {pe}")
        logging.info("Ensure the file is not open in another application and that you have write permissions.")
    except Exception as e:
        logging.error(f"Patient {patient_id}: An error occurred while saving CSV file: {e}")

def main():
    # Define directories
    anatomical_dir = 'C:/Users/Wilson Zhu/Desktop/UCSD Health/Full_Dataset/Adult_Anatomical/'    # Directory containing anatomical NIfTI files
    fat_fraction_dir = 'C:/Users/Wilson Zhu/Desktop/UCSD Health/Full_Dataset/Adult_FatFrac/'  # Directory containing fat fraction NIfTI files
    mask_dir = 'C:/Users/Wilson Zhu/Desktop/UCSD Health/Full_Dataset/Adult_Whole_Liver_Seg/'                # Directory containing mask NIfTI files
    output_dir = 'finalResults'                      # Output directory
    sub_dir = 'allan run'                      # Directory containing model weights
    os.makedirs(output_dir, exist_ok=True)

    # Get list of all patients by matching files across directories
    patient_ids = get_patient_list(anatomical_dir, fat_fraction_dir, mask_dir)
    total_patients = len(patient_ids)
    logging.info(f"Total number of patients found: {total_patients}")

    if total_patients == 0:
        logging.error("No patients found. Please check the directories and file naming conventions.")
        return

    # Perform patient-wise 80-20 split
    train_patients, test_patients = train_test_split(
        patient_ids,
        test_size=0.2,
        random_state=42  # Ensures the same split every time
    )
    logging.info(f"Number of training patients: {len(train_patients)}")
    logging.info(f"Number of testing patients: {len(test_patients)}")

    # **Optional:** Save the testing patient list for future reference
    test_patients_csv = os.path.join(output_dir, 'test_patients.csv')
    pd.DataFrame({'Patient_ID': test_patients}).to_csv(test_patients_csv, index=False)
    logging.info(f"Testing patient list saved at {test_patients_csv}.")

    # Initialize models
    encoder = Encoder().to(device)
    generator = Generator().to(device)

    # Load model weights
    try:
        encoder_weight_path = os.path.join(sub_dir, 'encoder_weights.h5')
        generator_weight_path = os.path.join(sub_dir, 'generator_weights.h5')

        # Check if weight files exist
        if not os.path.exists(encoder_weight_path):
            logging.error(f"Encoder weight file not found at {encoder_weight_path}.")
            return
        if not os.path.exists(generator_weight_path):
            logging.error(f"Generator weight file not found at {generator_weight_path}.")
            return

        load_model_weights(encoder, encoder_weight_path)
        load_model_weights(generator, generator_weight_path)
        logging.info("Model weights loaded successfully.")
    except Exception as e:
        logging.error(f"Error loading model weights: {e}")
        return

    # Set models to evaluation mode
    encoder.eval()
    generator.eval()
    logging.info("Models set to evaluation mode.")

    # Process each patient in the testing set
    for patient_id in tqdm(test_patients, desc="Processing Testing Patients"):
        process_patient(
            patient_id,
            anatomical_dir,
            fat_fraction_dir,
            mask_dir,
            output_dir,
            encoder,
            generator,
            scaling_factor=100.0
        )

    logging.info("Processing completed for all testing patients.")

if __name__ == '__main__':
    main()
