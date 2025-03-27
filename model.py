import torch
import os
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
import imageio


def double_conv(in_channels, out_channels):
   return nn.Sequential(
       nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
       nn.BatchNorm3d(out_channels),  # Add BatchNorm3d after convolution
       nn.ReLU(inplace=True),
       nn.Dropout(0.1 if out_channels <= 32 else 0.2 if out_channels <= 128 else 0.3),
       nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
       nn.BatchNorm3d(out_channels),  # Add BatchNorm3d after second convolution
       nn.ReLU(inplace=True)
   )

def get_default_device():
   gpu_available = torch.cuda.is_available()
   return torch.device('cuda' if gpu_available else 'cpu'), gpu_available

class UNETR(nn.Module):
    def __init__(self, input_dim=3, output_dim=4,):
        super().__init__()

        # Contraction path
        self.conv1 = double_conv(in_channels=input_dim, out_channels=16) # [4, 16, 128, 128, 128]
        # self.tr1 = TransformerLayer(in_channels=16, image_size=image_size)
        self.pool1 = nn.MaxPool3d(kernel_size=2)

        self.conv2 = double_conv(in_channels=16, out_channels=32) # [1, 32, 64, 64, 64]
        # self.tr2 = TransformerLayer(in_channels=32, image_size=image_size//2)
        self.pool2 = nn.MaxPool3d(kernel_size=2)

        self.conv3 = double_conv(in_channels=32, out_channels=64)
        # self.tr3 = TransformerLayer(in_channels=64, image_size=image_size//4)
        self.pool3 = nn.MaxPool3d(kernel_size=2)

        self.conv4 = double_conv(in_channels=64, out_channels=128)
        # self.tr4 = TransformerLayer(in_channels=128, image_size=image_size//8)
        self.pool4 = nn.MaxPool3d(kernel_size=2)

        #BottleNeck
        self.conv5 = double_conv(in_channels=128, out_channels=256)

        #Decoder or Expansive path
        self.upconv6 = nn.ConvTranspose3d(in_channels=256, out_channels=128, kernel_size=2, stride=2)
        self.conv6 = double_conv(in_channels=256, out_channels=128)

        self.upconv7 = nn.ConvTranspose3d(in_channels=128, out_channels=64, kernel_size=2, stride=2)
        self.conv7 = double_conv(in_channels=128, out_channels=64)

        self.upconv8 = nn.ConvTranspose3d(in_channels=64, out_channels=32, kernel_size=2, stride=2)
        self.conv8 = double_conv(in_channels=64, out_channels=32)

        self.upconv9 = nn.ConvTranspose3d(in_channels=32, out_channels=16, kernel_size=2, stride=2)
        self.conv9 = double_conv(in_channels=32, out_channels=16)

        self.out_conv = nn.Conv3d(in_channels=16, out_channels=output_dim, kernel_size=1)

    def forward(self, x):
        # Contracting path
        c1 = self.conv1(x)
        # c1 = self.tr1(c1)
        # print("c1: ", c1.shape)
        p1 = self.pool1(c1)

        c2 = self.conv2(p1)
        # c2 = self.tr2(c2)
        # print("c2: ", c2.shape)
        p2 = self.pool2(c2)

        c3 = self.conv3(p2)
        # c3 = self.tr3(c3)
        # print("c3: ", c3.shape)
        p3 = self.pool3(c3)

        c4 = self.conv4(p3)
        # c4 = self.tr4(c4)
        # print("c4: ", c4.shape)
        p4 = self.pool4(c4)

        # Bottleneck
        c5 = self.conv5(p4)

        # Expansive path
        u6 = self.upconv6(c5)  # upscale
        u6 = torch.cat([u6, c4], dim=1)  # skip connections along channel dim
        c6 = self.conv6(u6)

        u7 = self.upconv7(c6)
        u7 = torch.cat([u7, c3], dim=1)
        c7 = self.conv7(u7)

        u8 = self.upconv8(c7)
        u8 = torch.cat([u8, c2], dim=1)
        c8 = self.conv8(u8)

        u9 = self.upconv9(c8)
        u9 = torch.cat([u9, c1], dim=1)
        c9 = self.conv9(u9)

        outputs = self.out_conv(c9)

        return outputs

@torch.inference_mode()
def inference(model, loader, device="cpu", num_batches_to_process=8):
    for idx, (batch_img, batch_mask) in enumerate(loader):
 
        # Move batch images to the device (CPU or GPU)
        batch_img = batch_img.to(device).float()
        # Get the predictions from the model
        pred_all = model(batch_img)
 
        # Move the predictions to CPU and apply argmax to get predicted classes
        pred_all = pred_all.cpu().argmax(dim=1).numpy()
        # Optionally break after processing a fixed number of batches
        if idx == num_batches_to_process:
            break
 
        # Visualize images and predictions
        for i in range(0, len(batch_img)):
            fig, ax = plt.subplots(1, 5, figsize=(20, 8))
            middle_slice = batch_img.shape[2] // 2  # Along Depth
            # Visualize different modalities (e.g., T1ce, FLAIR, T2)
            ax[0].imshow(batch_img[i, 0, middle_slice, :, :].cpu().numpy(), cmap="gray")
            ax[1].imshow(batch_img[i, 1, middle_slice, :, :].cpu().numpy(), cmap="gray")
            ax[2].imshow(batch_img[i, 2, middle_slice, :, :].cpu().numpy(), cmap="gray")
 
            # Get the ground truth mask as class indices using argmax (combine all classes)
            gt_combined = (
                batch_mask[i, :, middle_slice, :, :].argmax(dim=0).cpu().numpy()
            )
 
            # Visualize the ground truth mask
            ax[3].imshow(gt_combined, cmap="viridis")
            ax[3].set_title("Ground Truth (All Classes)")
            # Visualize the predicted mask
            ax[4].imshow(pred_all[i, middle_slice, :, :], cmap="viridis")
            ax[4].set_title("Predicted Mask")
 
            # Set titles for the image subplot
            ax[0].set_title("T1ce")
            ax[1].set_title("FLAIR")
            ax[2].set_title("T2")
 
            # Turn off axis for all subplots
            for a in ax:
                a.axis("off")
            # Show the plot
            plt.show()
 

# # Define a directory to save frames
# output_dir = "3d_vis_frames"
# os.makedirs(output_dir, exist_ok=True)

# # Define a colormap for multiple classes (e.g., 3 classes: background, tumor core, enhancing tumor)
# # You can modify these colors to better suit your needs
# CLASS_COLORS = {
#     0: (0, 0, 0),     # Class 0: Black (Background)
#     1: (1, 0, 0),     # Class 1: Red (Tumor Core)
#     2: (0, 1, 0),     # Class 2: Green (Edema)
#     3: (0, 0, 1),     # Class 3: Blue (Enhancing Tumor)
# }

# def overlay_mask_on_image(image, mask, alpha=0.5, class_colors=CLASS_COLORS):
#     """Overlay a multi-class mask on the grayscale image."""
#     # Normalize the image and convert it to RGB
#     img = (image - image.min()) / (image.max() - image.min())
#     img = np.stack([img, img, img], axis=-1)  # Convert grayscale to RGB

#     # Apply the class colors for the mask
#     color_mask = np.zeros_like(img)
#     for class_id, color in class_colors.items():
#         for i in range(3):  # Apply color per channel
#             color_mask[:, :, i] += color[i] * (mask == class_id)

#     # Blend the color mask with the original image
#     overlayed = img * (1 - alpha) + color_mask * alpha
#     return overlayed


# @torch.inference_mode()
# def create_3d_vis_video(model, loader, device="cpu", num_batches_to_process=8):
#     for idx, (batch_img, batch_mask) in enumerate(loader):
#         # Move batch images to the device (CPU or GPU)
#         batch_img = batch_img.to(device).float()

#         # Get the predictions from the model
#         pred_all = model(batch_img)

#         # Move the predictions to CPU and apply argmax to get predicted classes
#         pred_all = pred_all.cpu().argmax(dim=1).numpy()

#         # Optionally break after processing a fixed number of batches
#         if idx == num_batches_to_process:
#             break

#         # Iterate through each volume in the batch
#         for i in range(len(batch_img)):
#             frames = []  # To store each frame as an image
            
#             num_slices = batch_img.shape[2]  # Number of slices along the depth axis
            
#             for slice_idx in range(num_slices):  # Iterate through all slices in the 3D volume
#                 fig, ax = plt.subplots(1, 3, figsize=(15, 8))

#                 # Get the FLAIR slice (index 1)
#                 flair_slice = batch_img[i, 1, slice_idx, :, :].cpu().numpy()

#                 # Ground Truth Mask
#                 gt_mask = batch_mask[i, :, slice_idx, :, :].argmax(dim=0).cpu().numpy()

#                 # Predicted Mask
#                 pred_slice = pred_all[i, slice_idx, :, :]

#                 # Plot the FLAIR slice
#                 ax[0].imshow(flair_slice, cmap='gray')
#                 ax[0].set_title('FLAIR Slice')

#                 # Plot the FLAIR slice with ground truth mask overlaid using consistent color coding
#                 overlay_gt = overlay_mask_on_image(flair_slice, gt_mask, alpha=0.5, class_colors=CLASS_COLORS)
#                 ax[1].imshow(overlay_gt)
#                 ax[1].set_title('FLAIR with Ground Truth Overlay')

#                 # Plot the FLAIR slice with predicted mask overlaid using the same color coding
#                 overlay_pred = overlay_mask_on_image(flair_slice, pred_slice, alpha=0.5, class_colors=CLASS_COLORS)
#                 ax[2].imshow(overlay_pred)
#                 ax[2].set_title('FLAIR with Prediction Overlay')

#                 # Turn off axis for all subplots
#                 for a in ax:
#                     a.axis('off')

#                 # Save the frame as a temporary file
#                 frame_filename = os.path.join(output_dir, f"frame_{i}_{slice_idx}.png")
#                 plt.savefig(frame_filename)
#                 plt.close(fig)

#                 # Append the frame to the list
#                 frames.append(frame_filename)

#             # Create a video using the frames with half the speed (fps=5)
#             video_filename = f"3d_prediction_video_batch_{idx}_sample_{i}.mp4"
#             with imageio.get_writer(video_filename, mode='I', fps=5) as writer:
#                 for frame in frames:
#                     image = imageio.imread(frame)
#                     writer.append_data(image)

#             print(f"Saved 3D visualization video: {video_filename}")
    

if __name__ == '__main__':
    DEVICE, GPU_AVAILABLE = get_default_device()
    trained_model = UNETR(input_dim=3, output_dim=4)

    checkpoint = torch.load(
        "./3D_U-Net_BraTS_ckpt.tar", 
        map_location="cpu", 
    )['model']
    trained_model.load_state_dict(checkpoint, strict=False)

    image = np.load("data\images\image_447.npy")

    mask = np.load("data\mask\mask_447.npy")

    mask = torch.from_numpy(mask).permute(3, 2, 0, 1).argmax(dim=0)
    plt.imshow(image[128//2, :, :], cmap='viridis')
    plt.show()

    plt.imshow(mask[128//2, :, :], cmap='viridis')
    plt.show()

    image = torch.from_numpy(image).permute(3, 2, 0, 1).float().unsqueeze(0)
    pred = trained_model(image).argmax(dim=1).numpy()
    plt.imshow(pred[0, 128//2, :, :], cmap='viridis')
    plt.show()
    
    # loader = create_dataloader()
    # inference(trained_model, loader, device=DEVICE)