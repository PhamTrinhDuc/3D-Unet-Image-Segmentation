import streamlit as st
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from model import UNETR

MODEL_PATH = "./3D_U-Net_BraTS_ckpt.tar"

def get_default_device():
   gpu_available = torch.cuda.is_available()
   return torch.device('cuda' if gpu_available else 'cpu'), gpu_available

def safe_load_numpy_image(uploaded_file):
    transformer = transforms.Normalize(mean=[0.5], std=[0.5])
    try:
        origin_image = np.load(uploaded_file)
        transform_image = torch.from_numpy(origin_image).permute(3, 2, 0, 1).float().unsqueeze(0)
        transform_image = transformer(transform_image)
        return transform_image
    
    except Exception as e:
        st.error(f"Error loading NumPy image: {e}")
        return None

def simple_inference(image_tensor, device="cpu"):
    """
    Simple inference with a basic 3D convolution model
    """
    try:
        DEVICE, GPU_AVAILABLE = get_default_device()
        trained_model = UNETR(input_dim=3, output_dim=4).to(DEVICE)

        checkpoint = torch.load(
            MODEL_PATH, 
            map_location="cpu", 
        )['model']
        trained_model.load_state_dict(checkpoint, strict=False)
        image_tensor = image_tensor.to(DEVICE).float()

        with torch.no_grad():
            pred_all = trained_model(image_tensor)
        # Process predictions
        pred_all = pred_all.cpu().argmax(dim=1).numpy()
        return pred_all
    except Exception as e:
        st.error(f"Error during model inference: {e}")
        return None

def visualize_results(original_image, prediction, middle_slice):
    """
    Create visualization of original image and prediction
    """
    try:
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        original_slice = original_image[0, 0, middle_slice, :, :]
        pred_slice = prediction[0, middle_slice, :, :]
        # Original image
        ax[0].imshow(original_slice, cmap='gray')
        ax[0].set_title('Original Image (Middle Slice)')
        ax[0].axis('off')
        # Prediction
        ax[1].imshow(pred_slice, cmap='viridis')
        ax[1].set_title('Predicted Segmentation')
        ax[1].axis('off')
        
        plt.tight_layout()
        return fig
    except Exception as e:
        st.error(f"Error creating visualization: {e}")
        return None


def main():
    st.title('🏥 3D Medical Image Segmentation')
    st.sidebar.header('Model Configuration')
    available_devices = ['CPU']
    if torch.cuda.is_available():
        available_devices.append('CUDA')
    device = st.sidebar.selectbox('Inference Device', available_devices)
    
    uploaded_file = st.sidebar.file_uploader(
        "Upload 3D Medical Image (NumPy .npy format)", 
        type=['npy']
    )
    
    if uploaded_file is not None:
        st.write('### Image Processing')
        
        with st.spinner('Loading and preprocessing image...'):
            transform_image = safe_load_numpy_image(uploaded_file)
            
            if transform_image is not None:
                st.write(f"Image Shape: {transform_image.shape}")
                st.write(f"Image Data Type: {transform_image.dtype}")
                st.write(f"Min Value: {transform_image.min()}")
                st.write(f"Max Value: {transform_image.max()}")
                
                if transform_image is not None:
                    device_selected = 'cuda' if device == 'CUDA' and torch.cuda.is_available() else 'cpu'
                    
                    with st.spinner('Running model inference...'):
                        prediction = simple_inference(transform_image, device=device_selected)
                    
                    if prediction is not None:
                        st.write('### Segmentation Results')
                        middle_slice = transform_image.shape[2] // 2
                        fig = visualize_results(transform_image, prediction, middle_slice)
                        st.pyplot(fig)

if __name__ == '__main__':
    main()