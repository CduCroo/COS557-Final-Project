# -----------------------------
# IMPORTS
# -----------------------------
import os
import cv2
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage.transform import resize
from ultralytics import YOLO
# from segment_anything import sam_model_registry, SamPredictor
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from segment_anything.utils.transforms import ResizeLongestSide

# -----------------------------
# CONFIGURATION
# -----------------------------
labels_path = '/Users/anneducroo/Library/CloudStorage/Dropbox/Mac (2)/Documents/PRINCETON/SPRING 2025/COS557/Stanford LERA/leralowerextremityradiographs-6/LERA Dataset/labels.csv'
base_dir = '/Users/anneducroo/Library/CloudStorage/Dropbox/Mac (2)/Documents/PRINCETON/SPRING 2025/COS557/Stanford LERA/leralowerextremityradiographs-6/LERA Dataset'
save_base_dir = '/Users/anneducroo/Library/CloudStorage/Dropbox/Mac (2)/Documents/PRINCETON/SPRING 2025/COS557/Processed Images LERA'

# load YOLOv11 segmentation model
model_yolo = YOLO('yolo11n-seg.pt')

# load SAM model
# use base model (smallest)
sam_checkpoint = "/Users/anneducroo/Library/CloudStorage/Dropbox/Mac (2)/Documents/PRINCETON/SPRING 2025/COS557/sam_vit_b_01ec64.pth"
model_type = "vit_b"
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
# sam_predictor = SamPredictor(sam)
mask_generator = SamAutomaticMaskGenerator(model=sam, points_per_side=16, stability_score_thresh=0.90)

# -----------------------------
# STEP 0: REMOVE MARKERS
# -----------------------------
def remove_markers(img, dark_thresh=50, bright_thresh=200, kernel_size=(5, 5)):
    '''
    Removes bright or dark markers depending on image background.

    img: RGB image (as NumPy array)
    dark_thresh: Threshold for dark markers (on white background)
    bright_thresh: Threshold for bright markers (on dark background)
    kernel_size: Morphological kernel size

    returns: cleaned image with markers removed
    '''
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Estimate background brightness using corner pixels
    h, w = gray.shape
    corner_vals = [
        gray[0, 0], gray[0, w - 1],
        gray[h - 1, 0], gray[h - 1, w - 1]
    ]
    avg_corner_brightness = np.mean(corner_vals)

    # Decide thresholding based on background
    if avg_corner_brightness > 127:
        # White background → remove dark markers
        _, mask = cv2.threshold(gray, dark_thresh, 255, cv2.THRESH_BINARY_INV)
    else:
        # Black background → remove bright markers
        _, mask = cv2.threshold(gray, bright_thresh, 255, cv2.THRESH_BINARY)

    # Morphological operations to clean mask
    kernel = np.ones(kernel_size, np.uint8)
    dilated = cv2.dilate(mask, kernel, iterations=2)

    # Inpaint detected regions
    img_cleaned = cv2.inpaint(img, dilated, 3, cv2.INPAINT_TELEA)

    return img_cleaned

# -----------------------------
# STEP 1: DENOISING
# -----------------------------
def denoise_image(img, d=5, sigmaColor=25, sigmaSpace=25):
    '''
    Applies bilateral filtering to reduce noise while preserving edges.
    
    img: RBG image
    d: diameter of the pixel neighborhood used for filtering
    sigmaColor: filter sigma in the color space --> higher = more color smoothing
    sigmaSpace: filter sigma in the coordinate space --> higher = more spatial smoothing

    returns: denoised image (RGB)

    --> technically returns image in same format (RGB or BGR) as input image
    '''
    filtered = cv2.bilateralFilter(img, d=d, sigmaColor=sigmaColor, sigmaSpace=sigmaSpace)
    return filtered

# -----------------------------
# STEP 2: NORMALIZATION
# -----------------------------
def normalize_and_resize(img_rgb, target_shape=(640, 640)):
    '''
    resizes images to target shape, then normalized pixel intensity using min-max normalization

    img_rgb: image to be resized
    target_shape: desired shape (size for YOLO model)

    returns: resized image
    '''
    resized = resize(img_rgb, target_shape, preserve_range=True, anti_aliasing=False)
    normalized = (resized - resized.min()) / (resized.max() - resized.min())
    return normalized

# -----------------------------
# STEP 3: SEGMENTATION (YOLO)
# -----------------------------
def segment_image_yolo(img_rgb):
    '''
    segments image using pretrained YOLOv8 segmentation model

    img_rgb: image to be segmented

    returns: segmented mask image (same size as input)
    '''
    
    # run YOLO segmentation on the image
    results = model_yolo(img_rgb)  # model assumed to be preloaded globally
    
    for result in results:
        segmented_mask = None  # initialize placeholder

        # CASE 1: No segmentation masks (fallback to bounding boxes or empty)
        if result.masks is None:

            # Bounding boxes were detected
            if len(result.boxes) > 0:
                # Create a blank mask of the same size as input image
                segmented_mask = np.zeros_like(img_rgb)

                # Draw white filled rectangles for each bounding box
                for box in result.boxes.xywh:  # format: center_x, center_y, width, height
                    x, y, w, h = box.tolist()
                    top_left = (int(x - w / 2), int(y - h / 2))
                    bottom_right = (int(x + w / 2), int(y + h / 2))
                    cv2.rectangle(segmented_mask, top_left, bottom_right, (255, 255, 255), -1)
            
            # No objects detected — return empty mask
            else:
                segmented_mask = np.zeros_like(img_rgb)

        # CASE 2: Segmentation masks available
        else:
            # Convert torch tensor masks to NumPy array: (num_masks, H, W)
            masks = result.masks.data.cpu().numpy()

            # Merge all masks into a single binary mask (logical OR across all masks)
            combined_mask = np.any(masks, axis=0).astype(np.uint8)  # (H, W)

            # Convert to 3-channel RGB mask for visualization and saving
            segmented_mask = np.stack([combined_mask]*3, axis=-1) * 255  # (H, W, 3)

        # Return the final mask
        return segmented_mask

# -----------------------------
# STEP 3: SEGMENTATION (SAM)
# -----------------------------
def segment_image_sam(img_rgb):
    '''
    Segments image using Segment Anything Model (SAM) automatic mask generator.

    img_rgb: RGB image

    returns: segmented mask (same size as input)
    '''
    masks = mask_generator.generate(img_rgb)

    # Create an empty RGB mask
    overlay = np.zeros_like(img_rgb)

    np.random.seed(42)
    for i, mask_dict in enumerate(masks):
        mask = mask_dict['segmentation']
        color = np.random.randint(0, 255, size=(3,), dtype=np.uint8)
        overlay[mask] = color

    return overlay

'''
def segment_image_sam(img_rgb):
'''
'''
    Segments image using Segment Anything Model (SAM) with automatic point prompts.

    img_rgb: RGB image

    returns: segmented mask (same size as input)
    '''
'''

    # Resize image for SAM input
    transformer = ResizeLongestSide(sam.image_encoder.img_size)
    img_sam = transformer.apply_image(img_rgb)
    img_tensor = sam_predictor.transform.apply_image(img_sam)
    input_image = torch.as_tensor(img_tensor).permute(2, 0, 1).unsqueeze(0)

    # Prepare predictor
    sam_predictor.set_image(img_rgb)

    # Sample 1–3 points around the center as positive prompts
    h, w, _ = img_rgb.shape
    input_points = np.array([[w // 2, h // 2]])  # Midpoint prompt
    input_labels = np.array([1])  # Positive point

    '''
'''
    # binary mask (maybe not wanted)
    masks, _, _ = sam_predictor.predict(
        point_coords=input_points,
        point_labels=input_labels,
        multimask_output=False,
    )

    mask = masks[0]  # Binary mask (H, W)
    segmented_mask = np.stack([mask]*3, axis=-1).astype(np.uint8) * 255
    '''

'''
    # multi-class masks
    segmented_masks, scores, logits = sam_predictor.predict(
        point_coords=input_points,
        point_labels=input_labels,
        multimask_output=True,  # <--- get more than one mask
    )

    return segmented_masks
    '''
'''
'''

# -----------------------------
# IMAGE PROCESSING PIPELINE
# -----------------------------
def process_image(image_path, image_file, save_folder, use_sam=False, skip_preprocessing=False):
    '''
    processes image (step 1-3)

    img_rgb: image to be processed
    image_file: name of the image file, used for saving masks
    save_folder: directory where the segmented image/mask is saved
    use_sam: if True, use SAM instead of YOLO
    skip_preprocessing: if True, skip denoising and normalization steps

    returns: processed image --> not doing this atm
    '''
    img = cv2.imread(image_path)

    # ensure image is loaded correctly
    if img is None:
        print(f"[ERROR] Could not read {image_path}")
        return

    # convert to rgb
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # ensure save folder exists
    os.makedirs(save_folder, exist_ok=True)

    if skip_preprocessing:
        img_normalized = img_rgb
    else:
        # remove markers (on original image)
        img_cleaned = remove_markers(img_rgb)

        # denoise (on cleaned image)
        img_denoised = denoise_image(img_cleaned)

        # normalize (on denoised image)
        img_normalized = normalize_and_resize(img_denoised)

        # save normalized image
        normalized_uint8 = (img_normalized * 255).astype(np.uint8)
        norm_path = os.path.join(save_folder, f"{image_file}_normalized.png")
        cv2.imwrite(norm_path, cv2.cvtColor(normalized_uint8, cv2.COLOR_RGB2BGR))

    # segment image
    # choose segmentation model
    if use_sam:
        '''
        # this was for SamPredictor
        masks = segment_image_sam(img_normalized)
        
        # Color-overlay each mask (choose random colors)
        overlay = np.zeros_like(img_normalized)
        np.random.seed(42)
        colors = np.random.randint(0, 255, size=(len(masks), 3), dtype=np.uint8)

        for i, mask in enumerate(masks):
            for c in range(3):
                overlay[..., c] += (mask.astype(np.uint8) * colors[i][c])

        segmented_mask = np.clip(overlay, 0, 255).astype(np.uint8)

        # Save overlaid mask image
        segmentation_path = os.path.join(save_folder, f"{image_file}_segmentation_mask.png")
        cv2.imwrite(segmentation_path, cv2.cvtColor(segmented_mask, cv2.COLOR_RGB2BGR))
        '''
        segmented_mask = segment_image_sam(img_normalized)
    else:
        segmented_mask = segment_image_yolo(img_normalized)

        # save segmented image (ensure it's in uint8 format before saving)
        if segmented_mask is not None:
            segmented_mask_uint8 = (segmented_mask * 255).astype(np.uint8) if segmented_mask.dtype != np.uint8 else segmented_mask
            segmentation_path = os.path.join(save_folder, f"{image_file}_segmentation_mask.png")
            cv2.imwrite(segmentation_path, segmented_mask_uint8)

    # save comparison original vs. fully processed image
    if segmented_mask is not None:
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        axes[0].imshow(img_rgb)
        axes[0].set_title(f'Original ({image_file})')
        axes[0].axis('off')

        # axes[1].imshow(segmented_mask, cmap='gray')
        axes[1].imshow(segmented_mask)
        axes[1].set_title(f'Mask ({image_file})')
        axes[1].axis('off')

        plot_path = os.path.join(save_folder, f'{image_file}_comparison.png')
        plt.savefig(plot_path)
        plt.close()

# -----------------------------
# PATIENT-LEVEL HANDLER
# -----------------------------
def process_patient(patient_id, base_path, save_base_path, use_sam=False, skip_preprocessing=False):
    '''
    process each image for patient

    patient_id: patient identifier
    base_path: path to patient folders
    save_base_path: path where images are to be saved
    use_sam: if True, use SAM instead of YOLO
    skip_preprocessing: if True, skip denoising and normalization steps
    '''
    patient_folder = os.path.join(base_path, str(patient_id), 'ST-1')
    
    # ensure patient found
    if not os.path.isdir(patient_folder):
        print(f"[SKIP] Folder not found: {patient_folder}")
        return

    save_folder = os.path.join(save_base_path, str(patient_id))
    image_files = [f for f in os.listdir(patient_folder) if f.endswith('.png')]

    for image_file in image_files:
        image_path = os.path.join(patient_folder, image_file)
        process_image(image_path, image_file, save_folder, use_sam=use_sam, skip_preprocessing=skip_preprocessing)

# -----------------------------
# DRIVER SCRIPT
# -----------------------------
def main():
    df_labels = pd.read_csv(labels_path, header=None, names=['patient_id', 'type', 'some_flag'])
    df_ankle = df_labels[df_labels['type'] == 'XR ANKLE']

    for _, row in df_ankle.iterrows():
        # right now, using sam on just original image (no denoising/normalization)
        process_patient(row['patient_id'], base_dir, save_base_dir, use_sam=True, skip_preprocessing=True)
        print(f'images processed for patient {row['patient_id']}')

    print("All images processed.")

if __name__ == '__main__':
    main()
