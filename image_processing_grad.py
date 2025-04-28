# -----------------------------
# FUTURE STEPS (IDEAS)
# -----------------------------
# - denoising could be optimized
#     --> parameters have just been chosen, could be better
# - resizing
#     --> do we want to smush everything into a square?
#     --> should we 'randomly' cut a square from the image and then resize this square?
# - order of operations?
#     --> perhaps resizing is better after segmentation?
#     --> i do think denoising and normalization should be done beforehand
# - better remove text from images
#     --> current marker removal kinda works ish
#         --> perhaps run twice to get rid of residuals?
# - see if there is a way to remove the background?
#     --> right now the background sometimes gets segmented as well
#         --> removing background could improve overall segmentation
# - values for SAM could be optimized (just randomly chosen rn based on vibes)

# -----------------------------
# LERA ANKLE PATIENTS WITH IMPLANTS/SCREWS
# 1019
# 1023
# 1028
# 1031
# 1062
# 1138
# -----------------------------

# -----------------------------
# FIRST/LAST PATIENT
# 1002 - 11
# -----------------------------

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
mask_generator = SamAutomaticMaskGenerator(model=sam, points_per_side=16, stability_score_thresh=0.75)

# -----------------------------
# Save images for each step (True)
# Should be false unless images are explicitly wanted
# -----------------------------
SAVE_INTERMEDIATE_STEPS = True

# -----------------------------
# STEP 0: MAKE BACKGROUND BLACK AND REMOVE MARKERS
# -----------------------------
def background_dark(img, threshold=127, sample_border=20):
    '''
    Ensures all images have a dark background by detecting the current background
    and inverting if the background is light.
    
    img: RGB image as numpy array
    threshold: brightness threshold to determine if background is light (0-255)
    sample_border: number of pixels from border to sample for background detection
    
    returns: RGB image with dark background
    '''
    # Convert to grayscale if image is in RGB
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray = img.copy()
    
    # Sample the border pixels to determine background color
    h, w = gray.shape
    
    # Create a border mask
    border_mask = np.zeros_like(gray, dtype=bool)
    border_mask[:sample_border, :] = True  # Top border
    border_mask[-sample_border:, :] = True  # Bottom border
    border_mask[:, :sample_border] = True  # Left border
    border_mask[:, -sample_border:] = True  # Right border
    
    # Calculate average brightness of border pixels
    border_pixels = gray[border_mask]
    avg_brightness = np.mean(border_pixels)
    
    # Determine if background is light based on threshold
    is_light_background = avg_brightness > threshold
    
    # If background is light, invert the image
    if is_light_background:
        if len(img.shape) == 3:
            # For RGB image, invert each channel
            return 255 - img
        else:
            # For grayscale image
            return 255 - gray
    else:
        # Background is already dark, return original
        return img

def remove_markers(img, bright_thresh=200, kernel_size=(5, 5)):
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

    # Decide thresholding based on background (should always be dark with dark function)
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
def denoise_image(img, d=15, sigmaColor=50, sigmaSpace=50):
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
def normalize(img_rgb):
    '''
    normalizes pixel intensity using min-max normalization

    img_rgb: image to be resized

    returns: normalized image
    '''
    normalized = (img_rgb - img_rgb.min()) / (img_rgb.max() - img_rgb.min())
    return np.clip(normalized, 0, 1)

def resize(img_rgb, target_shape=(640, 640)):
    ## CHANGE TO CROP
    ## PERHAPS CHANGE TO ADDING UNTIL SQUARE
    '''
    resizes images to target shape

    img_rgb: image to be resized
    target_shape: desired shape (size for YOLO model)

    returns: resized image
    '''
    resized = resize(img_rgb, target_shape, preserve_range=True, anti_aliasing=False)
    return resized

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

# -----------------------------
# STEP 3: SEGMENTATION (K-MEANS)
# -----------------------------
def segment_image_kmeans(img_rgb, n_clusters=3, max_iter=100, epsilon=1.0):
    '''
    Segments image using K-means clustering.
    img_rgb: RGB image (0-255 or 0-1 range)
    n_clusters: number of clusters (segments) to create
    max_iter: maximum iterations for k-means
    epsilon: termination criteria
    returns: segmented mask image (same size as input)
    '''

    # Convert to float32 and reshape for k-means
    if img_rgb.dtype != np.float32:
        # Check if image is in 0-1 range or 0-255 range
        if img_rgb.max() <= 1.0:
            data = img_rgb.astype(np.float32)
        else:
            data = img_rgb.astype(np.float32) / 255.0
    else:
        data = img_rgb.copy()
    
    # Reshape to 2D array of pixels (n_pixels, n_channels)
    h, w, c = data.shape
    reshaped_data = data.reshape((h * w, c))
    
    # Define criteria and apply K-means
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, max_iter, epsilon)
    _, labels, centers = cv2.kmeans(
        reshaped_data, 
        n_clusters, 
        None, 
        criteria, 
        10, 
        cv2.KMEANS_RANDOM_CENTERS
    )
    
    # Convert back to uint8 image
    centers = centers.astype(np.uint8) * 255 if data.max() <= 1.0 else centers.astype(np.uint8)
    segmented_data = centers[labels.flatten()]
    segmented_image = segmented_data.reshape((h, w, c))
    
    # Create a mask image (each cluster gets a different color)
    mask = np.zeros((h, w, 3), dtype=np.uint8)
    labels_2d = labels.reshape(h, w)
    
    # Assign different colors to different segments
    for i in range(n_clusters):
        mask[labels_2d == i] = np.random.randint(0, 255, size=3, dtype=np.uint8)
    
    return mask

def determine_clusters_by_edge_information(img_rgb, min_clusters=2, max_clusters=8):
    '''
    Determines optimal number of clusters based on edge information in the image.
    This is particularly useful for ankle X-rays where different structures 
    (bones, soft tissue, background) have distinct boundaries.
    
    img_rgb: RGB image as numpy array
    min_clusters: minimum number of clusters to consider
    max_clusters: maximum number of clusters to consider
    
    returns: recommended number of clusters
    '''
    # Convert to grayscale if needed
    if img_rgb.dtype == np.float64:
        img_rgb = (img_rgb * 255).astype(np.uint8)
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_rgb.copy()
    
    # Compute gradient magnitude using Sobel operators
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
    
    # Normalize gradient magnitude to 0-255
    gradient_magnitude = cv2.normalize(gradient_magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # Apply threshold to get significant edges
    _, binary_edges = cv2.threshold(gradient_magnitude, 50, 255, cv2.THRESH_BINARY)
    
    # Find connected components (distinct regions)
    num_labels, labels = cv2.connectedComponents(binary_edges)
    
    # Count significant components (filtering out small noise)
    min_component_size = (gray.shape[0] * gray.shape[1]) // 500  # Adjust divisor as needed
    significant_components = 0
    
    for i in range(1, num_labels):  # Skip background (0)
        if np.sum(labels == i) > min_component_size:
            significant_components += 1
    
    # Use edge density to weight the number of clusters
    edge_density = np.count_nonzero(binary_edges) / binary_edges.size
    
    # Calculate recommended clusters
    # More edges typically indicates more structures
    recommended_clusters = min(max(min_clusters, 
                                  int(min_clusters + (max_clusters - min_clusters) * edge_density * 2)), 
                             max_clusters)
    
    # Further adjust based on significant edge components
    if significant_components > 0:
        component_factor = min(significant_components / 10, 1.0)  # Scale factor
        recommended_clusters = max(recommended_clusters, 
                                 min(int(recommended_clusters + component_factor * 2), max_clusters))
    
    return recommended_clusters

def segment_image_kmeans_with_edge_detection(img_rgb, max_iter=100, epsilon=1.0):
    # Determine optimal number of clusters using edge information
    n_clusters = determine_clusters_by_edge_information(img_rgb)
    print(f"Edge analysis recommends {n_clusters} clusters for this image")
    
    # Then proceed with k-means segmentation using this number
    return segment_image_kmeans(img_rgb, n_clusters=n_clusters, max_iter=max_iter, epsilon=epsilon)

# -----------------------------
# IMAGE PROCESSING PIPELINE
# -----------------------------
def process_image(image_path, image_file, save_folder, segmentation_method='yolo', skip_preprocessing=False, n_clusters=3):
    '''
    processes image (step 1-3)
    img_rgb: image to be processed
    image_file: name of the image file, used for saving masks
    save_folder: directory where the segmented image/mask is saved
    segmentation_method: 'yolo', 'sam', or 'kmeans'
    skip_preprocessing: if True, skip denoising and normalization steps
    kmeans_clusters: number of clusters for k-means segmentation
    returns: processed image --> not doing this atm
    '''
    img = cv2.imread(image_path)
    # ensure image is loaded correctly
    if img is None:
        print(f"[ERROR] Could not read {image_path}")
        return
    
    # convert to rgb
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Ensure dark background
    img_rgb = background_dark(img_rgb)
    
    # ensure save folder exists
    os.makedirs(save_folder, exist_ok=True)
    
    if skip_preprocessing:
        img_normalized = img_rgb
    else:
        # remove markers (on original image)
        img_cleaned = remove_markers(img_rgb)
        if SAVE_INTERMEDIATE_STEPS:
            cv2.imwrite(os.path.join(save_folder, f"{image_file}_step0_markers_removed.png"), 
                        cv2.cvtColor(img_cleaned, cv2.COLOR_RGB2BGR))
        
        # denoise (on cleaned image)
        img_denoised = denoise_image(img_cleaned)
        if SAVE_INTERMEDIATE_STEPS:
            cv2.imwrite(os.path.join(save_folder, f"{image_file}_step1_denoised.png"), 
                        cv2.cvtColor(img_denoised, cv2.COLOR_RGB2BGR))
        
        # normalize (on denoised image) (for now no resizing)
        img_normalized = normalize(img_denoised)
        if SAVE_INTERMEDIATE_STEPS:
            normalized_uint8 = (img_normalized * 255).astype(np.uint8)
            cv2.imwrite(os.path.join(save_folder, f"{image_file}_step2_normalized.png"), 
                        cv2.cvtColor(normalized_uint8, cv2.COLOR_RGB2BGR))
    
    # Choose segmentation method
    if segmentation_method == 'sam':
        # SAM expects uint8 RGB image in [0, 255] range
        if img_normalized.dtype != np.uint8:
            img_for_segmentation = (img_normalized * 255).astype(np.uint8)
        else:
            img_for_segmentation = img_normalized
        segmented_mask = segment_image_sam(img_for_segmentation)
    elif segmentation_method == 'kmeans':
        segmented_mask = segment_image_kmeans_with_edge_detection(img_normalized)
    else:  # default to YOLO
        segmented_mask = segment_image_yolo(img_normalized)
    
    if SAVE_INTERMEDIATE_STEPS and segmented_mask is not None:
        mask_save = segmented_mask if segmented_mask.dtype == np.uint8 else (segmented_mask * 255).astype(np.uint8)
        cv2.imwrite(os.path.join(save_folder, f"{image_file}_step3_{segmentation_method}_segmentation.png"), mask_save)

def process_patient(patient_id, base_path, save_base_path, segmentation_method='yolo', skip_preprocessing=False, n_clusters=3):
    '''
    process each image for patient
    patient_id: patient identifier
    base_path: path to patient folders
    save_base_path: path where images are to be saved
    segmentation_method: 'yolo', 'sam', or 'kmeans'
    skip_preprocessing: if True, skip denoising and normalization steps
    kmeans_clusters: number of clusters for k-means segmentation (if used)
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
        process_image(image_path, image_file, save_folder, 
                     segmentation_method=segmentation_method, 
                     skip_preprocessing=skip_preprocessing,
                     n_clusters=n_clusters)

# -----------------------------
# DRIVER SCRIPT
# -----------------------------
def main():
    df_labels = pd.read_csv(labels_path, header=None, names=['patient_id', 'type', 'some_flag'])
    df_ankle = df_labels[df_labels['type'] == 'XR ANKLE']
    
    for _, row in df_ankle.iterrows():
        # Change 'sam' to 'kmeans' to use k-means segmentation instead
        process_patient(row['patient_id'], base_dir, save_base_dir, 
                       segmentation_method='kmeans',  # or 'yolo' or 'sam'
                       skip_preprocessing=False)
        print(f'Images processed for patient {row["patient_id"]}')
    
    print("All images processed.")

if __name__ == '__main__':
    main()
