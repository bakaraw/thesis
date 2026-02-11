# augmentation.py
from PIL import Image, ImageFilter
from io import BytesIO
import cv2
import numpy as np
import random

def apply_gaussian_blur(image, sigma_range=(0, 3)):
    """Apply Gaussian blur with random sigma"""
    sigma = random.uniform(sigma_range[0], sigma_range[1])
    if sigma > 0:
        image = image.filter(ImageFilter.GaussianBlur(radius=sigma))
    return image

def apply_jpeg_compression(image, quality_range=(30, 100), use_opencv=None):
    """Apply JPEG compression with random quality"""
    quality = random.randint(quality_range[0], quality_range[1])
    
    # Randomly choose between OpenCV and PIL
    if use_opencv is None:
        use_opencv = random.choice([True, False])
    
    if use_opencv:
        # Convert PIL to OpenCV format
        img_array = np.array(image)
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        _, encoded = cv2.imencode('.jpg', img_array, encode_param)
        decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        image = Image.fromarray(cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB))
    else:
        # Use PIL
        buffer = BytesIO()
        image.save(buffer, format='JPEG', quality=quality)
        buffer.seek(0)
        image = Image.open(buffer).convert('RGB')
    
    return image

def random_crop(image, crop_size=224):
    """Randomly crop image to specified size"""
    width, height = image.size
    
    if width < crop_size or height < crop_size:
        # Resize if image is smaller than crop size
        image = image.resize((max(width, crop_size), max(height, crop_size)), Image.LANCZOS)
        width, height = image.size
    
    left = random.randint(0, width - crop_size)
    top = random.randint(0, height - crop_size)
    right = left + crop_size
    bottom = top + crop_size
    
    return image.crop((left, top, right, bottom))

def center_crop(image, crop_size=224):
    """Center crop image to specified size"""
    width, height = image.size
    
    if width < crop_size or height < crop_size:
        image = image.resize((max(width, crop_size), max(height, crop_size)), Image.LANCZOS)
        width, height = image.size
    
    left = (width - crop_size) // 2
    top = (height - crop_size) // 2
    right = left + crop_size
    bottom = top + crop_size
    
    return image.crop((left, top, right, bottom))

def random_horizontal_flip(image, p=0.5):
    """Randomly flip image horizontally"""
    if random.random() < p:
        image = image.transpose(Image.FLIP_LEFT_RIGHT)
    return image

def apply_augmentation(image, mode='blur_jpeg_0.5', crop_size=224):
    """
    Apply augmentation based on mode
    
    Args:
        image: PIL Image
        mode: str - 'no_aug', 'blur', 'jpeg', 'blur_jpeg_0.5', 'blur_jpeg_0.1'
        crop_size: int - size to crop to (default 224)
    
    Returns:
        PIL Image with augmentations applied
    """
    
    # Always apply random horizontal flip
    image = random_horizontal_flip(image, p=0.5)
    
    if mode == 'no_aug':
        # No additional augmentation
        pass
    
    elif mode == 'blur':
        # Gaussian blur with 50% probability
        if random.random() < 0.5:
            image = apply_gaussian_blur(image, sigma_range=(0, 3))
    
    elif mode == 'jpeg':
        # JPEG compression with 50% probability
        if random.random() < 0.5:
            image = apply_jpeg_compression(image, quality_range=(30, 100))
    
    elif mode == 'blur_jpeg_0.5':
        # Blur with 50% probability
        if random.random() < 0.5:
            image = apply_gaussian_blur(image, sigma_range=(0, 3))
        # JPEG with 50% probability (independent)
        if random.random() < 0.5:
            image = apply_jpeg_compression(image, quality_range=(30, 100))
    
    elif mode == 'blur_jpeg_0.1':
        # Blur with 10% probability
        if random.random() < 0.1:
            image = apply_gaussian_blur(image, sigma_range=(0, 3))
        # JPEG with 10% probability (independent)
        if random.random() < 0.1:
            image = apply_jpeg_compression(image, quality_range=(30, 100))
    
    # Crop to target size
    image = random_crop(image, crop_size=crop_size)
    
    return image