import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import utils
import h5py
import numpy as np
from multiprocessing import Pool, cpu_count
import warnings
import os
import pandas as pd
from PIL import Image
from io import BytesIO
import augmentation  # Import your augmentation module

warnings.filterwarnings('ignore', category=UserWarning)

import gc
gc.enable()

# TARGET SIZE FOR ALL IMAGES
TARGET_SIZE = (256, 256)  # Resize all images to 256x256 first
CROP_SIZE = 224  # Then crop to 224x224

# CHOOSE YOUR AUGMENTATION MODE HERE
AUGMENTATION_MODE = 'blur_jpeg_0.5'  # Options: 'no_aug', 'blur', 'jpeg', 'blur_jpeg_0.5', 'blur_jpeg_0.1'


def func(args):
    """Process a single parquet file and convert to HDF5"""
    doc, split = args  # split can be 'train', 'val', or 'test'
    
    # Only apply augmentation to training data
    is_training = (split == 'train')
    
    try:
        # Retrieve metadata from the parquet file
        (img_id, real_prompt, fake_prompt, img_url,
         image_gen0_meta, image_gen1_meta, image_gen2_meta, image_gen3_meta) = \
            utils.get_data(f'{split}/{doc:d}.parquet')

        # Load the parquet to get image bytes
        df = pd.read_parquet(f'{split}/{doc:d}.parquet')

        results = []

        # Convert image bytes to numpy arrays with optional augmentation
        def bytes_to_array(img_bytes, apply_aug=False):
            if img_bytes and len(img_bytes) > 0:
                # Load image
                img = Image.open(BytesIO(img_bytes)).convert('RGB')
                
                # Resize to target size first
                img = img.resize(TARGET_SIZE, Image.LANCZOS)
                
                # Apply augmentation for training, center crop for val/test
                if apply_aug:
                    # Apply training augmentations
                    img = augmentation.apply_augmentation(img, mode=AUGMENTATION_MODE, crop_size=CROP_SIZE)
                else:
                    # Just center crop for validation/test
                    img = augmentation.center_crop(img, crop_size=CROP_SIZE)
                
                # Convert to numpy array
                return np.array(img)
            return None

        # Process each row
        for i in range(len(img_id)):
            try:
                row = df.iloc[i]
                
                # Get real image
                real = None
                for col_name in ['image', 'real', 'real_image']:
                    if col_name in row and row[col_name] is not None:
                        real = bytes_to_array(row[col_name], apply_aug=is_training)
                        if real is not None:
                            break
                
                if real is None:
                    continue
                
                # Get generated images with same augmentation setting
                image_gen0 = bytes_to_array(row.get('image_gen0', b''), apply_aug=is_training)
                image_gen1 = bytes_to_array(row.get('image_gen1', b''), apply_aug=is_training)
                image_gen2 = bytes_to_array(row.get('image_gen2', b''), apply_aug=is_training)
                image_gen3 = bytes_to_array(row.get('image_gen3', b''), apply_aug=is_training)
                
                if all(img is not None for img in [real, image_gen0, image_gen1, image_gen2, image_gen3]):
                    results.append((
                        img_id[i],
                        real_prompt[i],
                        fake_prompt[i],
                        real,
                        image_gen0,
                        image_gen1,
                        image_gen2,
                        image_gen3
                    ))
                    
            except Exception as e:
                continue

        if not results:
            return

        # Unpack results
        img_id, original_prompt, positive_prompt, real, image_gen0, image_gen1, image_gen2, image_gen3 = zip(*results)

        # Convert to numpy arrays (now all images are CROP_SIZE x CROP_SIZE)
        real = np.array(real)
        image_gen0 = np.array(image_gen0)
        image_gen1 = np.array(image_gen1)
        image_gen2 = np.array(image_gen2)
        image_gen3 = np.array(image_gen3)

        # Encode strings to bytes
        original_prompt_bytes = [
            prompt.encode('utf-8') if isinstance(prompt, str) else prompt 
            for prompt in original_prompt
        ]
        positive_prompt_bytes = [
            prompt.encode('utf-8') if isinstance(prompt, str) else prompt 
            for prompt in positive_prompt
        ]

        # Save to HDF5 (different output folder based on split)
        output_folder = f'h5_{split}'
        os.makedirs(output_folder, exist_ok=True)
        
        c = {'compression': 'gzip', 'compression_opts': 1}

        with h5py.File(os.path.join(output_folder, f'{doc:04d}.h5'), 'w') as fw:
            fw.create_dataset('img_id', data=img_id, **c)
            fw.create_dataset('real', data=real, **c)
            fw.create_dataset('image_gen0', data=image_gen0, **c)
            fw.create_dataset('image_gen1', data=image_gen1, **c)
            fw.create_dataset('image_gen2', data=image_gen2, **c)
            fw.create_dataset('image_gen3', data=image_gen3, **c)
            fw.create_dataset('original_prompt', data=original_prompt_bytes, **c)
            fw.create_dataset('positive_prompt', data=positive_prompt_bytes, **c)

        aug_note = f" (aug: {AUGMENTATION_MODE})" if is_training else " (center crop)"
        print(f"✓ Saved h5_{split}/{doc:04d}.h5{aug_note}")
        
    except Exception as e:
        print(f"✗ Error processing {split}/{doc}: {e}")


if __name__ == '__main__':
    # Set the number of CPU workers
    cpu_workers = int(cpu_count() * 0.8)
    print(f'>> cpu_workers = {cpu_workers}')
    print(f'>> Target image size: {TARGET_SIZE} → Crop size: {CROP_SIZE}x{CROP_SIZE}')
    print(f'>> Augmentation mode: {AUGMENTATION_MODE}')
    print()

    # Process TRAIN data
    print("="*60)
    print("Processing TRAIN data")
    print("="*60)
    
    if os.path.exists('train'):
        train_files = [file[:-8] for file in os.listdir('train') if file.endswith('.parquet')]
        train_docs = sorted([int(doc) for doc in train_files])
        
        print(f'Found {len(train_docs)} train parquet files: {train_docs}')
        print(f'Augmentation: {AUGMENTATION_MODE} (flip + blur/jpeg + random crop)')
        
        # Create list of (doc, split) tuples for train
        train_args = [(doc, 'train') for doc in train_docs]
        
        with Pool(cpu_workers) as pool:
            pool.map(func, train_args)
        
        print("✓ Train processing complete\n")
    else:
        print("⚠ Warning: 'train' folder not found\n")

    # Process VAL data
    print("="*60)
    print("Processing VAL data")
    print("="*60)
    
    if os.path.exists('val'):
        val_files = [file[:-8] for file in os.listdir('val') if file.endswith('.parquet')]
        val_docs = sorted([int(doc) for doc in val_files])
        
        print(f'Found {len(val_docs)} val parquet files: {val_docs}')
        print('Augmentation: None (center crop only)')
        
        # Create list of (doc, split) tuples for val
        val_args = [(doc, 'val') for doc in val_docs]
        
        with Pool(cpu_workers) as pool:
            pool.map(func, val_args)
        
        print("✓ Val processing complete\n")
    else:
        print("⚠ Warning: 'val' folder not found\n")

    # Process TEST data
    print("="*60)
    print("Processing TEST data")
    print("="*60)
    
    if os.path.exists('test'):
        test_files = [file[:-8] for file in os.listdir('test') if file.endswith('.parquet')]
        test_docs = sorted([int(doc) for doc in test_files])
        
        print(f'Found {len(test_docs)} test parquet files: {test_docs}')
        print('Augmentation: None (center crop only)')
        
        # Create list of (doc, split) tuples for test
        test_args = [(doc, 'test') for doc in test_docs]
        
        with Pool(cpu_workers) as pool:
            pool.map(func, test_args)
        
        print("✓ Test processing complete\n")
    else:
        print("⚠ Warning: 'test' folder not found\n")

    print("\n" + "="*60)
    print("=== ALL PROCESSING COMPLETE ===")
    print("="*60)
    print(f"Images processed with augmentation: {AUGMENTATION_MODE}")
    print(f"Final image size: {CROP_SIZE}x{CROP_SIZE}")
    
    # Summary
    if os.path.exists('h5_train'):
        train_h5_count = len([f for f in os.listdir('h5_train') if f.endswith('.h5')])
        print(f"Train: {train_h5_count} HDF5 files in h5_train/")
    
    if os.path.exists('h5_val'):
        val_h5_count = len([f for f in os.listdir('h5_val') if f.endswith('.h5')])
        print(f"Val: {val_h5_count} HDF5 files in h5_val/")
    
    if os.path.exists('h5_test'):
        test_h5_count = len([f for f in os.listdir('h5_test') if f.endswith('.h5')])
        print(f"Test: {test_h5_count} HDF5 files in h5_test/")