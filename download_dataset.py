from datasets import load_dataset
import pandas as pd
import os
from io import BytesIO
import numpy as np
import requests
from PIL import Image

# Load dataset in streaming mode
dataset = load_dataset("elsaEU/ELSA_D3", split="train", streaming=True)

batch_size = 100  # Adjust based on your memory
train_batch_num = 0
val_batch_num = 0
test_batch_num = 0

# Create output folders
train_folder = "train"
val_folder = "val"
test_folder = "test"
os.makedirs(train_folder, exist_ok=True)
os.makedirs(val_folder, exist_ok=True)
os.makedirs(test_folder, exist_ok=True)

# SET YOUR LIMIT HERE
MAX_EXAMPLES = 700  # Download 700 examples total
TRAIN_SPLIT = 0.7   # 70% for training
VAL_SPLIT = 0.15    # 15% for validation
TEST_SPLIT = 0.15   # 15% for testing

TRAIN_SIZE = int(MAX_EXAMPLES * TRAIN_SPLIT)  # 490 for train
VAL_SIZE = int(MAX_EXAMPLES * VAL_SPLIT)       # 105 for val
TEST_SIZE = MAX_EXAMPLES - TRAIN_SIZE - VAL_SIZE  # 105 for test

print(f"Dataset split:")
print(f"  Train: {TRAIN_SIZE} examples")
print(f"  Val:   {VAL_SIZE} examples")
print(f"  Test:  {TEST_SIZE} examples")
print(f"  Total: {MAX_EXAMPLES} examples\n")

def download_image_from_url(url, timeout=10):
    """Download image from URL and return as bytes"""
    try:
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
        # Convert to PIL Image and back to bytes to ensure it's valid
        img = Image.open(BytesIO(response.content)).convert('RGB')
        buffer = BytesIO()
        img.save(buffer, format='PNG')
        return buffer.getvalue()
    except Exception as e:
        print(f"    Error downloading image from {url}: {e}")
        return None

def convert_and_map_columns(example, index):
    """Convert PIL Images to bytes and map to expected column names"""
    converted = {}
    
    # Map column names to what get_data.py expects
    converted['img_id'] = example.get('id', '')
    converted['original_prompt'] = example.get('original_prompt', '')
    converted['positive_prompt'] = example.get('positive_prompt', '')
    converted['img_url'] = example.get('url', '')
    
    # Create metadata as strings for each generated image
    for i in range(4):
        meta = {
            'model': example.get(f'model_gen{i}', ''),
            'width': example.get(f'width_gen{i}', 0),
            'height': example.get(f'height_gen{i}', 0),
            'num_inference_steps': example.get(f'num_inference_steps_gen{i}', 0),
            'filepath': example.get(f'filepath_gen{i}', '')
        }
        converted[f'image_gen{i}_meta'] = str(meta)
    
    # Download the REAL image from URL
    real_image_url = example.get('url', '')
    if real_image_url:
        print(f"  Example {index}: Downloading real image from URL...")
        real_image_bytes = download_image_from_url(real_image_url)
        if real_image_bytes:
            converted['image'] = real_image_bytes  # Store as 'image' column
        else:
            print(f"  Example {index}: Failed to download real image, skipping this example")
            return None
    else:
        print(f"  Example {index}: No URL for real image, skipping")
        return None
    
    # Convert generated PIL Images to bytes
    for key, value in example.items():
        if hasattr(value, 'save') and hasattr(value, 'mode'):
            buffer = BytesIO()
            value.save(buffer, format='PNG')
            converted[key] = buffer.getvalue()
        elif key not in converted and not key.startswith('model_') and \
             not key.startswith('width_') and not key.startswith('height_') and \
             not key.startswith('num_inference_steps_') and not key.startswith('filepath_'):
            # Keep other non-metadata columns
            converted[key] = value
    
    return converted

train_batch = []
val_batch = []
test_batch = []
total_processed = 0
train_count = 0
val_count = 0
test_count = 0
skipped = 0

for i, example in enumerate(dataset):
    # Stop after reaching the limit
    if total_processed >= MAX_EXAMPLES:
        print(f"\nReached limit of {MAX_EXAMPLES} examples. Stopping download.")
        break
    
    try:
        result = convert_and_map_columns(example, i)
        if result is not None:
            # Assign to train/val/test based on count
            if train_count < TRAIN_SIZE:
                train_batch.append(result)
                train_count += 1
                split_name = "TRAIN"
            elif val_count < VAL_SIZE:
                val_batch.append(result)
                val_count += 1
                split_name = "VAL"
            else:
                test_batch.append(result)
                test_count += 1
                split_name = "TEST"
            
            total_processed += 1
            
            # Show progress every 10 examples
            if total_processed % 10 == 0:
                print(f"Progress: {total_processed}/{MAX_EXAMPLES} (Train: {train_count}/{TRAIN_SIZE}, Val: {val_count}/{VAL_SIZE}, Test: {test_count}/{TEST_SIZE}, Skipped: {skipped})")
        else:
            skipped += 1
            
    except Exception as e:
        print(f"Error processing example {i}: {e}")
        skipped += 1
        continue
    
    # Save train batch when full
    if len(train_batch) >= batch_size:
        df = pd.DataFrame(train_batch)
        df.to_parquet(
            f"{train_folder}/{train_batch_num}.parquet",
            engine='pyarrow',
            compression='gzip'
        )
        print(f"✓ Saved {train_folder}/{train_batch_num}.parquet ({len(train_batch)} rows)")
        train_batch = []
        train_batch_num += 1
    
    # Save val batch when full
    if len(val_batch) >= batch_size:
        df = pd.DataFrame(val_batch)
        df.to_parquet(
            f"{val_folder}/{val_batch_num}.parquet",
            engine='pyarrow',
            compression='gzip'
        )
        print(f"✓ Saved {val_folder}/{val_batch_num}.parquet ({len(val_batch)} rows)")
        val_batch = []
        val_batch_num += 1
    
    # Save test batch when full
    if len(test_batch) >= batch_size:
        df = pd.DataFrame(test_batch)
        df.to_parquet(
            f"{test_folder}/{test_batch_num}.parquet",
            engine='pyarrow',
            compression='gzip'
        )
        print(f"✓ Saved {test_folder}/{test_batch_num}.parquet ({len(test_batch)} rows)")
        test_batch = []
        test_batch_num += 1

# Save remaining train data
if train_batch:
    df = pd.DataFrame(train_batch)
    df.to_parquet(
        f"{train_folder}/{train_batch_num}.parquet",
        engine='pyarrow',
        compression='gzip'
    )
    print(f"✓ Saved final train batch {train_folder}/{train_batch_num}.parquet ({len(train_batch)} rows)")

# Save remaining val data
if val_batch:
    df = pd.DataFrame(val_batch)
    df.to_parquet(
        f"{val_folder}/{val_batch_num}.parquet",
        engine='pyarrow',
        compression='gzip'
    )
    print(f"✓ Saved final val batch {val_folder}/{val_batch_num}.parquet ({len(val_batch)} rows)")

# Save remaining test data
if test_batch:
    df = pd.DataFrame(test_batch)
    df.to_parquet(
        f"{test_folder}/{test_batch_num}.parquet",
        engine='pyarrow',
        compression='gzip'
    )
    print(f"✓ Saved final test batch {test_folder}/{test_batch_num}.parquet ({len(test_batch)} rows)")

print(f"\n=== Download Complete ===")
print(f"Train: {train_batch_num + 1} batches, {train_count} examples")
print(f"Val:   {val_batch_num + 1} batches, {val_count} examples")
print(f"Test:  {test_batch_num + 1} batches, {test_count} examples")
print(f"Total examples processed: {total_processed}")
print(f"Total examples skipped: {skipped}")

# Verify splits
if os.path.exists(f"{train_folder}/0.parquet"):
    test_df = pd.read_parquet(f"{train_folder}/0.parquet")
    print("\n=== Train Set Verification ===")
    print(f"Columns: {test_df.columns.tolist()}")
    print(f"Rows in first batch: {len(test_df)}")
    print(f"Has 'image' column: {'image' in test_df.columns}")

if os.path.exists(f"{val_folder}/0.parquet"):
    test_df = pd.read_parquet(f"{val_folder}/0.parquet")
    print("\n=== Val Set Verification ===")
    print(f"Rows in first batch: {len(test_df)}")

if os.path.exists(f"{test_folder}/0.parquet"):
    test_df = pd.read_parquet(f"{test_folder}/0.parquet")
    print("\n=== Test Set Verification ===")
    print(f"Rows in first batch: {len(test_df)}")