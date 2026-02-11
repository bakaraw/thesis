import zipfile
import random
import os
from pathlib import Path

# --- CONFIGURATION ---
zip_source = r"G:\TrueFake.zip"  # Path to your ZIP on external drive
output_base_dir = r"C:\Users\bacal\Documents\Testing" # Path on your internal drive

# Target Constraints
TARGET_TOTAL = 2000
PLATFORMS = ["PreSocial"] # PreSocial is excluded based on your prompt
CLASSES = ["Real", "Fake"]
EXTENSIONS = (".jpeg", ".jpg", ".png") # Added png just in case

def create_dataset():
    # 1. Calculate quotas
    # 1200 total / 3 platforms = 400 per platform.
    # 400 / 2 classes = 200 images per class per platform.
    samples_per_bucket = int((TARGET_TOTAL / len(PLATFORMS)) / 2)
    
    print(f"Plan: Extracting {samples_per_bucket} images per class (Real/Fake) per platform.")
    print(f"Total operations: {samples_per_bucket * len(PLATFORMS) * 2} images.")
    print("-" * 40)

    # 2. Inventory the ZIP file
    # Structure: inventory['Facebook']['Real'] = [list of paths]
    inventory = {p: {'Real': [], 'Fake': []} for p in PLATFORMS}
    
    try:
        with zipfile.ZipFile(zip_source, 'r') as zf:
            all_files = zf.namelist()
            
            print("Scanning ZIP contents...")
            for file_path in all_files:
                # Skip directories and non-image files
                if file_path.endswith('/') or not file_path.lower().endswith(EXTENSIONS):
                    continue
                
                # Analyze path structure
                # Expected: TureFake / Platform / Class / ...
                parts = file_path.split('/')
                
                if len(parts) < 4: continue # Skip if path is too short
                
                platform = parts[1]
                category = parts[2] # Real or Fake
                
                # Only process if it matches our target platforms and classes
                if platform in PLATFORMS and category in CLASSES:
                    inventory[platform][category].append(file_path)

            # 3. Perform Random Sampling and Extraction
            print("Sampling and Extracting...")
            
            extracted_count = 0
            
            for platform in PLATFORMS:
                for category in CLASSES:
                    available_images = inventory[platform][category]
                    
                    # Check if we have enough images
                    if len(available_images) < samples_per_bucket:
                        print(f"WARNING: Not enough images in {platform}/{category}. Found {len(available_images)}, needed {samples_per_bucket}. Taking all.")
                        selected_files = available_images
                    else:
                        # Randomly select the specific amount
                        selected_files = random.sample(available_images, samples_per_bucket)
                    
                    # Extract them
                    for src_path in selected_files:
                        # Create a simplified destination path:
                        # Output / Platform / Class / original_filename
                        filename = os.path.basename(src_path)
                        dest_path = Path(output_base_dir) / platform / category / filename
                        
                        # Ensure directory exists
                        dest_path.parent.mkdir(parents=True, exist_ok=True)
                        
                        # Read bytes from zip and write to local file
                        with zf.open(src_path) as source_file, open(dest_path, "wb") as target_file:
                            target_file.write(source_file.read())
                        
                        extracted_count += 1
            
            print("-" * 40)
            print(f"Success! {extracted_count} images transferred to: {output_base_dir}")
            
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    create_dataset()