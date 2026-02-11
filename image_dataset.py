import torch
from torch.utils.data import Dataset
from PIL import Image
import os
import clip
import numpy as np
from transformers import BlipProcessor, BlipForConditionalGeneration

class ImageFolderDatasetWithBLIP(Dataset):
    """
    Dataset for loading images from folder structure with BLIP auto-captioning:
    TrueFake/
        non-shared/
            true/
                00001.jpg
                ...
            fake/
                00001.jpg
                ...
    """
    def __init__(self, root_dir, use_blip=True, cache_captions=True):
        """
        Args:
            root_dir: Path to TrueFake folder
            use_blip: If True, generate captions using BLIP
            cache_captions: If True, cache generated captions to avoid regenerating
        """
        self.root_dir = root_dir
        self.use_blip = use_blip
        self.cache_captions = cache_captions
        
        # Paths
        self.true_dir = os.path.join(root_dir, 'non-shared', 'real')
        self.fake_dir = os.path.join(root_dir, 'non-shared', 'fake')
        
        # Setup device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load CLIP model
        print("Loading CLIP model...")
        self.clip_model, self.clip_preprocess = clip.load("ViT-L/14", device=self.device)
        self.clip_model.eval()
        
        # Load BLIP model if needed
        if self.use_blip:
            print("Loading BLIP model for caption generation...")
            self.blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
            self.blip_model = BlipForConditionalGeneration.from_pretrained(
                "Salesforce/blip-image-captioning-large"
            ).to(self.device)
            self.blip_model.eval()
        
        # Collect all image paths and labels
        self.samples = []
        self.labels = []
        self.filenames = []
        
        # Cache for captions
        self.caption_cache = {}
        
        # Load true images (label 0)
        if os.path.exists(self.true_dir):
            true_images = [f for f in os.listdir(self.true_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            for img_name in sorted(true_images):
                self.samples.append(os.path.join(self.true_dir, img_name))
                self.labels.append(0)  # 0 = real/true
                self.filenames.append(img_name)
        
        # Load fake images (label 1)
        if os.path.exists(self.fake_dir):
            fake_images = [f for f in os.listdir(self.fake_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            for img_name in sorted(fake_images):
                self.samples.append(os.path.join(self.fake_dir, img_name))
                self.labels.append(1)  # 1 = fake
                self.filenames.append(img_name)
        
        print(f"\nLoaded {len(self.samples)} images:")
        print(f"  Real: {self.labels.count(0)}")
        print(f"  Fake: {self.labels.count(1)}")
        
        # Pre-generate all captions if using BLIP and caching
        if self.use_blip and self.cache_captions:
            print("\nGenerating captions for all images...")
            self._pregenerate_captions()
    
    def _generate_caption(self, image):
        """Generate caption for an image using BLIP"""
        if not self.use_blip:
            return "a photo"
        
        # Process image
        inputs = self.blip_processor(image, return_tensors="pt").to(self.device)
        
        # Generate caption
        with torch.no_grad():
            out = self.blip_model.generate(**inputs, max_length=50)
        
        caption = self.blip_processor.decode(out[0], skip_special_tokens=True)
        return caption
    
    def _pregenerate_captions(self):
        """Pre-generate captions for all images to speed up training"""
        from tqdm import tqdm
        
        for idx in tqdm(range(len(self.samples)), desc="Generating captions"):
            img_path = self.samples[idx]
            filename = self.filenames[idx]
            
            # Load image
            image = Image.open(img_path).convert('RGB')
            
            # Generate caption
            caption = self._generate_caption(image)
            
            # Cache it
            self.caption_cache[filename] = caption
        
        print(f"✓ Generated and cached {len(self.caption_cache)} captions")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path = self.samples[idx]
        label = self.labels[idx]
        filename = self.filenames[idx]
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        # Get caption (from cache or generate)
        if filename in self.caption_cache:
            caption = self.caption_cache[filename]
        else:
            caption = self._generate_caption(image)
            if self.cache_captions:
                self.caption_cache[filename] = caption
        
        # Extract CLIP features
        with torch.no_grad():
            # Image features
            image_input = self.clip_preprocess(image).unsqueeze(0).to(self.device)
            image_features = self.clip_model.encode_image(image_input)
            image_features = image_features.cpu().squeeze()
            
            # Text features from generated caption
            text_input = clip.tokenize([caption], truncate=True).to(self.device)
            text_features = self.clip_model.encode_text(text_input)
            text_features = text_features.cpu().squeeze()
        
        return image_features, text_features, torch.tensor(label, dtype=torch.float32)


class ImageFolderDataset(Dataset):
    """
    Simple dataset without BLIP (uses default caption)
    """
    def __init__(self, root_dir, transform=None, extract_clip_features=True):
        self.root_dir = root_dir
        self.transform = transform
        self.extract_clip_features = extract_clip_features
        
        # Paths
        self.true_dir = os.path.join(root_dir, 'non-shared', 'real')
        self.fake_dir = os.path.join(root_dir, 'non-shared', 'fake')
        
        # Load CLIP if needed
        if self.extract_clip_features:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.clip_model, self.clip_preprocess = clip.load("ViT-L/14", device=self.device)
            self.clip_model.eval()
        
        # Collect all image paths and labels
        self.samples = []
        self.labels = []
        
        # Load real images (label 0)
        if os.path.exists(self.true_dir):
            true_images = [f for f in os.listdir(self.true_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            for img_name in sorted(true_images):
                self.samples.append(os.path.join(self.true_dir, img_name))
                self.labels.append(0)
        
        # Load fake images (label 1)
        if os.path.exists(self.fake_dir):
            fake_images = [f for f in os.listdir(self.fake_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            for img_name in sorted(fake_images):
                self.samples.append(os.path.join(self.fake_dir, img_name))
                self.labels.append(1)
        
        print(f"Loaded {len(self.samples)} images:")
        print(f"  Real: {self.labels.count(0)}")
        print(f"  Fake: {self.labels.count(1)}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path = self.samples[idx]
        label = self.labels[idx]
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        if self.extract_clip_features:
            # Extract CLIP features
            with torch.no_grad():
                # Image features
                image_input = self.clip_preprocess(image).unsqueeze(0).to(self.device)
                image_features = self.clip_model.encode_image(image_input)
                image_features = image_features.cpu().squeeze()
                
                # Use a default caption
                text = "a photo"
                text_input = clip.tokenize([text]).to(self.device)
                text_features = self.clip_model.encode_text(text_input)
                text_features = text_features.cpu().squeeze()
            
            return image_features, text_features, torch.tensor(label, dtype=torch.float32)
        else:
            if self.transform:
                image = self.transform(image)
            return image, torch.tensor(label, dtype=torch.float32)