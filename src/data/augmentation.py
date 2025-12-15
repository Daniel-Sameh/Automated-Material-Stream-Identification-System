import os
import numpy as np
import random
from PIL import Image
from torchvision import transforms
from src.config.settings import Settings
from tqdm import tqdm
import io

class Augmentor:
    def __init__(self,target_count=500):
        self.target_count = target_count
        self.settings = Settings()

        self.class_to_index={name:i for i,name in enumerate(self.settings.classes)}
        self.processed_dir = "data\processed"
        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)

        self.transform = transforms.Compose([
            # transforms.RandomResizedCrop(
            #     size=224,  # match ResNet crop target better than 128
            #     scale=(0.6, 1.0),
            #     ratio=(0.8, 1.2),
            #     antialias=True
            # ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([transforms.RandomPerspective(distortion_scale=0.25, p=1.0)], p=0.35),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=7, sigma=(0.1, 2.0))], p=0.35),
            transforms.ColorJitter(
                brightness=0.6,
                contrast=0.6,
                saturation=0.25,
                hue=0.08
            ),
            transforms.RandomRotation(degrees=15),
        ])
    
    def jpeg_compress(self, img: Image.Image) -> Image.Image:
        """Simulate webcam compression artifacts."""
        quality = random.randint(30, 85)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality)
        buf.seek(0)
        return Image.open(buf).convert("RGB")

    def add_sensor_noise(self, img: Image.Image) -> Image.Image:
        """Simulate sensor noise in low light."""
        arr = np.array(img).astype(np.float32)
        sigma = random.uniform(2.0, 12.0)  # tune if needed
        noise = np.random.normal(0, sigma, arr.shape).astype(np.float32)
        arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(arr)
    
    def process_class(self,class_name,image_paths):
        out_dir = os.path.join(self.processed_dir, class_name)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        cls_idx=self.class_to_index[class_name]
        processed_imgs=[]

        for img_path in tqdm(image_paths, desc=f"Copying {class_name}", leave=False):
            file_name=os.path.basename(img_path)
            dst=os.path.join(out_dir,file_name)
            if not os.path.exists(dst):
                try:
                    Image.open(img_path).convert("RGB").save(dst)
                except:
                    continue

            processed_imgs.append((dst, cls_idx))

        current_count=len(os.listdir(out_dir))
        need = self.target_count - current_count
        if need>0:
            generated=0
            attempts = 0
            max_attempts = need * 10
            pbar = tqdm(total=need, desc=f"Augmenting {class_name}", leave=False)

            while generated<need and attempts < max_attempts:
                attempts += 1
                src=random.choice(image_paths)
                try:
                    img = Image.open(src).convert("RGB")
                    aug_img = self.transform(img)

                    # compression + noise sometimes
                    if random.random() < 0.6:
                        aug_img = self.jpeg_compress(aug_img)
                    if random.random() < 0.4:
                        aug_img = self.add_sensor_noise(aug_img)

                    # Include source filename for tracking
                    src_name = os.path.splitext(os.path.basename(src))[0]
                    aug_name = f"aug_{src_name}_{generated}.jpg"
                    aug_path = os.path.join(out_dir, aug_name)
                    aug_img.save(aug_path)
                    processed_imgs.append((aug_path, cls_idx))
                    generated += 1
                    pbar.update(1)
                except:
                    continue
            pbar.close()

        return processed_imgs

