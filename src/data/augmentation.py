import os
import random
from PIL import Image
from torchvision import transforms
from src.config.settings import Settings
from tqdm import tqdm

class Augmentor:
    def __init__(self,target_count=450):
        self.target_count = target_count
        self.settings = Settings()

        self.class_to_index={name:i for i,name in enumerate(self.settings.classes)}
        self.processed_dir = "data\processed"
        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)

        self.transform = transforms.Compose([
            transforms.RandomAffine(degrees=10, scale=(0.9, 1.1), translate=(0.08,0.08)),
            transforms.RandomHorizontalFlip(p=0.3),
            # transforms.RandomVerticalFlip(),
            transforms.ColorJitter(
                brightness=0.15,
                contrast=0.15,
                saturation=0.02,
                hue=0.02
            )
        ])
    
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
                    aug_name = f"aug_{generated}.jpg"
                    aug_path = os.path.join(out_dir, aug_name)
                    aug_img.save(aug_path)
                    processed_imgs.append((aug_path, cls_idx))
                    generated += 1
                    pbar.update(1)
                except:
                    continue
            pbar.close()

        return processed_imgs

