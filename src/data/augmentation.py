import os
import random
from PIL import Image
from torchvision import transforms
from src.config.settings import Settings

class Augmentor:
    def _init_(self,target_count=500):
        self.target_count = target_count
        self.settings = Settings()

        self.class_to_index={name:i for i,name in enumerate(self.settings.classes)}
        self.processed_dir = "processed"
        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)

        self.transform = transforms.Compose([
            transforms.RandomAffine(degrees=20, scale=(0.8, 1.2)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(
                brightness=0.3,
                contrast=0.3,
                saturation=0.3,
                hue=0.05
            )
        ])
    
    def process_class(self,class_name,image_paths):
        out_dir = os.path.join(self.processed_dir, class_name)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        cls_idx=self.class_to_index[class_name]
        processed_imgs=[]
        for img_path in image_paths:
            file_name=os.path.basename(img_path)
            dst=os.path.joint(out_dir,file_name)
            if not os.path.exists(dst):
                try:
                    Image.open(img_path).convert("RGB").save(dst)
                except:
                    continue

            processed_images.append((dst, cls_idx))
        current_count=len(os.listdir(out_dir))
        need = self.target_count - current_count
        if need>0:
            images_only=[os.path.basename(img_path) for img_path in image_paths]
            generated=0
            while generated<need:
                img_file=random.choice(images_only)
                src=os.path.join(os.path.dirname(img_paths[0]),img_file)
                try:
                    img = Image.open(src).convert("RGB")
                except:
                    continue
                aug_img = self.transform(img)
                aug_name = f"aug_{generated}.jpg"
                aug_path = os.path.join(out_dir, aug_name)
                aug_img.save(aug_path)

                processed_images.append((aug_path, cls_idx))
                generated += 1

        return processed_images


