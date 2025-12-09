import os
from src.config.settings import Settings


class DataLoader:
    def __init__(self, dataset_dir: str):
        self.dataset_dir = dataset_dir
        self.settings = Settings()
        self.class_to_index = {name: i for i, name in enumerate(self.settings.classes)}

    def load_data(self):
        images=[]
        labels=[]
        for class_name in os.listdir(self.dataset_dir):
            class_path=os.path.join(self.dataset_dir,class_name)
            if not os.path.isdir(class_path):
                continue
            if class_name not in self.settings.classes:
                continue
            class_index = self.class_to_index[class_name]
            for file in os.listdir(class_path):
                if file.lower().endswith((".jpg",".jpeg",".png")):
                    images.append(os.path.join(class_path, file))
                    labels.append(class_index)
        return images,labels

