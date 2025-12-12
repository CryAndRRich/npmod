# This script tests various custom conditional difffusion models
# The results are under print function calls in case you dont want to run the code

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from typing import List, Tuple, Dict
import json
import requests
import zipfile
import io
import tqdm
from concurrent.futures import ThreadPoolExecutor
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

# Importing the custom models
from models.deep_learning.diffusion.conditional import LatentDiffusionTrainer

# Loading COCO Dataset
class COCODataset(Dataset):
    def __init__(self, 
                 data_list: List[Dict[str, str]], 
                 img_dir: str, 
                 transform: transforms.Compose = None) -> None:
        self.data = data_list
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
        item = self.data[idx]
        img_path = os.path.join(self.img_dir, item["file_name"])
        caption = item["prompt"]

        try:
            image = Image.open(img_path).convert("RGB")
            if self.transform:
                image = self.transform(image)
            return image, caption
        except:
            return self.__getitem__((idx + 1) % len(self.data))

if __name__ == "__main__":
    # === Load Dataset === 
    # Load MSCOCO2017 coco_dataset
    num_samples = 500
    img_dir = "data/coco_real_500"
    os.makedirs(img_dir, exist_ok=True)

    url = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
    r = requests.get(url)
    z = zipfile.ZipFile(io.BytesIO(r.content))
    captions_data = json.loads(z.read("annotations/captions_val2017.json"))
    captions_data["annotations"].sort(key=lambda x: x["image_id"])

    dataset_pairs = []
    for item in captions_data["annotations"][:num_samples]:
        dataset_pairs.append({
            "prompt": item["caption"],
            "id": item["image_id"],
            "file_name": f"{item['image_id']:012d}.jpg"
        })

    def download_image(img_info: Dict[str, str]) -> None:
        img_url = f"http://images.cocodataset.org/val2017/{img_info['file_name']}"
        save_path = os.path.join(img_dir, img_info["file_name"])
        
        if not os.path.exists(save_path):
            try:
                r = requests.get(img_url, timeout=10)
                if r.status_code == 200:
                    with open(save_path, "wb") as f:
                        f.write(r.content)
            except:
                pass

    with ThreadPoolExecutor(max_workers=10) as executor:
        list(tqdm.tqdm(executor.map(download_image, dataset_pairs), total=len(dataset_pairs)))

    transform_coco = transforms.Compose([
        transforms.Resize((512, 512)), 
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    coco_dataset = COCODataset(dataset_pairs, img_dir, transform=transform_coco)
    coco_loader = DataLoader(coco_dataset, batch_size=2, shuffle=True, num_workers=2)
    # ====================

    # === Test Diffusion === 
    models = {
        "Stable Diffusion 1.5": "SD1.5",
        "Kandinsky 2.2": "KANDINSKY2.2",
        "Hunyuan DiT": "HUNYUAN",
        "Stable Diffusion 3": "SD3",
        "FLUX.1": "FLUX"
    }

    for name, model_type in models.items():
        print("==============================================================")
        print(f"{name} Result")
        print("==============================================================")

        trainer = LatentDiffusionTrainer(
            model_type=model_type,  
            learn_rate=1e-5,   
            number_of_epochs=5,  
            device="cuda"
        )

        trainer.fit(coco_loader, verbose=True)

    """
    ==============================================================
    Stable Diffusion 1.5 Result
    ==============================================================
    100%|██████████| 250/250 [02:26<00:00,  1.71it/s, Epoch=1/5, MSE=0.3458, Type=SD1.5]
    Epoch [1/5] | Avg Loss: 0.64049
    100%|██████████| 250/250 [02:37<00:00,  1.59it/s, Epoch=2/5, MSE=0.3099, Type=SD1.5]
    Epoch [2/5] | Avg Loss: 0.33342
    100%|██████████| 250/250 [02:37<00:00,  1.59it/s, Epoch=3/5, MSE=0.1409, Type=SD1.5]
    Epoch [3/5] | Avg Loss: 0.27078
    100%|██████████| 250/250 [02:37<00:00,  1.59it/s, Epoch=4/5, MSE=0.1486, Type=SD1.5]
    Epoch [4/5] | Avg Loss: 0.23306
    100%|██████████| 250/250 [02:37<00:00,  1.59it/s, Epoch=5/5, MSE=0.1203, Type=SD1.5]
    Epoch [5/5] | Avg Loss: 0.23839

    ==============================================================
    Kandinsky 2.2 Result
    ==============================================================
    100%|██████████| 250/250 [02:32<00:00,  1.63it/s, Epoch=1/5, MSE=3.9385, Type=KANDINSKY]
    Epoch [1/5] | Avg Loss: 8.66115
    100%|██████████| 250/250 [02:35<00:00,  1.61it/s, Epoch=2/5, MSE=4.4825, Type=KANDINSKY]
    Epoch [2/5] | Avg Loss: 4.60615
    100%|██████████| 250/250 [02:36<00:00,  1.60it/s, Epoch=3/5, MSE=2.9394, Type=KANDINSKY]
    Epoch [3/5] | Avg Loss: 3.17558
    100%|██████████| 250/250 [02:35<00:00,  1.60it/s, Epoch=4/5, MSE=2.3345, Type=KANDINSKY]
    Epoch [4/5] | Avg Loss: 2.41801
    100%|██████████| 250/250 [02:35<00:00,  1.60it/s, Epoch=5/5, MSE=2.2343, Type=KANDINSKY]
    Epoch [5/5] | Avg Loss: 1.93111
    
    ==============================================================
    Hunyuan DiT Result
    ==============================================================
    100%|██████████| 250/250 [03:04<00:00,  1.35it/s, Epoch=1/5, MSE=0.8856, Type=HUNYUAN]
    Epoch [1/5] | Avg Loss: 0.94362
    100%|██████████| 250/250 [03:10<00:00,  1.31it/s, Epoch=2/5, MSE=0.7298, Type=HUNYUAN]
    Epoch [2/5] | Avg Loss: 0.80987
    100%|██████████| 250/250 [03:10<00:00,  1.31it/s, Epoch=3/5, MSE=0.8306, Type=HUNYUAN]
    Epoch [3/5] | Avg Loss: 0.74873
    100%|██████████| 250/250 [03:10<00:00,  1.31it/s, Epoch=4/5, MSE=0.5746, Type=HUNYUAN]
    Epoch [4/5] | Avg Loss: 0.70021
    100%|██████████| 250/250 [03:10<00:00,  1.31it/s, Epoch=5/5, MSE=0.6452, Type=HUNYUAN]
    Epoch [5/5] | Avg Loss: 0.67128

    ==============================================================
    Stable Diffusion 3 Result
    ==============================================================
    100%|██████████| 250/250 [03:16<00:00,  1.27it/s, Epoch=1/5, MSE=1.5216, Type=SD3]
    Epoch [1/5] | Avg Loss: 1.65572
    100%|██████████| 250/250 [03:21<00:00,  1.24it/s, Epoch=2/5, MSE=1.4149, Type=SD3]
    Epoch [2/5] | Avg Loss: 1.42249
    100%|██████████| 250/250 [03:22<00:00,  1.24it/s, Epoch=3/5, MSE=1.2551, Type=SD3]
    Epoch [3/5] | Avg Loss: 1.29506
    100%|██████████| 250/250 [03:22<00:00,  1.24it/s, Epoch=4/5, MSE=1.3251, Type=SD3]
    Epoch [4/5] | Avg Loss: 1.23058
    100%|██████████| 250/250 [03:21<00:00,  1.24it/s, Epoch=5/5, MSE=1.0900, Type=SD3]
    Epoch [5/5] | Avg Loss: 1.18551

    ==============================================================
    FLUX.1 Result
    ==============================================================
    100%|██████████| 250/250 [06:29<00:00,  1.56s/it, Epoch=1/5, MSE=1.4684, Type=FLUX]
    Epoch [1/5] | Avg Loss: 1.60563
    100%|██████████| 250/250 [06:32<00:00,  1.57s/it, Epoch=2/5, MSE=1.1621, Type=FLUX]
    Epoch [2/5] | Avg Loss: 1.45842
    100%|██████████| 250/250 [06:32<00:00,  1.57s/it, Epoch=3/5, MSE=1.2669, Type=FLUX]
    Epoch [3/5] | Avg Loss: 1.31992
    100%|██████████| 250/250 [06:32<00:00,  1.57s/it, Epoch=4/5, MSE=1.0973, Type=FLUX]
    Epoch [4/5] | Avg Loss: 1.24472
    100%|██████████| 250/250 [06:32<00:00,  1.57s/it, Epoch=5/5, MSE=1.0846, Type=FLUX]
    Epoch [5/5] | Avg Loss: 1.19436
    """