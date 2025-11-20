# This script tests various custom gan models
# The results are under print function calls and in data/img/result_gan.png in case you dont want to run the code

import os
import sys 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Importing the custom models
from models.deep_learning.gan.VanillaGAN import VanillaGAN
from models.deep_learning.gan.DCGAN import DCGAN
from models.deep_learning.gan.LAPGAN import LAPGAN
from models.deep_learning.gan.WGAN import WGAN
from models.deep_learning.gan.ProGAN import ProGAN
from models.deep_learning.gan.BigGAN import BigGAN
from models.deep_learning.gan.StyleGAN import StyleGAN


if __name__ == "__main__":
    # === Load all datasets ===
    # Load rem images dataset
    # https://www.kaggle.com/datasets/andy8744/rezero-rem-anime-faces-for-gan-training/data
    data_rem_dir = r"D:\Project\npmod\data\rem"
    batch_rem_size = 64
    img_rem_size = 64 
    transform_rem = transforms.Compose([
        transforms.Resize((img_rem_size, img_rem_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    rem_dataset = datasets.ImageFolder(root=data_rem_dir, transform=transform_rem)
    rem_loader = DataLoader(rem_dataset, batch_size=batch_rem_size, shuffle=True, num_workers=0)
    # ====================

    # === Test GAN === 
    models = {
        "GAN": VanillaGAN(latent_dim=100, img_shape=(3, img_rem_size, img_rem_size), learn_rate=0.0002, number_of_epochs=200),
        "DCGAN": DCGAN(latent_dim=100, img_shape=(3, img_rem_size, img_rem_size), learn_rate=0.0002, number_of_epochs=200),
        "LAPGAN": LAPGAN(latent_dim=100, img_shape=(3, img_rem_size, img_rem_size), learn_rate=0.0002, number_of_epochs=200),
        "WGAN": WGAN(latent_dim=100, img_shape=(3, img_rem_size, img_rem_size), learn_rate=0.0002, number_of_epochs=200),
        "ProGAN": ProGAN(latent_dim=100, img_size=img_rem_size, learn_rate=0.0002, number_of_epochs=50),
        "BigGAN": BigGAN(latent_dim=100, img_shape=(3, img_rem_size, img_rem_size), learn_rate=0.0002, number_of_epochs=50),
        "StyleGAN": StyleGAN(latent_dim=100, img_shape=(3, img_rem_size, img_rem_size), learn_rate=0.0002, number_of_epochs=200)
    }

    for name, model in models.items():
        print("==============================================================")
        print(f"{name} Result")
        print("==============================================================")
        
        model.fit(rem_loader, verbose=True)

    """
    ==============================================================
    GAN Result
    ==============================================================
    Epoch [50/200] | Loss_D: 0.2337 | Loss_G: 4.0750
    Epoch [100/200] | Loss_D: 0.2342 | Loss_G: 3.4582
    Epoch [150/200] | Loss_D: 0.2241 | Loss_G: 3.4523
    Epoch [200/200] | Loss_D: 0.2239 | Loss_G: 3.4819

    ==============================================================
    DCGAN Result
    ==============================================================
    Epoch [50/200] | Loss_D: 0.3833 | Loss_G: 4.4049
    Epoch [100/200] | Loss_D: 0.3722 | Loss_G: 4.7590
    Epoch [150/200] | Loss_D: 0.2599 | Loss_G: 3.8049
    Epoch [200/200] | Loss_D: 0.4597 | Loss_G: 4.8593

    ==============================================================
    LAPGAN Result
    ==============================================================
    Epoch [50/200] | Loss_D: 0.2265 | Loss_G: 3.5284
    Epoch [100/200] | Loss_D: 0.2225 | Loss_G: 3.4074
    Epoch [150/200] | Loss_D: 0.2239 | Loss_G: 3.3810
    Epoch [200/200] | Loss_D: 0.2230 | Loss_G: 3.3782

    ==============================================================
    WGAN Result
    ==============================================================
    Epoch [50/200] | Loss_C: -8.2964 | Loss_G: -6.1611
    Epoch [100/200] | Loss_C: -9.5346 | Loss_G: 1.3593
    Epoch [150/200] | Loss_C: -12.8878 | Loss_G: 1.3990
    Epoch [200/200] | Loss_C: -16.4291 | Loss_G: 1.9837

    ==============================================================
    ProGAN Result
    ==============================================================
    Step [1/5] | Epoch [50/50] | Loss_D: 1.3549 | Loss_G: 0.7097
    Step [2/5] | Epoch [50/50] | Loss_D: 1.2642 | Loss_G: 0.7843
    Step [3/5] | Epoch [50/50] | Loss_D: 1.0677 | Loss_G: 0.9604
    Step [4/5] | Epoch [50/50] | Loss_D: 0.6857 | Loss_G: 1.3805
    Step [5/5] | Epoch [50/50] | Loss_D: 0.3198 | Loss_G: 2.0715

    ==============================================================
    BigGAN Result
    ==============================================================
    Epoch [10/50] | Loss_D: 2.0436 | Loss_G: -0.4828
    Epoch [20/50] | Loss_D: 1.4443 | Loss_G: -0.3151
    Epoch [30/50] | Loss_D: 0.9755 | Loss_G: 0.4011
    Epoch [40/50] | Loss_D: 0.3121 | Loss_G: 0.8580
    Epoch [50/50] | Loss_D: 0.0749 | Loss_G: 1.1855

    ==============================================================
    StyleGAN Result
    ==============================================================
    Epoch [50/200] | Loss_D: 1.2593 | Loss_G: 0.8845
    Epoch [100/200] | Loss_D: 1.2495 | Loss_G: 0.8419
    Epoch [150/200] | Loss_D: 1.1798 | Loss_G: 0.9054
    Epoch [200/200] | Loss_D: 1.0928 | Loss_G: 0.9465
    """