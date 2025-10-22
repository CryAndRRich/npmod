# This script tests various custom gan models
# The results are under print function calls in case you dont want to run the code

import os
import sys 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Importing the custom models
from models.deep_learning.gan.unsupervised.gan import VanillaGAN
from models.deep_learning.gan.unsupervised.dcgan import DCGAN
from models.deep_learning.gan.unsupervised.lapgan import LAPGAN
from models.deep_learning.gan.unsupervised.stylegan import StyleGAN
from models.deep_learning.gan.unsupervised.wgan import WGAN


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

    # GAN
    gan1 = VanillaGAN(
        latent_dim=100,
        img_shape=(3, 64, 64),
        learn_rate=0.0002,
        number_of_epochs=200
    )

    gan1.fit(rem_loader)

    """
    Epoch [001/200] Loss_D: 0.4359 | Loss_G: 1.2684
    Epoch [010/200] Loss_D: 0.2236 | Loss_G: 3.7194
    Epoch [020/200] Loss_D: 0.2502 | Loss_G: 4.0062
    Epoch [030/200] Loss_D: 0.2326 | Loss_G: 4.0180
    Epoch [040/200] Loss_D: 0.2344 | Loss_G: 3.9483
    Epoch [050/200] Loss_D: 0.2314 | Loss_G: 3.7936
    Epoch [060/200] Loss_D: 0.2369 | Loss_G: 4.0691
    Epoch [070/200] Loss_D: 0.2331 | Loss_G: 3.7132
    Epoch [080/200] Loss_D: 0.4770 | Loss_G: 4.4924
    Epoch [090/200] Loss_D: 0.2771 | Loss_G: 3.5463
    Epoch [100/200] Loss_D: 0.2380 | Loss_G: 3.6968
    Epoch [110/200] Loss_D: 0.2330 | Loss_G: 3.6717
    Epoch [120/200] Loss_D: 0.2665 | Loss_G: 4.3210
    Epoch [130/200] Loss_D: 0.2561 | Loss_G: 3.5855
    Epoch [140/200] Loss_D: 0.2521 | Loss_G: 4.0472
    Epoch [150/200] Loss_D: 0.2815 | Loss_G: 3.9268
    Epoch [160/200] Loss_D: 0.2571 | Loss_G: 4.0300
    Epoch [170/200] Loss_D: 0.2569 | Loss_G: 3.8555
    Epoch [180/200] Loss_D: 0.2691 | Loss_G: 4.3371
    Epoch [190/200] Loss_D: 0.2823 | Loss_G: 4.8428
    Epoch [200/200] Loss_D: 0.2652 | Loss_G: 3.7989
    """
    # ====================


    # DCGAN
    gan2 = DCGAN(
        latent_dim=100,
        img_shape=(3, 64, 64),
        learn_rate=0.0002,
        number_of_epochs=200
    )

    gan2.fit(rem_loader)

    """
    Epoch [001/200] Loss_D: 0.4277 | Loss_G: 6.1546
    Epoch [010/200] Loss_D: 0.5752 | Loss_G: 4.8088
    Epoch [020/200] Loss_D: 0.5715 | Loss_G: 3.9974
    Epoch [030/200] Loss_D: 0.4079 | Loss_G: 3.7986
    Epoch [040/200] Loss_D: 0.3842 | Loss_G: 3.8694
    Epoch [050/200] Loss_D: 0.3568 | Loss_G: 4.2101
    Epoch [060/200] Loss_D: 0.3790 | Loss_G: 4.0377
    Epoch [070/200] Loss_D: 0.3909 | Loss_G: 4.3103
    Epoch [080/200] Loss_D: 0.3624 | Loss_G: 3.8843
    Epoch [090/200] Loss_D: 0.3304 | Loss_G: 3.8102
    Epoch [100/200] Loss_D: 0.3022 | Loss_G: 4.1873
    Epoch [110/200] Loss_D: 0.3418 | Loss_G: 4.0031
    Epoch [120/200] Loss_D: 0.3079 | Loss_G: 4.2056
    Epoch [130/200] Loss_D: 0.2814 | Loss_G: 3.7282
    Epoch [140/200] Loss_D: 0.2903 | Loss_G: 3.7872
    Epoch [150/200] Loss_D: 0.2816 | Loss_G: 3.5888
    Epoch [160/200] Loss_D: 0.2871 | Loss_G: 3.7296
    Epoch [170/200] Loss_D: 0.2900 | Loss_G: 3.8996
    Epoch [180/200] Loss_D: 0.2623 | Loss_G: 3.5908
    Epoch [190/200] Loss_D: 0.2590 | Loss_G: 3.4880
    Epoch [200/200] Loss_D: 0.2636 | Loss_G: 3.5365
    """
    # ====================


    # LAPGAN
    gan3 = LAPGAN(
        latent_dim=100,
        img_shape=(3, 64, 64),
        learn_rate=0.0002,
        number_of_epochs=200
    )

    gan3.fit(rem_loader)

    """
    Epoch [001/200] Loss_D: 0.6280 | Loss_G: 0.8570
    Epoch [010/200] Loss_D: 0.3507 | Loss_G: 1.9665
    Epoch [020/200] Loss_D: 0.3361 | Loss_G: 2.3820
    Epoch [030/200] Loss_D: 0.2766 | Loss_G: 2.5243
    Epoch [040/200] Loss_D: 0.2537 | Loss_G: 2.8075
    Epoch [050/200] Loss_D: 0.2611 | Loss_G: 2.7074
    Epoch [060/200] Loss_D: 0.2670 | Loss_G: 2.6936
    Epoch [070/200] Loss_D: 0.2629 | Loss_G: 2.7458
    Epoch [080/200] Loss_D: 0.2523 | Loss_G: 2.7766
    Epoch [090/200] Loss_D: 0.2574 | Loss_G: 2.8392
    Epoch [100/200] Loss_D: 0.2562 | Loss_G: 2.7973
    Epoch [110/200] Loss_D: 0.2491 | Loss_G: 2.8886
    Epoch [120/200] Loss_D: 0.2457 | Loss_G: 2.9486
    Epoch [130/200] Loss_D: 0.2468 | Loss_G: 2.9169
    Epoch [140/200] Loss_D: 0.2422 | Loss_G: 2.9574
    Epoch [150/200] Loss_D: 0.2429 | Loss_G: 3.0082
    Epoch [160/200] Loss_D: 0.2442 | Loss_G: 3.0057
    Epoch [170/200] Loss_D: 0.2448 | Loss_G: 3.0320
    Epoch [180/200] Loss_D: 0.2418 | Loss_G: 3.0485
    Epoch [190/200] Loss_D: 0.2389 | Loss_G: 3.0774
    Epoch [200/200] Loss_D: 0.2393 | Loss_G: 3.0922
    """
    # ====================


    # WGAN
    gan4 = WGAN(
        latent_dim=100,
        img_shape=(3, 64, 64),
        learn_rate=0.0002,
        number_of_epochs=200
    )

    gan4.fit(rem_loader)

    """
    Epoch [001/200] Loss_D: -0.3393 | Loss_G: 0.2490
    Epoch [010/200] Loss_D: -1.1087 | Loss_G: 0.6307
    Epoch [020/200] Loss_D: -1.0912 | Loss_G: 0.6395
    Epoch [030/200] Loss_D: -1.0356 | Loss_G: 0.6224
    Epoch [040/200] Loss_D: -1.1297 | Loss_G: 0.6052
    Epoch [050/200] Loss_D: -0.9942 | Loss_G: 0.5992
    Epoch [060/200] Loss_D: -0.9632 | Loss_G: 0.5868
    Epoch [070/200] Loss_D: -0.9526 | Loss_G: 0.5936
    Epoch [080/200] Loss_D: -0.9271 | Loss_G: 0.5581
    Epoch [090/200] Loss_D: -0.9827 | Loss_G: 0.5601
    Epoch [100/200] Loss_D: -1.0289 | Loss_G: 0.5306
    Epoch [110/200] Loss_D: -0.8891 | Loss_G: 0.5321
    Epoch [120/200] Loss_D: -0.9470 | Loss_G: 0.5359
    Epoch [130/200] Loss_D: -0.8836 | Loss_G: 0.5252
    Epoch [140/200] Loss_D: -0.8951 | Loss_G: 0.5332
    Epoch [150/200] Loss_D: -0.8663 | Loss_G: 0.5238
    Epoch [160/200] Loss_D: -0.8190 | Loss_G: 0.5161
    Epoch [170/200] Loss_D: -0.8324 | Loss_G: 0.5001
    Epoch [180/200] Loss_D: -0.8121 | Loss_G: 0.4884
    Epoch [190/200] Loss_D: -0.8028 | Loss_G: 0.4908
    Epoch [200/200] Loss_D: -0.7943 | Loss_G: 0.4855
    """
    # ====================