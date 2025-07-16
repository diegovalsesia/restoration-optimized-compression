
# restoration-optimized-compression
Lidar-guided image compression optimized for Lidar-guided restoration. In this repository a Lidar-guided image compression model is integrated with a Lidar-guided iamge restoration model for end-to-end optimization so that rate-distortion-after-restoration performance is maximized. 

A demo is also provided with the source code of an iOS app that implements the restoration-optimized compressor on the smartphone via an Executoch exported model. Server-side decompression and restoration code is also provided for the demo.

## Install

```
pip install -U pip 
pip install torch torchvision
pip install pillow==9.2.0
pip install shapely==1.7.1 
pip install -e . 
pip install timm tqdm click
```

### Usage examples

```
cd examples
python myscript.py -c ../config/myconfig.yaml
```

Replace *myscript* with one of the ones available in the examples directory:

 - variablerate_with_lidar_with_restoration_finetune_mse.py: this allows end-to-end finetuning of compression and restoration with the MSE loss. Note it adds noise to training and testing data to simulate noisy data.
 - variablerate_with_lidar_with_restoration_finetune_mse.py: this allows end-to-end finetuning of compression and restoration with the LPIPS perceptual loss
 - variablerate_with_lidar_with_restoration_test.py: test joint compression and restoration model
Pretraining script for independent restorer and compression model are also available.

Replace *myconfig* with one of the ones available in the examples directory. Make sure to change the paths to your own data and models. Restoration models need to be specified with the examples/config/KernelDepthDeblur.yaml (for training) and examples/config/KernelDepthDeblur_test.yaml (for testing) configuration files.

Trained models as well as the Executorch exported compressor for the demo are avaiable at the [following link](https://www.dropbox.com/scl/fo/ve2vk6ucr0tjrn1v2lvuf/ALzo9INzaGI6fWdSvu4f58g?rlkey=g98ljrltjg4tw1wa92zx8frsk&st=rnlxc9i4&dl=0).


## Acknowledgements
This study was carried out within the “AI-powered LIDAR fusion for next-generation smartphone cameras (LICAM)” project – funded by European Union – Next Generation EU within the PRIN 2022 program (D.D. 104 - 02/02/2022 Ministero dell’Università e della Ricerca). This manuscript reflects only the authors' views and opinions and the Ministry cannot be considered responsible for them.

