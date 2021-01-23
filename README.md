# FACT-2021

## Notes
* The baseline codebase for this project is provided by the authors at [1] (https://github.com/liuem607/expVAE)

## Anomaly Maps Instructions
### Installation and Setup
```
git clone https://github.com/Carmoondedraak/FACT-2021.git
cd FACT-2021/expVAE_pl
pip install -r ./requirements.txt
```
### Usage
For a list of all possible arguments, run the following command:
```
python exp_vae_pl.py -h
```
There are 3 datasets used, namely MNIST, UCSD Pedestrian and MVTec-AD. Each can be downloaded and extracted manually in the folder ```./Datasets/```. However, the dataset specified by argument ```--dataset``` will be downloaded and extracted automatically if it not present in the Datasets folder already.
#### Training
To train, run the script ```python exp_vae_pl.py```. By default, this will train and evaluate on the MNIST dataset for 100 epochs, with inlier digit 1 and outlier digit 7.
Examples of training:
```
python exp_vae_pl.py --dataset mvtec --mvtec_object metal_nut --epochs 200 --lr 1e-4
```
This will train a Resnet-18 variational autoencoder on the MVTec Anomaly Detection datset, on images resized to 256x256. Log files are created in folders ```{dataset}_logs/lightning_logs/version_{version#}```, in which we occasionally save sampled input images, reconstruction images, and attention maps on the outlier classes. For the MVTec-AD and UCSD Pedestrian datasets, we include target masks for the outlier class images.

#### Testing
To test, run the script with the added argument of the specified dataset and model version ```python exp_vae_pl.py --dataset {dataset} --model_version {version#}```. This will create a new version folder in the lightning_logs folder, where it will save attention maps for outlier images in MNIST, and for MVTec-AD and UCSD Pedestrian, it will include target masks, binary localization images and the original outlier class images. Any quantitative metrics are output at the end of testing, such as AUROC and best IoU.

#### Tensorboard usage
For more detailed comparison between model performance and metrics, as well as the sample images, run tensorboard on any of the datasets' lightning_logs directory:
```tensorboard --logdir ./{dataset}_logs/lightning_logs```
