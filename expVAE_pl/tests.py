import os
from utils import set_work_directory
from datetime import datetime

set_work_directory()

mvtec_classnames = ['metal_nut', 'pill', 'screw', 'tile', 'toothbrush', 'transistor', 'wood', 'zipper', 
                    'bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut', 'leather']

def get_version_nums(dataset='mvtec'):
    logs_path = os.path.join(f'{dataset}_logs', 'lightning_logs')
    log_folders = os.listdir(logs_path)
    version_nums = sorted([int(folder.rsplit('_')[1]) for folder in log_folders])
    return version_nums

class RunLogger(object):
    def __init__(self, dataset=None, obj=None, version=None, train_digit=None, test_digit=None, train=None):
        self.dataset = dataset
        self.obj = obj
        self.version = version
        self.train = train
        self.train_digit = train_digit
        self.test_digit = test_digit

    def start(self):
        self.starttime = datetime.now()
        if self.dataset == 'mvtec':
            if self.train:
                print(f'TRAINING START dataset: {self.dataset} - object {self.obj} - version {self.version}, start time: {self.starttime.strftime("%H:%M:%S")}')
            else:
                print(f'TESTING START dataset: {self.dataset} - object {self.obj} - version {self.version}, start time: {self.starttime.strftime("%H:%M:%S")}')
        if self.dataset == 'ucsd':
            if self.train:
                print(f'TRAINING START dataset: {self.dataset} - version {self.version}, start time: {self.starttime.strftime("%H:%M:%S")}')
            else:
                print(f'TESTING START dataset: {self.dataset} - version {self.version}, start time: {self.starttime.strftime("%H:%M:%S")}')
        if self.dataset == 'mnist':
            if self.train:
                print(f'TRAINING START dataset: {self.dataset} - train_digit {self.train_digit} - version {self.version}, start time: {self.starttime.strftime("%H:%M:%S")}')
            else:
                print(f'TESTING START dataset: {self.dataset} - train_digit {self.train_digit} - test_digit {self.test_digit} - version {self.version}, start time: {self.starttime.strftime("%H:%M:%S")}')
        
    def stop(self):
        self.stoptime = datetime.now()
        delta = self.stoptime - self.starttime
        print(f'TRAINING END end time: {self.stoptime.strftime("%H:%M:%S")}, time elapsed: {str(delta)}')


def run_script(dataset='mvtec', seed=1, lr=5e-4):
    if dataset == 'mvtec':
        for obj in mvtec_classnames:
            curr_version = get_version_nums(dataset)[-1]+1

            logger = RunLogger(dataset=dataset, obj=obj, version=curr_version, train=True)
            logger.start()
            os.system(f'python exp_vae_pl.py --epochs 300 --dataset {dataset} --mvtec_object {obj} --init_seed {seed} --batch_size 16 --lr {lr}')
            logger.stop()

            logger = RunLogger(dataset=dataset, obj=obj, version=curr_version, train=False)
            logger.start()
            os.system(f'python exp_vae_pl.py --eval True --dataset {dataset} --init_seed {seed} --batch_size 16 --test_digit 5 --model_version {curr_version}')
            logger.stop()

    elif dataset == 'mnist':

        curr_version = get_version_nums(dataset)[-1]+1
        
        
        logger = RunLogger(dataset=dataset, version=curr_version, train=True, train_digit=1)
        logger.start()
        os.system(f'python exp_vae_pl.py --epochs 400 --dataset {dataset} --init_seed {seed} --batch_size 128 --train_digit 1 --lr {lr}')
        logger.stop()

        logger = RunLogger(dataset=dataset, version=curr_version, train=False, train_digit=1, test_digit=7)
        logger.start()
        os.system(f'python exp_vae_pl.py --eval True --dataset {dataset} --init_seed {seed} --batch_size 64 --test_digit 7 --model_version {curr_version}')
        logger.stop()
        
        logger = RunLogger(dataset=dataset, version=curr_version, train=False, train_digit=1, test_digit=4)
        logger.start()
        os.system(f'python exp_vae_pl.py --eval True --dataset {dataset} --init_seed {seed} --batch_size 64 --test_digit 4 --model_version {curr_version}')
        logger.stop()
        
        logger = RunLogger(dataset=dataset, version=curr_version, train=False, train_digit=1, test_digit=9)
        logger.start()
        os.system(f'python exp_vae_pl.py --eval True --dataset {dataset} --init_seed {seed} --batch_size 64 --test_digit 9 --model_version {curr_version}')
        logger.stop()
        
        logger = RunLogger(dataset=dataset, version=curr_version, train=False, train_digit=1, test_digit=2)
        logger.start()
        os.system(f'python exp_vae_pl.py --eval True --dataset {dataset} --init_seed {seed} --batch_size 64 --test_digit 2 --model_version {curr_version}')
        logger.stop()
        
        curr_version = get_version_nums(dataset)[-1]+1

        logger = RunLogger(dataset=dataset, version=curr_version, train=True, train_digit=3)
        logger.start()
        os.system(f'python exp_vae_pl.py --epochs 400 --dataset {dataset} --init_seed {seed} --batch_size 128 --train_digit 3 --lr {lr}')
        logger.stop()
        
        logger = RunLogger(dataset=dataset, version=curr_version, train=False, train_digit=3, test_digit=8)
        logger.start()
        os.system(f'python exp_vae_pl.py --eval True --dataset {dataset} --init_seed {seed} --batch_size 64 --test_digit 8 --model_version {curr_version}')
        logger.stop()
        
        logger = RunLogger(dataset=dataset, version=curr_version, train=False, train_digit=3, test_digit=5)
        logger.start()
        os.system(f'python exp_vae_pl.py --eval True --dataset {dataset} --init_seed {seed} --batch_size 64 --test_digit 5 --model_version {curr_version}')
        logger.stop()
    else:

        curr_version = get_version_nums(dataset)[-1]+1

        logger = RunLogger(dataset=dataset, version=curr_version, train=True)
        logger.start()
        os.system(f'python exp_vae_pl.py --epochs 500 --dataset {dataset} --init_seed {seed} --batch_size 128 --lr {lr}')
        logger.stop()

        logger = RunLogger(dataset=dataset, version=curr_version, train=False)
        logger.start()
        os.system(f'python exp_vae_pl.py --eval True --dataset {dataset} --init_seed {seed} --batch_size 64 --model_version {curr_version}')
        logger.stop()

run_script(dataset='mnist', seed=1)
run_script(dataset='mvtec', seed=3)
run_script(dataset='ucsd', seed=1)
