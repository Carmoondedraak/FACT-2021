"""utils.py"""

import os
import argparse
import subprocess


class DataGather(object):
    def __init__(self, *args):
        self.keys = args
        self.data = self.get_empty_data_dict()

    def get_empty_data_dict(self):
        return {arg:[] for arg in self.keys}

    def insert(self, **kwargs):
        for key in kwargs:
            self.data[key].append(kwargs[key])

    def flush(self):
        self.data = self.get_empty_data_dict()


def str2bool(v):
    # codes from : https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse

    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def grid2gif(image_str, output_gif, delay=100):
    """Make GIF from images.

    code from:
        https://stackoverflow.com/questions/753190/programmatically-generate-video-or-animated-gif-in-python/34555939#34555939
    """
    str1 = 'convert -delay '+str(delay)+' -loop 0 ' + image_str  + ' ' + output_gif
    subprocess.call(str1, shell=True)


def mkdirs(path):
    if not os.path.exists(path):
        os.makedirs(path)


# Implementation of the disentanglement metrics
import numpy

def factorvae_disentanglement_metric():
    """
        It is based on "Disentangling by Factorising" paper
    """
    scores_dict = {}

    global_variances = compute_variances() #TODO
    active_dims = prume_dimensions() #TODO
    
    if not active_dims().any(): # TODO if there is [None]*dims ??
        scores_dict[] = 0.
        scores_dict[] = 0.
        scores_dict[] = 0
        return scores_dict
    
    training_votes = generate_training_batch() # TODO
    classifier = np.argmax(training_votes, axis=0)
    other_index = np.arange(training_votes.shape[1])

    training_accuracy = np.sum(training_votes[classifier, other_index]) * 1. / np.sum(training_votes)
    print("Training set accuracy: %.2g", training_accuracy)

    eval_votes = generate_training_batch() # TODO
    eval_accuracy = np.sum(eval_votes[classifier,
                                    other_index]) * 1. / np.sum(eval_votes)
    print("Evaluation set accuracy: %.2g", eval_accuracy)

    scores_dict["train_accuracy"] = train_accuracy
    score_dict["eval_accuracy"] = eval_accuracy
    scores_dict["num_active_dims"] = len(active_dims)

    return scores_dict


def compute_variances():
    #TODO
    pass


def prume_dimensions():
    #TODO
    pass


def generate_training_batch():
    #TODO returns a numpy ndarray
    pass
