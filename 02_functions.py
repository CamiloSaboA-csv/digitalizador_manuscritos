import numpy as np
#import tensorflow as tf
import os

from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
from sklearn.preprocessing import MinMaxScaler, StandardScaler, FunctionTransformer
from sklearn.pipeline import Pipeline

import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def data_from_google_drive(url):
    """
    Access data from a google drive link.
    :param url: Url to data.
    :return: Data.
    """
    import requests
    import io
    response = requests.get(url, stream=True)
    file_like = io.BytesIO(response.content)
    return file_like






# build a function to import data and prepare for a machine learning model
def import_data(path):
    """
    Import images from path.
    :param path: Path to images.
    :return: List of images.
    """
    images = []
    labels = []
    for filename in os.listdir(path):
        img = mpimg.imread(os.path.join(path, filename))
        images.append(img)
        labels.append(filename.split('.')[0])
    return images, labels



