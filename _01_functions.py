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





# function for plot the images
def plot_images(dataset):
    for i in range(9):
        plt.subplot(330 + 1 + i)
        plt.imshow(dataset[i][0][0])
        plt.grid(False)
        plt.xticks([])
        plt.yticks([])
    plt.show()



