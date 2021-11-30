import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix


def create_confusion_matrix(y_test, y_pred):
    cf_matrix = confusion_matrix(y_test, y_pred)
    sns.heatmap(cf_matrix / np.sum(cf_matrix), linewidths=1,
                cmap='RdBu', linecolor='white', annot=True,
                fmt='.1%')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
  
    