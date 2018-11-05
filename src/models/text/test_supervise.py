#

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from sklearn.decomposition import TruncatedSVD
import matplotlib as mpl
mpl.use('TkAgg')  # or whatever other backend that you want
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()  # for plot styling
import warnings
warnings.filterwarnings('ignore')

data = pd.read_csv('/home/mickael/Documents/Challenge_Video_Audio_Text/features/text/violence.csv',sep='§')

