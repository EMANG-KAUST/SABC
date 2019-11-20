from utils import train_test
from utils import feach_gen
from utils.train_test import fit_predict
from utils import preprocessing
from utils.load_data import load_data
from utils.CNN import apply_CNN
import os


# define path from the project directory to your datafile
datapath = '/data/data.csv'
# getting explicit location
filename = os.getcwd() + datapath
# load the data of the format (trials X sample data), where the y label is the last column
X, y = load_data(filename, header = 0, index_col = 0, binarize = True)

# Filter the data, and plot both the original and filtered signals.
X_flt = preprocessing.butter_bandpass_filter(X,
                                             lowcut = 0.01,
                                             highcut = 40,
                                             sampling_rate = 178,
                                             order = 5,
                                             how_to_filt = 'simultaneously')
#
# Try different feature generation methods for ML classifiers
# pure score first
print('No feature generation applied')
fit_predict(X, y, selection=True)
# SCSA
print('Semi-classical signal analysis')
fit_predict(feach_gen.gen_eigen(X), y, channel = 'single')

# major freqs
print('Major frequncies analysis')
fit_predict(feach_gen.major_freqs(X), y, selection=True)

# 1D CNN
print('1D CNN model')
apply_CNN(X, y, channel = 'single')
