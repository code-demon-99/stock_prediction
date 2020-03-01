import numpy as np
import matplotlib.pyplot as plt , mpld3
from sklearn.preprocessing import scale
from TFANN import ANNR
from google.colab import files



# Dictionary used to map the company names with their relative file names
file_names={
    'Reliance': 'Data_sets/RELIANCErr2.csv' ,
    'HDFC' : 'Data-sets/HDFC.csv' ,
    'UltraCement'  : 'Data_sets/ULTRACEMCO.BO.csv',
    'Eicher':'Data_sets/EICHERMOT.BO.csv',
    'MRF':'Data_sets/MRF.NS.csv',
    'Surya Roshni': 'Data_sets/SURYAROSNI.BO.csv',

}
def import_dataset(file_name):
    stock_data = np.loadtxt(file_names[file_name], delimiter=",", skiprows=1, usecols=(1, 4))
    return scale(stock_data)
    
def train_dataset():

    
# prices = stock_data[:, 1].reshape(-1, 1)
# dates = stock_data[:, 0].reshape(-1, 1)
# plt.plot(dates[:, 0], prices[:, 0])
# plt.xlabel('Year', fontsize = 16)
# plt.ylabel('INR (in Cr)', fontsize = 16)
# plt.show()
