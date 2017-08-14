import time
import numpy as np
import pandas as pd
from logMFLearning import *
from gst_item import *
from evaluate import *
from data_preprocessing import *



if __name__ == "__main__":
    
    # Read data from CSV
    purchases = pd.read_csv('purchases.csv', sep = ',')
    
    #Changing
    
    
    # Preprocess the data based on exploratory analysis
    dataForWork = data_preprocessing(purchases)
    dataForWork.clean_data()
    
    # Choose number of items to drop that occur nItems times in the list
    # This is to drop un-common items from the model. 
    
    nItems = 3
    dataForWork.drop_items_gst_less_than_n(nItems)
    
   
    # Using 5 fold Cross Validation, we observed that the number of latent features 
    # was 1 and the regularization parameter was 0.045
    
    # Choose number of items to drop that occur nItems times in the list
    # This is to drop un-common items from the model. 
    nItems = 3
    dataForWork.drop_items_gst_less_than_n(nItems)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    



