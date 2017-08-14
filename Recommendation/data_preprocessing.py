import time
import numpy as np
import pandas as pd
from logMFLearning import *
from gst_item import *
from evaluate import *
from sklearn.utils import shuffle




class data_preprocessing():
    
    def __init__(self, data):
        self.data = data
    
    
    def clean_data(self):
        
        # Filter data with positive quantities only. 
        self.data = self.data[self.data.qty > 0.0]
    
        # Remove date purchased column as it is not used in the model
        self.data = self.data.drop('purchase_d', axis=1)
        
        # Remove rows with missing values
        self.data = self.data.dropna().reset_index(drop=True)
        #print len(self.data)
        
        # Remove entries with "?" mark. Hard coded keys. Could be replaced.
        self.data = self.data[self.data.item_i.str.contains('\?')==False]
        self.data = self.data[self.data.gst_i.str.contains('\?')==False]
            
              
        
        

    # Function to drop items which appear only once in the list 
    # These are the items that have been bought only once.
    def drop_items_gst_less_than_n(self, nItems):
    
        dataFiltered = self.data.groupby('item_i').filter(lambda x: len(x) > nItems)
        self.data = dataFiltered
    
    
    # Split the data into kFolds. Returns a list of PD dataframes.  
    def split_data_in_K_Folds(self, nFolds, cols):
        
        # This is the original data that has been cleaned
        Original_Data = self.data
        
        # Shuffle the original data for randomization
        self.data = shuffle(self.data)
        
        test_data_DFs = []
        
        train_data_DFs = []
        
        
        # Split the data into K_Splits. This is done by converting the df to a numpy array
        K_Splits = np.array_split(self.data, nFolds)
        
        for i in range(0, nFolds): 
            
            # For the ith fold, the test data is the split.
            test_data_DF = pd.DataFrame(K_Splits[i], columns = cols)
                
            # The train data is the Original Data minus the test data.
            train_data_DF = Original_Data.drop(test_data_DF.index).reset_index(drop = True)
            
            
            # Put the train and test data into a list
            train_data_DFs.append(train_data_DF)
            test_data_DFs.append(test_data_DF.reset_index(drop=True))
            
        return test_data_DFs, train_data_DFs
    