#===============================================================================
# The main function provides recommendations for guest_id
# based on 2 Models. 
# 
# 1. Logistic Matrix Factorization
# 2. Popularity Model
# 
# Sample input: guest_id = 2323
# 
# 
# Sample Output:
# 
# Items based on Logistic Matrix Factorization for Guest  2323 
# 
# [95543, 47326, 46061, 41191, 110407, 41858, 46059, 93412, 43941, 41552, 79599, 48584, 42596, 47328, 42027, 108573, 58299, 48174, 44042, 99480]
# Most Popular Items 
# 
# [ 109559.   46061.   47330.   95543.   47323.  109538.  105950.  222271.
#    47326.   41179.   42877.   47328.   46058.  110407.   54896.  106570.
#   109539.   41191.   46059.   53852.]
# 
# Based on the available information, it is better to recommend the most popular items
#===============================================================================



import time
import numpy as np
import pandas as pd
from logMFLearning import *
from gst_item import *
from evaluate import *
from data_preprocessing import *




if __name__ == "__main__":
    
    
    
    # Search for Guest ID. 
    guest_id = 2323
    
    
    # Read data from CSV
    purchases = pd.read_csv('purchases.csv', sep = ',')
    
    
    # Preprocess the data based on exploratory analysis
    dataForWork = data_preprocessing(purchases)
    dataForWork.clean_data()
   
    # Using 5 fold Cross Validation, we observed that the number of latent features 
    # was 1 and the regularization parameter was 0.045
    
    
     
    
    gst_item_Model = gst_item()
    gst_item_Model.prepare_data(dataForWork.data)
    
    gst_item_Matrix = gst_item_Model.load_matrix(collaborative=0)
    
    nfeatures = 1
    reg_param = 0.045
    niter = 3
    gamma = 1
    nrec = 20
    
    
    # Logistic Matrix Factorization Model
    Model = LogisticMF(gst_item_Matrix, nfeatures, reg_param, gamma, niter)
    Model.train_model()
    gst_item_Model.get_probabilities(Model.print_probabilities())
    
    gst_item_Model.get_top_n_recommendations_gst(nrec)
    top_n_recommendations_for_Guest = gst_item_Model.get_top_n_recommendations_for_user(guest_id)
    if (top_n_recommendations_for_Guest == None):
        print "User Not Found. Based on available data, it is anyways safe to recommend the most popular items"
    else:
        print "Items based on Logistic Matrix Factorization for Guest ",guest_id,"\n"
        print top_n_recommendations_for_Guest
        
        
    print "Most Popular Items \n"
    print gst_item_Model.get_top_n_popular_items(nrec)
    
    print "\n Based on the available information, it is better to recommend the most popular items"
    
    
    
    
       
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    



