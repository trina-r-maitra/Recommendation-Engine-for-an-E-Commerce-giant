#===============================================================================
# 
# This is the code that has been used for Cross Validation
# 
# The program iterates from 0.0 to 0.1 in the Regularization 
# Parameter space and generates file ValidationData_nFeatures
# where n is the number of features to be specified. The user would
# then use Parameterization.py to plot the Recall % vs Regularization
# plot. 
# 
# ******Please specify the filename at the bottom of the code****** 
#
#===============================================================================

import time
import numpy as np
import pandas as pd
from logMFLearning import *
from gst_item import *
from evaluate import *
from data_preprocessing import *




      
purchases = pd.read_csv('purchases.csv', sep = ',')
 
# Preprocess the data based on exploratory analysis
dataForWork = data_preprocessing(purchases)
dataForWork.clean_data()
 
# Choose number of items to drop that occur nItems times in the list
# This is to drop un-common items from the model. 
nItems = 3
dataForWork.drop_items_gst_less_than_n(nItems)
 
 
#For K-Fold Cross Validation
nfolds = 5
test_dataSets, train_dataSets = dataForWork.split_data_in_K_Folds(nfolds, ['qty', 'item_i', 'gst_i'])
 
# No. of latent features  
nfeatures = 10
 
# Logistic Matrix Factorization Model Parameters
gamma = 1
niter = 20
 
 
# Top nrec recommendations in the Logistic Matrix Factorization Model
nrec = 100
 
# No. of popular Items in the Popularity Model
nPopularItems = 100
 
 
ErrorCheck_Model1 = []
ErrorCheck_Model2 = []
  
# Two models are considered 
# Model = 1 is a popularity based model which checks for the most popular items in the set and recommends
#         that for every user
# Model = 0 is the logistic Matrix Factorization Model
 
 
ValidationData = []
for reg_param in np.arange(0.0, 0.1, 0.005):    
     
    Cumul_count_Model1 = 0.0
    Cumul_precision_Model1 = 0.0
    Cumul_recall_Model1 = 0.0
     
    Cumul_count_Model2 = 0.0
    Cumul_precision_Model2 = 0.0
    Cumul_recall_Model2 = 0.0   
    for i in range(nfolds):
         
        # Prepare the training and test sets for use in the calculations.
        # This entails breaking it into number of guests, items, quantity, etc.
        train_data = gst_item()
        test_data = gst_item()
        train_data.prepare_data(train_dataSets[i])
        test_data.prepare_data(test_dataSets[i])
     
        # Load the Matrix for calculations Collaborative = 0 indicates that we are NOT
        # working with the Item-Item collaborative filtering as that entails enormous
        # computational time.
        collaborative = 0
        Matrix_for_training = train_data.load_matrix(collaborative)
         
        # Logistic Matrix Factorization Model
        Model = LogisticMF(Matrix_for_training, nfeatures, reg_param + 0.005, gamma, niter)
        Model.train_model()
        train_data.get_probabilities(Model.print_probabilities())
        train_data.get_top_n_recommendations_gst(nrec)
         
         
        # Popularity Model
         
        PopularityModel = evaluate(train_data, test_data)
        [count_Model2, precision_Model2, recall_Model2] = PopularityModel.calculatePrecisionRecall(nPopularItems, 1)
         
         
        # Calculate Precision Recall
        precRecall_Model1 = evaluate(train_data, test_data)
        [count_Model1, precision_Model1, recall_Model1] = precRecall_Model1.calculatePrecisionRecall(nrec, popularityModel = 0)
         
        Cumul_count_Model1 += count_Model1
        Cumul_precision_Model1 += precision_Model1
        Cumul_recall_Model1 += recall_Model1
         
        Cumul_count_Model2 += count_Model2
        Cumul_precision_Model2 += precision_Model2
        Cumul_recall_Model2 += recall_Model2
         
         
    ErrorCheck_Model1.append([reg_param, nfeatures, Cumul_count_Model1/nfolds, Cumul_precision_Model1/nfolds, Cumul_recall_Model1/nfolds])
    ErrorCheck_Model2.append([reg_param, nfeatures, Cumul_count_Model2/nfolds, Cumul_precision_Model2/nfolds, Cumul_recall_Model2/nfolds])
 
    ErrorCheckDF_Model1 = pd.DataFrame.from_records(ErrorCheck_Model1, columns = ['RegParam', 'nfeatures', 'count', 'precision', 'recall'])
    ErrorCheckDF_Model2 = pd.DataFrame.from_records(ErrorCheck_Model2, columns = ['RegParam', 'nfeatures', 'count', 'precision', 'recall'])
     
    ValidationData_List = [reg_param+0.005, Cumul_precision_Model1/nfolds, Cumul_recall_Model1/nfolds, Cumul_precision_Model2/nfolds, Cumul_recall_Model2/nfolds]
     
    ValidationData.append(ValidationData_List)
    ValidationDF = pd.DataFrame(ValidationData, columns = ['RegParam', 'precision_M1', 'recall_M1', 'precision_M2', 'recall_M2'])
    ValidationDF.to_csv('ValidationData_Test', ',')

     




