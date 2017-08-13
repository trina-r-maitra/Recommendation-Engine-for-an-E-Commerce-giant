import time
import numpy as np
import pandas as pd
from logMFLearning import *
from gst_item import *
from evaluate import *
from data_preprocessing import *


purchases = pd.read_csv('purchases.csv', sep = ',')
print "length of Original Datasets: ",len(purchases)









nItems = 1
dataForWork = data_preprocessing(purchases)
dataForWork.clean_data()
dataForWork.drop_items_gst_less_than_n(nItems)


train, test = dataForWork.test_train_split(0.8)

train_data = gst_item()
test_data = gst_item()

train_data.prepare_data(train)
test_data.prepare_data(test)

precRecall = evaluate(train_data, test_data)
[count, precision, recall] = precRecall.calculatePrecisionRecall(40,1)
print count, precision, recall












#===============================================================================
# nItems = 1
# dataForWork.drop_items_gst_less_than_n(nItems)
#  
# nfolds = 10
#  
# test_dataSets, train_dataSets = dataForWork.split_data_in_K_Folds(nfolds, ['qty', 'item_i', 'gst_i'])
#  
# nfeatures = 10
# reg_param = 0.1
# gamma = 1
# niter = 20
# nrec = 40
# ErrorCheck = []
# 
# for reg_param in range(0.001, 0.05, 0.001):
#     for nfeatures in range(1, 20, 1): 
#         
#         count = 0.0
#         precision = 0.0
#         recall = 0.0
#         for i in range(nfolds):
#             train_data = gst_item()
#             test_data = gst_item()
#             train_data.prepare_data(train_dataSets[i])
#             test_data.prepare_data(test_dataSets[i])
#             Matrix_for_training = train_data.load_matrix(0)
#             Model = LogisticMF(Matrix_for_training, nfeatures, reg_param, gamma, niter)
#             Model.train_model()
#             # Model.print_vectors()
#             train_data.get_probabilities(Model.print_probabilities())
#             train_data.get_top_n_recommendations_gst(nrec)
#             precRecall = evaluate(train_data, test_data)
#             [count, precision, recall] = precRecall.calculatePrecisionRecall(nrec)
#             count += count
#             precision += precision
#             recall += recall
#           
#         ErrorCheck.append([reg_param, nfeatures, count/nfolds, precision/nfolds, recall/nfolds])
# 
#         ErrorCheckDF = pd.DataFrame.from_records(ErrorCheck, columns = ['RegParam', 'nfeatures', 'count', 'precision', 'recall'])
#         ErrorCheckDF.to_csv("Error.csv")
#         print ErrorCheckDF
#   
#    
# ErrorCheckDF = pd.DataFrame.from_records(ErrorCheck, columns = ['RegParam', 'nFeatures', 'count', 'precision', 'recall'])
# ErrorCheckDF.to_csv("Error.csv")
#===============================================================================






#===============================================================================
# 
# # This is for Item_Item Collaborative Filtering 
# II_CF = gst_item()
# II_CF.prepare_data(dataForWork.data)
# II_CF.load_matrix(1)
# II_CF.item_item_collaborative_filtering()
# 
# 
# 
# 
#===============================================================================





