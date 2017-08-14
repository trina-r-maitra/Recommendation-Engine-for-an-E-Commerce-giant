from gst_item import *
        
        
class evaluate(gst_item):
    def __init__(self, trainingSet, validationSet):
        self.common_gsts = []
        self.trainingSet = trainingSet
        self.validationSet = validationSet
    
    #
    def get_index_of_gst_in_list(self, gst_item_list, common_gst):
        #print "list", gst_item_list
        for i in range(len(gst_item_list)):
            if (gst_item_list[i][0] == common_gst):
                return i
        
    def calculatePrecisionRecall(self, nTop, popularityModel):
        
        
        # Get the list of Common Guests in both the training and validation Sets.
        self.common_gsts = set.intersection(*[set(self.trainingSet.gst_lookup_table), set(self.validationSet.gst_lookup_table)])  
        
        count = 0
        Metric_List = []
        precision = 0.0
        recall = 0.0
        
        # A np array that contains user: Item1 Item2 Item3 ... ItemN
        test_data_items_bought = self.validationSet.gst_item_matrix() 
        
        # Iterate over common guests
        for common_gst in list(self.common_gsts):
            
            # Get the index of the gst in the training set
            index_train = self.get_index_of_gst_in_list(self.trainingSet.top_n_recommendations_gsts, common_gst)
            
            # Get the index of the gst in the test set
            index_test  = self.get_index_of_gst_in_list(test_data_items_bought, common_gst)
            
            
            # If popularity based model is used then the number of data points recommended is nTop
            # which remains constant = TP + FP
            if (popularityModel):
                numItemsRecommended = nTop
            else:
                numItemsRecommended = self.trainingSet.nRec
            
            
            if (popularityModel):
                items_in_top_n = self.trainingSet.get_top_n_popular_items(numItemsRecommended)
                
            else:
                #print "index Train", index_train, "test", index_test
                items_in_top_n = self.trainingSet.top_n_recommendations_gsts[index_train][1:numItemsRecommended + 1]
            
            # Select the items in the Test data
            items_in_test = test_data_items_bought[index_test][1:]
            
            # No. of Common Items in the test and training data
            common_items = len(set.intersection(*[set(items_in_top_n), set(items_in_test)]))

            # List the common items
            listCommon = [common_gst, common_items, common_items/numItemsRecommended, common_items/len(items_in_test)]
            
            # Add this to a Metric list
            Metric_List.append(listCommon)
            
            #print "nRec: ", self.trainingSet.nRec
            
            # Cumulative Precision in percentage upto User i 
            precision += 100*common_items/numItemsRecommended
            
            # Cumulative Recall in percentage upto User i
            recall += 100*common_items/len(items_in_test)
            #print "prec: ", precision, "rec:", recall
            
            
            # Total number of common items
            count += common_items
            #tot_count.append(count)
        
           
        #print len(Metric_List)
        
        # Returns the Count, Avg(Precision), Avg(Recall)
        return count, precision/len(Metric_List), recall/len(Metric_List)#/len(Metric_List), recall*100/len(Metric_List)
            