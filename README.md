### Business Problem

Build a Recommendation System providing top 100 product recommendations given a guest id.

### Work Flow 

1. Read purchases.csv file and performed data pre-processing 

      - drop rows having missing values.
      - drop rows having quantity purchased <0. I see there are records with non integer quantities and choose not to drop them 
        since there might be products sold in pounds/etc.
      - deal with special characters in guest and item ids and choose to drop the rows containing them.
      - check whether a guest purchases an item multiple times on different days. Since a guest purchases an item just once, 
        there is no use of purchase date column and I drop it.
      - did not consider items.csv for this exercise.
      
      The clean data has 1000 guests and 17403 items.
      
2. Exploratory Analysis

      - On average a guest purchased 38 items. Median # of items purchased is 28.
      - 81% of quantity purchased is 1.
      - A guest 1904015 has bought 281 items which can be treated as an outlier. In this project, I have not treated it as one.
      - Drop items from the dataset which has been purchased maximum 3 times to make User-Item matrix dense. 
      - 45 guests have been dropped who have on average purchased 6 times. Median # of items purchased is 5.
      
      I obtain 955 guests and 2402 items. This is the Model Data.
      
3. Model Approach

      - 2 model approaches are chosen : popularity based and Logistic Matrix Factorization for implicit feedback data. 
      - 81% of the quantity purchased is 1. Hence, for simplicity and for this exercise I have used quantity as an indicator whether an 
        item has been purchased by a guest or not. I consider Logistic Matrix Factorization since the data is about item purchase(binary-1 
        for yes,0 for no) made by guest. This purchase behaviour is observed over time and deemed as implicit feedback. Logistic MF 
        belongs to the family of latent factor models where we attempt to understand the relationship between users and items using 
        latent/hidden factors. The top 100 personalized item recommendations for each guest is generated according to the purchase 
        probability given by the logistic function. 
      - Baseline Model is the popularity based model where I recommend top 100 most popular items (purchased maximum times) to each user. 
        All users have the same item recommendations. 

4. Model Evaluation Approach

      -  I have mainly used Recall to evaluate the above two models since there is a need to recommed all items that a guest purchases. 
 
5. Model Building & Validation Approach   

      - For Logistic Matrix Factorization model, using K fold cross validation I choose the L2 regularization parameter and number of 
        latent features based on Average Recall.
      - Popularity based model has better average recall than the Logistic Matrix Factorization model and hence we choose it for making 
        recommendations





### Potential Improvements

1. Further exploration of making User-Item matrix dense by dropping items which have been purchased <=n times. This is in turn drops  guests, hence have a separate recommendation technique for them. 

2. Use magnitude of quantity purchased rather than treat it as an indicator of purchase as we have done for this exercise. 

3. Explore Item similarity Collaborative Filtering technique after making the User-Item Matrix dense.

4. Group guests based on their purchase behaviour along with demographic/ financial information and build separate  
   recommendation models for them. 
   
5. Tackle cold start problem.
   
### References

1. https://stanford.edu/~rezab/nips2014workshop/submits/logmat.pdf

2. https://github.com/MrChrisJohnson/implicit-mf

3. http://yifanhu.net/PUB/cf.pdf


### General Comments about the code

The code has been developed using Anaconda build of Python 2. Numpy, Pandas and Matplotlib libraries have been used as necessary. The operator needs to input the guest_id and the program generates the top 100 recommendations. 

From the RegularizationVsRecall.png plot it can be observed that the recommendations based on Popularity is superior to the Logistic Matrix Factorization Model over wide range of number of latent features and regularization parameter. For the Logistic Matrix Factorization Model only, it can be observed that when the number of latent features is set to 1, the model performs the best (higher recall%). With number of latent features set to 1 the model is relatively better at a regularization parameter of 0.045, albeit the variation in the latter is almost flat. 



main.py - This is the main file. Put the Guest_ID and the top 100 recommendations are generated using the logMF and the Popularity models.

logMFLearning.py - This file contains the Logistic Matrix Factorization class. Most of the code has been inherited from ref. 1

gst_item.py - This file contains the gst_item class with all the methods for processing the data.

evaluate.py - This file contains the evaluate class with the methods for evaluating the models based on precision and recall

data_preprocessing.py - This file contains the methods for cleaning the data based on exploratory analysis, segregating the data for k-fold cross validation, etc. k-Fold cross-validation could have been used using the scikit-learn library as well.

Validation.py - This is a script for evaluating the performance of the models using k-fold cross validation. 

Parameterization.py - This is a script that generates the RecallVsRegularization.png plot. From the plot it can be observed that the popularity model is superior in terms of Recall over a wide range of number of features and regularization parameters. 


        
        
        
        
      
      
