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
      
      The clean data has 1000 guests and 17403 items.
      
2. Exploratory Analysis

      - On average a guest purchased 38 items. Median # of items purchased is 28.
      - A guest 1904015 has bought 281 items which can be treated as an outlier. In this project, I have not treated it as one.
      - Drop items from the dataset which has been purchased maximum 3 times to make User-Item matrix dense. 
      - 45 guests have been dropped who have on average purchased 6 times. Median # of items purchased is 5.
      
      I obtain 955 guests and 2402 items. This is the Model Data.
      
3. Model Approach

      - 2 model approaches are chosen : popularity based and Logistic Matrix Factorization for implicit feedback data. 
      - I consider Logistic Matrix Factorization since the data is about item purchase(binary-1 for yes,0 for no) made by guest. This 
        purchase behaviour is observed over time and deemed as implicit feedback. Logistic MF belongs to the family of latent factor model 
        where we attempt to understand the relationship between users and items using latent/hidden factors. The top 100 personalized item 
        recommendations for each guest is generated according to the purchase probability given by the logistic function. 
      - Baseline Model is the popularity based model where I recommend top 100 most popular items (purchased maximum times) to each user. 
        All users have the same item recommendations. 

4. Model Evaluation Approach

      -  I have mainly used Recall to evaluate the above two models since there is a need to recommed all items that a guest purchases. 
 
5. Model Building & Validation Approach   


        
        
        
        
      
      
