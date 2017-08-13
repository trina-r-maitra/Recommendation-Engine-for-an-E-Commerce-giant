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
