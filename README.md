
PROBLEM STATEMENT:

    A retail company “ABC Private Limited” wants to understand the customer purchase 
    behaviour (specifically, purchase amount) against various products of different
    categories. They have shared purchase summary of various customers for selected 
    high volume products from last month.
    The data set also contains customer demographics (age, gender, marital status,/
    city_type, stay_in_current_city), product details (product_id and product category)
    and Total purchase_amount from last month.
    Now, they want to build a model to predict the purchase amount of customer 
    against various products which will help them to create personalized offer 
    for customers against different products.
	
Source of the project problem: https://datahack.analyticsvidhya.com/contest/black-friday/

USED AVERAGE OF 3 XGBOOST MODELS FOR PREDICTION.

	model 1 (depth:8 trees:1450 iterations:20)

	model 2 (depth:12 trees:800 iterations:20)

	model 3 (depth:6 trees:3000 iterations:35)

VALIDATION RMSE ON VALIDATE DATA (from train.csv) = 2463.16 (TRAIN:VALIDATE = 80:20)

VALIDATION RMSE ON TEST DATA (from test.csv) = 2514.66 
