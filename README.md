# Car Price Prediction (Machine Learning Regression)

This project predicts the selling price of a used car based on its features such as brand, year, fuel type, transmission, and kilometers driven.

The goal of this project is to apply a supervised machine learning regression model to estimate market price from historical data.



## Problem Statement

Used car prices vary depending on multiple factors like age, fuel type, ownership history, and usage.  
Manually estimating a fair selling price is difficult and subjective.

This model learns from previous car sale records and predicts an approximate selling price for a new car entry.



## Features Used

- Car Name / Company
- Year of Purchase
- Present Price
- Kilometers Driven
- Fuel Type (Petrol/Diesel/CNG)
- Seller Type (Dealer/Individual)
- Transmission (Manual/Automatic)
- Number of Previous Owners



## Machine Learning Approach

1. Data cleaning and preprocessing
2. Handling categorical variables (encoding)
3. Feature selection
4. Train-test split
5. Model training using Linear Regression
6. Model evaluation

The trained model learns the relationship between vehicle attributes and market value.



## Project Files

- data_preprocessing.py → data cleaning and encoding
- model_training.py → trains the regression model
- prediction.py → predicts price for new input
- car_data.csv → dataset
- model.pkl → saved trained model
- app.py (if you have Streamlit) → web interface




