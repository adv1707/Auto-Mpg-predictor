# Auto-Mpg-predictor
My first project on Polynomial Regression
This model basically takes 7 features of a car such as no.of cylinders,displacement,weight etc and predicts what would be the mileage of the car in Miles-per-Gallon.

# About and Approach
This model uses polynomial regression and stochastic gradient descent method to basically determine and predict what could be the mileage of your car .
Approach :- 
          - Import and study the dataset
          - Did some feature engineering like imputing missing values , OneHotEncoding and removing irrelevant column
          - Did some statistics as most of the columns were left skewed so used FunctionTransformers to try to make it normal
          - Scaling the final training data set 
          - Then trying and guessing different powers for polynomial regression , Honestly I tried from 1 to 4 and after power 3 the model started to Overfit hence                 chose 2 as my ideal value
          - Using SDG in the pipeline 
          - Calculating final R2 score , which was 0.7663213614689879 and yeah its good as its my first project without use of any AI fully done by myself
          - As a bonus I also tried to make a CLI using OOPs that is in CLI.py

# Files 
- Main.py :- its the main model file
- CLI.py :- its the code containing CLI that is very simple user enters the details and gets the output
- auto-mpg.csv :- the dataset I used for training my model , obtained from UCI Machine Learning Repository . Link -> https://archive.ics.uci.edu/dataset/9/auto+mpg

# Next Steps 
- I know you might be thinking that I have not used cross-validation as it gives more accurate R2 score , so my answer for that is I haven't learned it yet but surely after a week I will learn it and update the code and also update it here

# Note!! :- This model is trained on the model_year of cars ranging from 1972 to 1995 so it should not be trusted for years before and after it.
