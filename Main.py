import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import FunctionTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import cross_val_score


#Importing data
data = pd.read_csv(r"auto-mpg.csv")

#Filling missing values
imputer = SimpleImputer(strategy="mean")
y = imputer.fit_transform(data["horsepower"].reset_index().set_index("index")).ravel()
y = pd.Series(y)
data["horsepower"]=y

#Dropping irrelevant column
data=data.drop(columns=["car_name"])

#Creating training and test data sets with mpg as target
X=data.drop(columns=["mpg"])
Y = data["mpg"]
x_train , x_test , y_train ,y_test = train_test_split(X,Y,test_size=0.2,shuffle=True,random_state=17)

#Creating functions 
logfunc = FunctionTransformer(func=np.log)
sqrtfunc = FunctionTransformer(func=np.sqrt)
scaler = StandardScaler()

#Creating log transformer
logtransformer = ColumnTransformer(transformers=[
    ("logtransformers for other columns",logfunc,[1,2,3])
],remainder="passthrough")

onehottransformer  = ColumnTransformer(transformers=[
    ("Onehotencoding",OneHotEncoder(drop="first"),[6])
],remainder="passthrough")


#Creating scaler transformer
scalertransformer = ColumnTransformer(transformers=[
    ("scaler",scaler,[0,1,2,3,4,5])
])

#Creating side pipeline for cylinder transformer
transformer1 = ColumnTransformer(transformers=[
    ("sqrt",sqrtfunc,[0]),
],remainder="passthrough")

transformer2 = ColumnTransformer(transformers=[
    ("log",logfunc,[0]),
],remainder="passthrough")

sidepipecylinder = Pipeline([
    ("cylinder transformer1",transformer1),
    ("cylinder transformer2",transformer2)
])


#Creating polymomial features
polynomial = PolynomialFeatures(degree=(1,2))

#Creating model
sgd = SGDRegressor()


#Creating mainpipeline 
mainpipe = Pipeline([
    ("cylinderpipeline",sidepipecylinder),
    ("logtransformer",logtransformer),
    ("onehottransformer",onehottransformer),
    ("scaling",scalertransformer),
    ("Raising power",polynomial),
    ("model",sgd)
])

mainpipe.fit(x_train,y_train)
y_pred = mainpipe.predict(x_test)
cross = cross_val_score(estimator=mainpipe,X=x_train,y=y_train,cv=10).mean()
