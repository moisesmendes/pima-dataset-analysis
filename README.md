# pima-dataset-analysis

This is a repository for studying and applying Machine 
Learning (ML) techniques to the pima dataset.

## Dataset description

* Name: Pima Indians Diabetes Database
* Provider: Kaggle
* Link: [PIMA dataset](https://www.kaggle.com/uciml/pima-indians-diabetes-database?select=diabetes.csv) 

#### Context

*Excerpt extracted from Kaggle description.*

This dataset is originally from the National Institute of Diabetes 
and Digestive and Kidney Diseases. The objective of the dataset is 
to diagnostically predict whether or not a patient has diabetes, 
based on certain diagnostic measurements included in the dataset. 
Several constraints were placed on the selection of these instances 
from a larger database. In particular, all patients here are females 
at least 21 years old of Pima Indian heritage.

#### Predictor variables

1. **Pregnancies**: 
Number of times pregnant
2. **Glucose**: 
Plasma glucose concentration a 2 hours in an oral glucose tolerance test
3. **BloodPressure**:
Diastolic blood pressure (mm Hg)
4. **SkinThickness**:
Triceps skin fold thickness (mm)
5. **Insulin**:
2-Hour serum insulin (mu U/ml)
6. **BMI**:
Body mass index (weight in kg/(height in m)^2)
7. **DiabetesPedigreeFunction**:
Diabetes pedigree function
8. **Age**:
Age (years)

#### Target variable

- **Outcome**:
Class variable (0 or 1) 268 of 768 are 1, the others are 0

## Preprocessing

Some preprocessing methods are applied to the original dataset.

First, we split the data into predictor variables `X` and target variable `y`.

Then, we separate data into train and test according to a specified ratio.
In this step, we can also mantain the classes proportion according to the 
original data distribution for imbalanced datasets. 

Finally, we scale the predictor variables using some scaler (default
option is `StandardScaler` from scikit-learn). 