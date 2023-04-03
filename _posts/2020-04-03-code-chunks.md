---
layout: post
title: You're up and running!
---


# Code chunks

## Contents

- [Reminder](#reminder)
- [Loading data](#loading_data)
 - [Loading a CSV data file](#loading_a_CSV_data_file)
- [Data cleaning](#data_cleaning)
 - [Duplicate rows and columns](#duplicate_rows_and_columns)
 - [Missing values](#missing_values)
 - [DateTime](#datetime)
 - [Splitting strings](#splitting_strings)
- [Dataframe manipulation](#dataframe_manipulation)
 - [Create a dataframe](#create_a_dataframe)
 - [Show a dataframe](#show_a_datframe)
 - [Conditional selection of a column](#conditional_selection_of_a_column)
 - [Turning a series into a dataframe](#turning_a_series_into_a_dataframe)
 - [Renaming columns](#renaming_columns)
 - [Dropping and adding columns and rows](#dropping_and_adding_columns_and_rows)
 - [Concatenation](#concatenation)
 - [Groupby](#groupby)
- [Exploratory data analysis (EDA](#Exploratory_data_analysis)
 - [Dataframe outline](#dataframe_outline)
 - [Numerical and non-numerical columns](#numerical_and_non_numerical_columns)
 - [Dataframe calculations](#dataframe_calculations)
 - [Dummy variables](#dummy_variables)
 - [Column sampling](#column_sampling)
 - [Basic EDA function](#basic_eda_function)
- [Statistical analysis and modelling](#statistical_analysis_and_modelling)
  - [Covariance matrix](#covariance_matrix)
  - [Correlation matrix](#correlation_matrix)
  - [Z-score](#z_score)
  - [One-sample t-test](#one_sample_t_test)
  - [Two-sample unpaired t-test](#two_sample_unpaired_t_test)
  - [Paired two-sample t-test](#paired_two_sample_t_test)
  - [Multi-collinearity heatmap](#multi_collinearity_heatmap)
 - [Linear regression](#linear_regression)
  - [Split X and y](#split_x_and_y)
  - [Covariance](#Covariance)
  - [Correlation coefficient](#correlation_coefficient)
  - [Plotting linear relationship](#plotting_linear_relationship)
  
  
- [Machine learning](#machine_learning)
 - [Import library datasets](#importing_library_datasets)
 - [Generate synthetic data](#generate_synthetic_data)
 - [Generate synthetic data](#generating_synthetic_data)
 - [Add noise to data](#add_noise_to_data)
 - [Describe the data](#describe_the_data)
 - [Choose independent and target variables](#choose_independent_and_target_variables)
 - [Scale and transform a dataset](#scale_and_transform_a_dataset)
 - [Import model](#import_model)
 - [Instantiate a model](#instantiate_a_model)
   - [Instantiate a linear regression model](#instantiate_a_linear_regression_model)
   - [Instantiate a logistic regression model](#instantiate_a_logistic_regression_model)
   - [Instantiate a KNN model](#instantiate_a_knn_model)
   - [Instantiate a support vector machine (SVC) regression model](#instantiate_an_svc_regression_model)
   - [Instantiate a linear support vector machine (SVC) model](#instantiate_a_linear_svc_model)
   - [Instantiate a naive bayes model](#instantiate_a_naive_bayes_model)
 - [Splitting testing and training datasets](#splitting_the_testing_and_training_datasets)
 - [Fitting a model](#fitting_a_model)
   - [Fitting a linear regression model](#fitting_a_linear_regression_model)
   - [Fitting a logistic regression model](#fitting_a_logistic_regression_model)
   - [Fitting a KNN model](#fitting_a_KNN_model)
   - [Fitting a linesr SVM model](#fitting_a_linear_svm_model)
   - [Fitting a naive bayes model](#fitting_a_naive_bayes_model)
 - [Make predictions](#make_predictions)
 - [Score predictions](#score_predictions)
 - [Score model](#score_model)
 - [Test a model with main parameter variation](#test_a_model_with_main_parameter_variation)
 - [Principal component analysis (PCA)](#principal_component_analysis)
- [Visualisation](#visualisation)
 - [Plot the performance of a model with main parameter variation](#plot_the_performance_of_a_model_with_main_parameter_variation) 
 - [Plot the boundary of a logistic regression model](#plot_the_performance_of_a_model_with_main_parameter_variation)
- [Miscellanea](#miscellanea)

## Reminder
<a id='reminder'></a>


```python
# Most operations in Python are applied in the following ways
my_df.some_method()

my_df['col_name'].some_method()
```

## Loading data
<a id='loading_data'></a>

#### Loading a CSV data file
<a id='loading_a_CSV_data_file'></a>


```python
# Loads a CSV data file to a variable
mydata = pd.read_csv('my_data_file.csv')
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    /var/folders/3w/7b2j0crd0hvc3bb28cmq6j_40000gn/T/ipykernel_1818/1110480082.py in <module>
          1 # Loads a CSV data file to a variable
    ----> 2 mydata = pd.read_csv('my_data_file.csv')
    

    NameError: name 'pd' is not defined


#### Prevent Pandas from creating an 'Unnamed: 0' column
<a id='loading_a_CSV_data_file'></a>


```python
df = pd.read_csv('data/faithful.csv', index_col=0)
```

### Data cleaning
<a id='data_cleaning'></a>

#### Duplicate rows and columns
<a id='duplicate_rows_and_columns'></a>


```python
# Gives duplicated rows in the dataframe
mydata.duplicated()
```


```python
# Gives duplicated columns in the dataframe, using a transpose
mydata.T.duplicated().sum()
```


```python
# Gives the sum of the duplicated rows in the dataframe
mydata.duplicated().sum()
```


```python
# Locates the duplicated rows of in the dataframe
mydata.loc[raw_data.duplicated() , :]
```


```python
# Drops the duplicates rows in the dataframe
mydata.drop_duplicates()
```

#### Missing values
<a id='missing_values'></a>


```python
# Gives the total of null values
mydata.isna().sum()
```


```python
# Gives missing values by the rows in the dataframe
mydata.isna().sum(axis = 0)
```


```python
# Gives missing values by the columns in the dataframe
clean_data.isna().sum(axis = 1)
```


```python
# Gives the number of missing entries as a percentage of the dataframe, using the result from 'shape'
my.isna().sum(axis = 0) / mydata.shape[0] * 100
```


```python
# Overwrites the original data with the median values for NAs 
mydata['column'] = mydata['column'].fillna(mydata['column'].median())
```

#### DateTime
<a id='datetime'></a>


```python
# Converts a date object to the DateTime format
mydata['date_column'] = pd.to_datetime(mydata['date_column'], format='%Y%m%d')

```

#### Splitting strings
<a id='splitting_strings'></a>


```python
#Uses the split function - applied to a hyphen - to divide two pieces of information, putting them into two columns
mydata['string_column'].str.split(" - ", expand = True)
```

### Dataframe manipulation
<a id='dataframe_manipulation'></a>

#### Create a dataframe
<a id='create_a_dataframe'></a>


```python
columns = ['size','flies','eggs','aquatic','is_bird']

bat = [10,1,0,0,0]
rat = [10,0,0,0,0]
flying_lizard = [5,1,1,0,1]
penguin = [20,0,1,1,1]
robin = [10,1,1,0,1]
pigeon = [15,1,1,0,1]
emu = [150,0,1,0,1]
fish = [20,0,1,1,0]
wasp = [1,1,1,0,0]
frog = [5,0,0,1,0]
duck = [20,1,1,1,1]
heron = [50,1,1,1,1]
pterodactyl = [200,1,1,0,0]

df = pd.DataFrame([bat,rat,flying_lizard,
                   penguin,robin,pigeon,emu,fish,
                   wasp,frog,duck,heron,pterodactyl], columns = columns)
```

#### Show a dataframe
<a id='show_a_datframe'></a>


```python
display(df)
```

#### Create a column
<a id='conditional_selection_of_a_column'></a>


```python
retail_df['margin_gt_15'] = (retail_df['profit']/retail_df['sales'] > 0.15).astype(int)
```

#### Column types
<a id='conditional_selection_of_a_column'></a>


```python
# Checks change to the 'Reviewer_Score' column type
hotel_reviews['Reviewer_Score'].dtypes
```

#### Conditional selection of a column
<a id='conditional_selection_of_a_column'></a>


```python
# Gives the conditional selection of a column 
mydata[mydata['column'] == "column_name"]
```

#### Turning a series into a dataframe
<a id='turning_a_series_into_a_dataframe'></a>


```python
# Makes a series into a dataframe
mydata_groups.to_frame()
```

#### Changing the datatype of a column
<a id='renaming_columns'></a>


```python
df['column_name'].astype(int)
```

#### Renaming columns
<a id='renaming_columns'></a>


```python
# Renames columns using a dictionary and index locations
mydataframe = mydataframe.rename(columns = {0: 'occupation', 1: 'location'})
```

#### Dropping and adding columns and rows
<a id='dropping_and_adding_columns_and_rows'></a>


```python
# Drops a column
mydata.drop(columns = 'columnn name')
```

#### Concatenation
<a id='concatenation'></a>


```python
# Concatenates two dataframes, on the columns
pd.concat([mydata, another_dataframe], axis = 1)
```

#### Groupby
<a id='groupby'></a>


```python
anscombe.groupby('Dataset').agg(['mean', 'median', 'std', 'var'])
```


```python
#Gropups by the registered column, using the last contact duration, and compares the mean to the median
clean_data.groupby('registered')['last_contact_duration'].agg(['mean', 'median'])
```

#### Sampling
<a id='Sampling'></a>


```python
# Creates a ten percent sample
hotel_reviews.sample(frac=0.1, random_state=42)
```

#### Timestamp
<a id='timestamp'></a>

The `pd.Timestamp` objects allow us to extract various datetime features or do arithmetic with dates:


```python
print(f"first_day year: {first_day.year}, first_day month:{first_day.month} and first_day day: {first_day.day}")
```

    first_day year: 1979, first_day month:1 and first_day day: 1


The result of subtracting two dates is a `pd.Timedelta` object:


```python
last_day - first_day
```




    Timedelta('8491 days 00:00:00')



We can add durations using `pd.DateOffset`:


```python
first_day + pd.DateOffset(years=1, months=3)
```




    Timestamp('1980-04-01 00:00:00')




```python
The result of subtracting two dates is a `pd.Timedelta` object:

last_day - first_day

We can add durations using `pd.DateOffset`:

first_day + pd.DateOffset(years=1, months=3)
```

#### More DateTime
<a id='timestamp'></a>


```python
full_range = pd.date_range(start=first_day, end=last_day, freq="D")
full_range
```




    DatetimeIndex(['1979-01-01', '1979-01-02', '1979-01-03', '1979-01-04',
                   '1979-01-05', '1979-01-06', '1979-01-07', '1979-01-08',
                   '1979-01-09', '1979-01-10',
                   ...
                   '2002-03-23', '2002-03-24', '2002-03-25', '2002-03-26',
                   '2002-03-27', '2002-03-28', '2002-03-29', '2002-03-30',
                   '2002-03-31', '2002-04-01'],
                  dtype='datetime64[ns]', length=8492, freq='D')




```python
full_range.difference(air_traffic.index)
```




    DatetimeIndex(['1979-05-13', '1980-01-19', '1980-02-18', '1980-03-03',
                   '1980-03-08', '1980-04-21', '1980-06-18', '1980-08-31',
                   '1980-11-29', '1981-03-07',
                   ...
                   '2001-03-22', '2001-03-26', '2001-04-09', '2001-06-23',
                   '2001-07-01', '2001-07-14', '2001-09-26', '2001-12-08',
                   '2002-03-03', '2002-03-27'],
                  dtype='datetime64[ns]', length=172, freq=None)



Now, to add the missing dates, we can either use the `reindex` method with our full date range, or simply call the `asfreq` method:


```python
air_traffic_clean = air_traffic.reindex(full_range)

# the same without defining the range by hand

# air_traffic_clean = air_traffic.asfreq("D")
```


```python
# the "MS" option specifies Monthly frequency by Start day
air_traffic_monthly = air_traffic_clean.resample("MS").sum()

air_traffic_monthly.head()
```

#### Seasonal plot
<a id='seasonal plot'></a>


```python
from statsmodels.graphics.tsaplots import month_plot

plt.figure(figsize=(15, 5))

# create the seasonal plot
month_plot(air_traffic_monthly["Revenue Passenger Miles"], ax=plt.gca())

plt.title("Seasonal Revenue Passenger Miles per Month")
sns.despine()
plt.show()
```

 monthly average


```python
# monthly average
monthly_mean = air_traffic_monthly.groupby(air_traffic_monthly.index.month_name()).mean()

# relative deviation from the overall mean
monthly_mean_diff = (monthly_mean - monthly_mean.mean())/monthly_mean

# month names in right order
month_names = pd.date_range(start='2000-01', freq='M', periods=12).month_name()

# reorder columns to follow the month order
monthly_mean_diff = monthly_mean_diff.loc[month_names, ]

monthly_mean_diff.T
```


```python
# Gives the rolling, weekly mean from 2020-01-27 onwards
india_deaths['Weekly mean'] = india_deaths['India deaths daily']['2020-01-28':].rolling(7).mean()
```

### Exploratory data analysis (EDA)
<a id='Exploratory_data_analysis'></a>

#### Dataframe outline
<a id='dataframe_outline'></a>


```python
# Gives the top five rows of the dataframe
mydata.head()
```


```python
# Gives the bottom five rows of the dataframe
mydata.tail()
```


```python
# Gives a random sample of the dataframe
mydata.sample()
```


```python
# Gives the number of rows and columns in the dataframe
mydata.shape
```


```python
# Gives the data types in the dataframe
mydata.info()
```


```python
# Gives basic stats on the dataframe
mydata.describe()
```

#### Numerical and non-numerical columns
<a id='numerical_and_non_numerical_columns'></a>


```python
# Gives a list of the float and int columns in the dataframe
mydata.select_dtypes(['float', 'int']).columns
```


```python
# Gives a list of the numerical columns in the dataframe
num_col_list = df_cleaned.select_dtypes('number')
```


```python
# Creates a subset of the dataframe with only numerical columns
df_cleaned_num = mydata[num_col_list].copy()
```


```python
# Gives a list of the object columns in the dataset
mydata.select_dtypes(['object']).columns
```

#### Dataframe calculations
<a id='dataframe_calculations'></a>


```python
# Divides one group by, by another
mydata.groupby(['column1', 'column2'])['column3'].count() / clean_data.groupby('column1')['column2'].count() 
```


```python
# Create profit margin column
retail_df['profit_margin'] = retail_df['profit'] / retail_df['sales']
```


```python

```

#### Dummy variables
<a id='dummy_variables'></a>


```python
# Gives dummy variable columns
pd.get_dummies(mydata['column_name'], prefix = 'new_column_name')
```

#### Column sampling
<a id='column_sampling'></a>


```python
# Gives a sample of a column
mydata['column'].sample(10)
```

#### Basic EDA function
<a id='basic_eda_function'></a>


```python
# Basic EDA function
def run_EDA_checks(dataset):
        """
    Add function description

    Args:
        Add the arguments that the function takes

    """
    #Prints the row labels
    print('ROW LABELS')
    print('')
    print(dataset.index)
    print('')
    #Prints the column names
    print('COLUMN NAMES')
    print('')
    print(dataset.columns)
    print('')
    #Prints the data types of the columns
    print('COLUMN DATA TYPES')
    print('')
    print(dataset.dtypes)
    print('')
    #Prints the number of rows and columns
    print('NUMBER OF ROWS AND COLUMNS')
    print('')
    print(dataset.shape)
    print('')
    #Prints a concise summary of the dataset
    print('CONCISE SUMMARY')
    print('')
    print(dataset.info())
    print('')
    #Prints a concise summary of the dataset
    print('DESCRIPTION OF ALL COLUMNS')
    print('')
    print(dataset.describe(include = 'all'))
    print('')
    #Prints the number of null values in the dataset
    print('NUMBER OF NULL VALUES BY COLUMN')
    print('')
    print(dataset.describe(include = 'all'))
    print('')
    print(dataset.isnull().sum())
```


```python
# Options for expanding the basic EDA function
def basic_eda(df, df_name):
    """
    getting some basic information about each dataframe
    shape of dataframe i.e. number of rows and columns
    total number of rows with null values
    total number of duplicates
    data types of columns

    Args:
        df (dataframe): dataframe containing the data for analysis
        df_name (string): name of the dataframe
    """
    print(df_name.upper())
    print()
    print(f"Rows: {df.shape[0]} \t Columns: {df.shape[1]}")
    print()
    
    print(f"Total null rows: {df.isnull().sum().sum()}")
    print(f"Percentage null rows: {round(df.isnull().sum().sum() / df.shape[0] * 100, 2)}%")
    print()
    
    print(f"Total duplicate rows: {df[df.duplicated(keep=False)].shape[0]}")
    print(f"Percentage dupe rows: {round(df[df.duplicated(keep=False)].shape[0] / df.shape[0] * 100, 2)}%")
    print()
    
    print(df.dtypes)
    print("-----\n")
```

## Statistical analysis and modelling
<a id='statistical_analysis_and_modelling'></a>

#### Covariance matrix
<a id='covariance_matrix'></a>


```python
#Covariance matrix 
ansI = anscombe.loc[anscombe['Dataset'] == 'I', ['X', 'Y']]
ansI.cov()
```

#### Correlation matrix
<a id='correlation_matrix'></a>


```python
# Correlation matrix
diamonds = pd.read_csv('diamonds.csv', index_col=0)
diamonds.corr()
```

#### Z-score
<a id='z_score'></a>


```python
# Z score

height_of_interest = 200 
mean = 170
sdev = 20

zscore = (height_of_interest - mean)/sdev
print(zscore)

# By default, `norm` assumes a standard normal distribution
norm.cdf(zscore)
```

#### One-sample t-test
<a id='one_sample_t_test'></a>


```python
one_sample_test = stats.ttest_1samp(store1, 14.5)
print(one_sample_test)
```

#### Two-sample unpaired t-test
<a id='two_sample_unpaired_t_test'></a>


```python
# we can use the built in function: ttest_ind
two_sample_test = stats.ttest_ind(store_suburbs, store_downtown) 

print(two_sample_test)
```

#### Paired two-sample t-test
<a id='paired_two_sample_t_test'></a>


```python
stats.ttest_rel(store_suburbs, store_downtown)
```

#### Multi-collinearity heatmap
<a id='multi_collinearity_heatmap'></a>


```python
# Heatmap
corr_df = X.corr()

# TRIANGLE MASK
mask = np.triu(corr_df)
# heatmap
plt.figure(figsize = (20, 20))
sns.heatmap(corr_df.round(2), annot = True, vmax = 1, vmin = -1, center = 0, cmap = 'Spectral', mask = mask)
plt.show()
```

### Linear regression
<a id='linear_regression'></a>

#### Split X and y
<a id='split_x_and_y'></a>


```python
X = kids['Age'] # independent
y = kids['Weight'] # dependent
```

#### Covariance
<a id='Covariance'></a>


```python
# Covariance
np.cov(X, y)
```

#### Correlation coefficient
<a id='correlation_coefficient'></a>


```python
# Correlation coefficient
np.corrcoef(X,y)
```


```python
# Pearson correlation coefficient
stats.pearsonr(apples, pears)
```

#### Plotting linear relationship
<a id='plotting_linear_relationship'></a>


```python
# Plotting linear relationship
corr = np.corrcoef(X,y)[0,1]

tval = corr * np.sqrt((len(X)-2)/(1- corr **2))
print(tval)

p = stats.t.sf(tval, len(X)-2)*2 #two tailed!
print(p)

stats.pearsonr(X,y)
```

### 6) Check for multi-colinearity, perhaps using a Carr heatmap

### 7) Add a constant to X


```python
X_withconstant = sm.add_constant(X) #we have to add in our intercept manually!
```


```python
#Finds the constant from the x variable, and adds it
age_withconstant = sm.add_constant(age)
age_withconstant
```

### 8) Transformations and scale data


```python
beta_1 = np.sum((X - np.mean(X))*(y - np.mean(y)))/np.sum((X - np.mean(X))**2)
beta_1
```


```python
beta_0 = np.mean(y) - beta_1 * np.mean(X)
beta_0
```


```python
my_beta_1 = np.sum((systolic_blood_pressure - np.mean(systolic_blood_pressure))*(age - np.mean(age)))/np.sum((systolic_blood_pressure - np.mean(systolic_blood_pressure))**2)
my_beta_1
```


```python
my_beta_0 = np.mean(age) - my_beta_1 * np.mean(systolic_blood_pressure)
my_beta_0
```

Do these values make sense? Let's plot our data again, but this time add in the line of best fit. 


```python
plt.figure()
plt.scatter(X,y)

xvals = np.arange(0,9,0.01)
yvals = beta_0 + beta_1*xvals 

plt.plot(xvals, yvals, c = 'black');
plt.xlabel('Age')
plt.ylabel('Weight')
plt.show();
```


```python
plt.figure()
plt.scatter(age, systolic_blood_pressure)

xvals = np.arange(0)
yvals = my_beta_0 + my_beta_1*xvals 

plt.plot(xvals, yvals, c = 'black')
plt.xlabel('Age')
plt.ylabel('Systolic blood pressure')
plt.show()
```

### 9) Instantiate model


```python
# 1. Instantiate Model
myregression = sm.OLS(y, X_withconstant)
```

### 10) Fit model


```python
# 2. Fit Model (this returns a seperate object with the parameters)
myregression_results = myregression.fit()
```

### 11) Summarise model


```python
# Looking at the summary
myregression_results.summary()
```

### 12) Check the predictive power of R2

There is a lot of information summarized in one table. The values that we can recognize at this point are the model parameters (coefficients). The slope is the value beside x1 (3.0054) and the intercept is the value beside const (2.6198).


```python
sns.lmplot('Age', 'Weight', data=kids);
```

### 13) Check the distribution of the residuals, looking at normality by perhaps using a histogram, a Shapiro test and a Q-Q plot; and looking at homoscedasticity with residuals as fitted

### 14) Check the p-values of the variables and hence their significance


```python
#Checks the B0 and B1 values
my_example_regression.params
```


```python
plt.scatter(age, systolic_blood_pressure, label = 'data')

x_val = np.arange(18,70,0.01) #This gives the steps in the x values, starting at 18 and going to 70
y_val = my_example_regression.predict(sm.add_constant(x_val))
plt.plot(x_val, y_val, color = 'black', label = 'model', marker = 'o')

plt.xlabel('Age')
plt.xlabel('Blood pressure')
plt.title('Age and blood pressure')
plt.legend()
```

### 15) Get new input


```python
new_x = 5
predicted_weight = beta_0 + new_x * beta_1
print(predicted_weight)
```


```python
new_kids = np.array([3,5.5,7,9,4.1])
new_kids = sm.add_constant(new_kids)
```

### 16) Make predictions 


```python
# Predictions
myregression_results.predict(new_kids)
```


```python

```


```python

```


```python

```


```python

```

#### Multiple Linear Regression


```python
drinking = pd.read_csv('drinking.csv')
drinking.head()
```


```python
# Rememeber we can get an array of the column names 
drinking.columns
```


```python
X = drinking[drinking.columns[:-1]]
y = drinking['Cirrhosis_death_rate']
# OR 
#y = drinking[drinking.columns[-1]]
```


```python
# using a loop to make 4 plots 
for col in X.columns: 
    plt.figure()
    plt.scatter(X[col], y)
    plt.ylabel('Cirrhosis Death Rate')
    plt.xlabel(col)
    plt.show();
```


```python
X_withconstant = sm.add_constant(X)
```


```python
sns.heatmap(X.corr())
```


```python
# 1. Instantiate model
lm_drinking = sm.OLS(y,X_withconstant)

# 2. Fit model
lm_drinking_results = lm_drinking.fit()

lm_drinking_results.summary()
```

### Linear Model Diagnostics


```python
model_resids = lm_drinking_results.resid # this is where the residuals are stored

model_fittedvals = lm_drinking_results.fittedvalues # this is where the fitted values are stored
```


```python
plt.figure()
plt.hist(model_resids, bins = 12)
plt.show()
```


```python
plt.figure()
plt.scatter(model_fittedvals, model_resids)
plt.xlabel('Fitted Values')
plt.ylabel('Residuals')
plt.show()
```


```python

```

# Logistic regression

### 1) Load data


```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline

import statsmodels.api as sm
```


```python
cr = pd.DataFrame({'Hours Researched':[0, 0, 0.5, 0.75, 1.0, 1.0, 1.25, 1.25, 1.5, 1.5, 1.75, 1.75, 
                                       1.9, 2.0, 2.0, 2.0, 2.0, 2.25, 2.25, 2.25, 2.5, 2.5, 2.75, 3.0, 3.0],
                   'Hired':[0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,1,1,0,1,1,0,1,1,1,1]})

cr.head(10)
```


```python
cr.info()
```

### X) Understand and frame problem


```python

```

### 2) Clean data 


```python

```

### 3) Choose independent and dependent variables


```python

```

###  4) Split X and Y


```python
# The independent variable
X = cr['Hours Researched']
# The dependent variable
y = cr['Hired']
```

###  5) Plot X and Y to look for a linear relationship, perhaps using a Pearson correlation coefficient


```python

```

###  6) Check for multi-colinearity, perhaps using a Carr heatmap



```python

```

### 7) Add a constant to X


```python
# We still need to manually add an intercept
X_withconstant = sm.add_constant(X)
X_withconstant.head()

new_X_withconstant = sm.add_constant(new_X)
mylogreg_results.predict(new_X_withconstant)
# same as manual calculation above
```

### 8) Transformations and scale data


```python

```

###  9) Instantiate model


```python
# 1. Instantiate model
mylogreg = sm.Logit(y, X_withconstant)
```


```python

```

### 10) Fit model


```python
#2. Fit the model (this returns a separate object with the parameters)
mylogreg_results = mylogreg.fit()
```


```python
# 1. Instantiate model
mylogreg = sm.Logit(y, X_withconstant)

#2. Fit the model (this returns a separate object with the parameters)
mylogreg_results = mylogreg.fit()
mylogreg_results.summary()
```

### 11) Summarise model


```python
mylogreg_results.summary()
```


```python
mylogreg_results.params # calculated coefficients are stored in params
```


```python
beta0 = mylogreg_results.params[0]
```


```python
beta1 = mylogreg_results.params[1]
```


```python
np.exp(beta0)
```


```python
np.exp(beta1)
```

### Logistic regression model evaluation


```python
# remember we fit our model on X_withconstant
model_predictions_prob = mylogreg_results.predict(X_withconstant)
# getting the binary predictions
model_predictions_binary = np.where(model_predictions_prob>0.5,1,0)
```


```python
# comparing true and predicted 
(model_predictions_binary == cr['Hired']).sum()
```


```python

```

## Model Evaluation & Summary


```python
plt.figure(figsize = (10, 10))
bank_logit_fitted.params.sort_values().plot(kind = 'barh')
```


```python
# Model coefficients (i.e. the betas)
params_series = bank_logit_fitted.params
p_values_series = bank_logit_fitted.pvalues

significant_params = params_series[p_values_series < 0.05]
significant_params.drop('const', inplace=True)


plt.figure(figsize=(10,10))
significant_params.sort_values().plot(kind='barh')
plt.show()
```


```python
coefficients_df = pd.DataFrame({'coeff': bank_logit_fitted.params, 
                                'p-values': round(bank_logit_fitted.pvalues, 2),
                               'odds change': np.exp(bank_logit_fitted.params)})
coefficients_df.reset_index(inplace = True)
coefficients_df.rename({'index':'variable'}, axis='columns', inplace=True)


coefficients_df.sort_values(by='coeff')
```


```python

```

### Datetime


```python
from datetime import datetime

now = datetime.now()
print(now)

2022-10-28 18:11:06.352732
```


```python
# One of the most common transformations you’re likely to need to do when it comes to times is the one from a string, like “4 July 2002”, to a datetime. You can do this using datetime.strptime. Here’s an example:

date_string = "16 February in 2002"
datetime.strptime(date_string, "%d %B in %Y")

datetime.datetime(2002, 2, 16, 0, 0)
```


```python
# Differences in times

time_diff = now - datetime(year=2020, month=1, day=1)
print(time_diff)
```


```python
# Creates timezone aware datetime

import pytz
from pytz import timezone

aware = datetime(tzinfo=pytz.UTC, year=2020, month=1, day=1)
unaware = datetime(year=2020, month=1, day=1)

us_tz = timezone("US/Eastern")
us_aware = us_tz.localize(unaware)

print(us_aware - aware)
```


```python
# Create a time series examples

date + pd.to_timedelta(np.arange(12), "D")

pd.date_range(start="2018/1/1", end="2018/1/8")

pd.date_range("2018-01-01", periods=3, freq="H")
```


```python
# Identifies the datetime format
pd.to_datetime(small_df["date"], format="%m, '%y, %d")
```


```python
# Datetime offset
df["date"] = df["date"] + pd.offsets.MonthEnd()
df.head()
```


```python
# Resample with an aggregator
df.resample("A").mean()
```


```python
# Do a rolling mean with select periods
df.rolling(2).mean()
```


```python
# Shift time windows around. Shifting can move series around in time; it’s what we need to create leads and lags of time series. Let’s create a lead and a lag in the data. Remember that a lead is going to shift the pattern in the data to the left (ie earlier in time), while the lag is going to shift patterns later in time (ie to the right).

lead = 12
lag = 3
orig_series_name = df.columns[0]
df[f"lead ({lead} months)"] = df[orig_series_name].shift(-lead)
df[f"lag ({lag} months)"] = df[orig_series_name].shift(lag)
df.head()
```


```python
# Perform classical seasonal decomposition
classical_res = sm.tsa.seasonal_decompose(df, period=7)

# Extract the trend and seasonal components
classical_trend = classical_res.trend
classical_seasonal = classical_res.seasonal
classical_resid = classical_res.resid
```


```python
# Dealing with holidays
# We’ll be using the workalendar package which, as ever, you may need to install separately. Workalendar is a Python module that can provide lists of secular and religious holidays for a wide range of countries. (An alternative package is holidays).) Once we have the right holidays for the right country, we can proceed to control for them.

from rich import print
from datetime import date
from workalendar.usa import UnitedStates

cal = UnitedStates()
usa_hols_21_22 = cal.holidays(2020) + cal.holidays(2021)
print(usa_hols_21_22)
```


```python
air_traffic = air_traffic.set_index("Date") # Sets the index to date
```


```python
first_day = air_traffic.index.min()
last_day = air_traffic.index.max()

# pandas `Timestamp` objects
first_day, last_day
```


```python
The pd.Timestamp objects allow us to extract various datetime features or do arithmetic with dates:
    
    first_day.year, first_day.month, first_day.day
```


```python
The result of subtracting two dates is a pd.Timedelta object
```


```python
We can add durations using pd.DateOffset:
    
first_day + pd.DateOffset(years=1, months=3)
```


```python
Now, to add the missing dates, we can either use the reindex method with our full date range, or simply call the asfreq method:
    
    air_traffic_clean = air_traffic.reindex(full_range) 
```


```python
# With a date condition
air_traffic_monthly.loc[air_traffic_monthly.index <= "2000-12-31", ["Revenue Passenger Miles"]]
```


```python
# decompose the time series
decomposition = tsa.seasonal_decompose(air_traffic_monthly, model='additive')
```


```python
# Between time
the_timed_df=df["my_time_column"].between_time(date_from,date_to)
```


```python
# Uses indexing to get time slices
# Gets the 1993-01-01 to 1997-01-01 train data
train_rpm = new_rpm.loc['1993-01-01':'1997-01-01']

# Shows the data
train_rpm

# Gets the data past 1997-01-01 for testing
test_rpm = new_rpm.loc[new_rpm.index > "1997-01-01"]

# Shows the data
test_rpm

attempt = new_rpm.loc['1997-01-01': ]

attempt
```

## Machine learning
<a id='machine_learning'></a>

#### Importing library datasets
<a id='importing_library_datasets'></a>


```python
# Imports dataset From SKLearn
from sklearn.datasets import dataset
mydata = fetch_dataset()
```

#### Generate synthetic data
<a id='generate_synthetic_data'></a>


```python
# Generates synthetic data
x_one = np.array([[-1,1],[-1,0],[1,0],[1,-1]])
y_one = np.full((4),0)

x_two = np.array([[1,1],[-1,-1],[0,-1]])
y_two = np.full((3),1)

X = np.concatenate((x_one,x_two),axis=0)
y = np.concatenate((y_one,y_two),axis=0)
```


```python
# [Insert]
np.random.seed(12345)
# Generate the data
points_per_class = 500
mean = [0,0]
cov = [[1,0.8],[0.8,5]]
X_unnormalized = np.random.multivariate_normal(mean, cov, (points_per_class))
```


```python

```


```python

```

#### Add noise to data
<a id='add_noise_to_data'></a>


```python
# Add some noise to the data
random_state = np.random.RandomState(10)
n_samples, n_features = X.shape
X = X + (random_state.randn(n_samples, n_features))
```

#### Describe the data
<a id='describe_the_data'></a>


```python
# Describes the data
dataset.DESCR
```

#### Choose independent and target variables
<a id='choose_independent_and_target_variables'></a>


```python
# Assigns the dependent and target variables
X = dataset.data
y = dataset.target
```

#### Scale and transform a dataset
<a id='scale_and_transform_a_dataset'></a>


```python
# Imports a scaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
```


```python
# Assigns a scaler
my_scaler = StandardScaler()
my_minmax_scaler = MinMaxScaler()
```


```python
# Scales training data
scaler.fit(X_train)
X_train = scaler.transform(X_train)
my_minmax_scaler.fit(loans_df[["Credit Score", "Loan Request Amount"]])
```


```python
# Scales the testing data using the same scaler as with the training data
X_test = scaler.transform(X_test)
X_test_transformed = bagofwords.transform(X_test)
```


```python
# Transforms the data
scaled_loans_data = my_minmax_scaler.transform(loans_df[["Credit Score", "Loan Request Amount"]])
X_test_transformed = bagofwords.transform(X_test)
```

#### Import model
<a id='import_model'></a>


```python
# Imports model from SKLearn
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.svm import LinearSVC
```

#### Instantiate a model
<a id='instantiate_a_model'></a>

##### Instantiate a linear regression model
<a id='instantiate_a_linear_regression_model'></a>


```python
# Instantiates a linear regression model
my_linear_regression_model = LinearRegression() # 'max_iter = 10000' for changing the number of iterations
```

##### Instantiate a logistic regression model
<a id='instantiate_a_logistic_regression_model'></a>


```python
# Instantiates a logistic regression model
my_logreg_model = LogisticRegression()
```

##### Instantiate a KNN model
<a id='instantiate_a_knn_model'></a>


```python
# Instantiates a KNN model
my_KNN_model = KNeighborsClassifier(n_neighbors=3) # 'n_neighbors' is about the number of neighbours that the model is being asked to consider
```

##### Instantiate an SVC regression model
<a id='instantiate_an_svc_regression_model'></a>


```python
# Instantiates an SVC regression model
svc = SVC(kernel='rbf')
```

##### Instantiate a linear SVC model
<a id='instantiate_a_linear_svc_model'></a>


```python
# Instantiates a linear SVC model
linear_svc = LinearSVC(C=1)
```

##### Instantiate a naive bayes model
<a id='instantiate_a_naive_bayes_model'></a>


```python
# Instatiates a naive bayes model
nbmodel = BernoulliNB()
```

#### Splitting the testing and training datasets
<a id='splitting_the_testing_and_training_datasets'></a>


```python
# Imports the train-test-split function
from sklearn.model_selection import train_test_split
```


```python
# Gives the training and testing datasets. #'test_size' is the percentage of the data that goes to the test set, # random_state is about producing the same randomisation across runs, with the number not being that important
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=17) 
```

#### Fitting a model
<a id='fitting_a_model'></a>

##### Fitting a linear regression model
<a id='fitting_a_linear_regression_model'></a>


```python
# Fits a linear regression model
my_linear_regression_model.fit(X, y)
```

##### Fitting a logistic regression model
<a id='fitting_a_logistic_regression_model'></a>


```python
# Fits a logistic regression model
my_logreg_model.fit(X_train, y_train)
```

##### Fitting a KNN model
<a id='fitting_a_KNN_model'></a>


```python
# Fits a KNN model
my_KNN_model.fit(X_train, y_train)
```

##### Fitting a linesr SVM model
<a id='fitting_a_linear_svm_model'></a>


```python
SVM_model.fit(X_train, y_train)
```

##### Fitting a naive bayes model
<a id='fitting_a_naive_bayes_model'></a>


```python
# Fits a KNN model
nbmodel.fit(x,y)
```

#### Make predictions
<a id='make_predictions'></a>


```python
# Makes predictions
train_prediction = my_model.predict(X_train)
test_prediction = my_model.predict(X_test)
```

#### Score predictions
<a id='score_predictions'></a>


```python
# Scores predictions
accuracy_train = accuracy_score(train_prediction, y_train)
accuracy_test = accuracy_score(test_prediction, y_test)
```

#### Score model
<a id='score_model'></a>


```python
# Score the the model on the training and testing data
my_model.score(X_train, y_train)
my_model.score(X_test, y_test)
```

#### Test a model with main parameter variation
<a id='test_a_model_with_main_parameter_variation'></a>


```python
# Loops through a parameter of the model to show how the training and testing accuracies change
# max_depth = d
# C = c

## Get a list that varies a modelling parameter 
my_values = list(range(1, 1000))
my_values = list(range(1, len(X_train)))

## List for taking the training model accuracy scores
train_acc = []

## List for taking the testing model accuracy scores
test_acc = []

# Loops through the list, adjusting the model on the select parameter
for value in my_values: 
    # Assigns the value to the parameter
    my_model = model_function(parameter = value, max_iter=10000)
    # Fits the model on the X and y trainings
    my_model.fit(X_train, y_train)
    # Appends the training accuracy score to the list
    train_acc.append(my_model.score(X_train, y_train))
    # Appends the testing accuracy score to the list
    test_acc.append(my_model.score(X_test, y_test))
```

#### Plot the performance of a model with main parameter variation
<a id='plot_the_performance_of_a_model_with_main_parameter_variation'></a>


```python
# Plots the train and test scores re the adjusted parameter
plt.figure(figsize = (10, 7))
plt.plot(my_values, train_acc, label = 'train')
plt.plot(my_values, test_acc, label = 'test')
plt.legend()
plt.xlabel('Insert')
plt.ylabel('Insert')
```

#### Plot the boundary of a logistic regression model
<a id='plot_the_boundary_of_a_logistic_regression_model'></a>


```python
# Plot decision boundary with the Test Data overlaid on it
PlotBoundaries(LR_model, X_test, y_test)
```

### NLP
<a id='nlp'></a>

#### Packages


```python
#Importing the packages we will be using
# Basic Packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import string

# SK Packages
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.svm import SVC, LinearSVC

## Vectorizers
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

# NLTK
import nltk
```


```python
# NLTK Packages
# Use the code below to download the NLTK package, a straightforward GUI should pop up
# nltk.download()
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
```

#### Lowercase letters
<a id='lowercase_letters'></a>


```python
text = str.lower(text)
```

#### Remove punctuation
<a id='remove_punctuation'></a>


```python
def remove_punctuation(document, punc):
    '''
    Removes punctuation provided (string) from document (string)
    '''
    document = re.sub(f"\{punc}", '', document)
    
    return document
```


```python
remove_punctuation(text, ',')
```


```python
string.punctuation
```


```python
def remove_all_punctuation(document):
    '''
    Removes all punctuation (found in string.punctuation) from document (string)
    '''
    
    for punc in string.punctuation:
        document = remove_punctuation(document, punc)
    
    return document
```

#### Tokenise
<a id='tokenise'></a>


```python
tokenized_document = text.split(' ')
```


```python
def tokenizer(document):
    '''
    Tokenizes the document
    '''
    document = str.lower(document)
    document = remove_all_punctuation(document)
    tokenized_document = document.split(' ')
    return tokenized_document
```

#### Remove stopwords
<a id='remove_stopwords'></a>


```python
#Load up our stop words
stop_words = stopwords.words('english')
# look at your list of stopwords
print(stop_words)
```


```python
def remove_stopwords(list_of_tokens):
    """
    Literally removes stopwords
    """
    
    cleaned_tokens = [] 
    
    for token in list_of_tokens: 
        if token not in stop_words: 
            cleaned_tokens.append(token)
            
    return cleaned_tokens
```


```python
tokenized_text2 = remove_stopwords(tokenized_document)
print(tokenized_text2)
```

#### Stemming and lemmatisation
<a id='stemming_and_lemmatisation'></a>

##### Stemming
<a id='stemming'></a>


```python
porterStemmer = PorterStemmer()
```


```python
def stemmer(list_of_tokens):
    '''
    Takes in an input which is a list of tokens, and spits out a list of stemmed tokens.
    '''
    
    stemmed_tokens_list = []
    
    for i in list_of_tokens:
        token = PorterStemmer().stem(i)
        stemmed_tokens_list.append(token)
        
    return stemmed_tokens_list
```


```python

```

##### Lemmatisation
<a id='lemmatisation'></a>


```python
def lemmatizer(list_of_tokens):
    
    lemmatized_tokens_list = []
    
    for i in list_of_tokens: 
        token = WordNetLemmatizer().lemmatize(i)
        lemmatized_tokens_list.append(token)
        
    return lemmatized_tokens_list
```


```python
tokenized_text2 = lemmatizer(tokenized_text2)
print(tokenized_text2)
```

#### Vectorisation
<a id='vectorisation'></a>

##### Bag of words
<a id='bag_of_words'></a>


```python
def remove_punctuation(document, punc):
    '''
    Removes punctuation provided (string) from document (string)
    '''
    document = re.sub(f"\{punc}", '', document)
    
    return document
```


```python
def remove_all_punctuation(document):
    '''
    Removes all punctuation (found in string.punctuation) from document (string)
    '''
    
    for punc in string.punctuation:
        document = remove_punctuation(document, punc)
    
    return document
```


```python
def remove_stopwords(list_of_tokens):
    """
    Literally removes stopwords
    """
    
    stop_words = stopwords.words('english')
    
    cleaned_tokens = [] 
    
    for token in list_of_tokens: 
        if token not in stop_words: 
            cleaned_tokens.append(token)
            
    return cleaned_tokens
```


```python
def stemmer(list_of_tokens):
    '''
    Takes in an input which is a list of tokens, and spits out a list of stemmed tokens.
    '''
    
    stemmed_tokens_list = []
    
    for i in list_of_tokens:
        token = PorterStemmer().stem(i)
        stemmed_tokens_list.append(token)
        
    return stemmed_tokens_list
```


```python
def lemmatizer(list_of_tokens):
    
    lemmatized_tokens_list = []
    
    for i in list_of_tokens: 
        token = WordNetLemmatizer().lemmatize(i)
        lemmatized_tokens_list.append(token)
        
    return lemmatized_tokens_list
```


```python
def my_tokenizer(document, lemmatization=False, stemming=True):
    '''
    Function for use in Vectorizer that tokenizes the document
    '''
    # lowercase
    document = str.lower(document)
    # remove punctuation
    document = remove_all_punctuation(document)
    # tokenize - split on whitespace
    tokenized_document = document.split(' ')
    # remove stopwords
    tokenized_document = remove_stopwords(tokenized_document)
    # lemmatization
    if lemmatization:
        tokenized_document = lemmatizer(tokenized_document)
    # stemming
    if stemming:
        tokenized_document = stemmer(tokenized_document)
        
    return tokenized_document
    
```


```python
vectorizer = CountVectorizer(tokenizer=my_tokenizer, ngram_range=(1, 3))
bag_of_words = vectorizer.fit_transform([text])
```


```python
bag_of_words.toarray()
```




    array([[8, 1, 1, 5, 1, 1, 2, 1, 1, 1, 1, 4, 4, 1, 3, 1, 1, 1, 1, 1, 1, 3,
            1, 1, 2, 1, 1, 7, 1, 1, 2, 1, 1, 3, 1, 1, 1, 6, 1, 1, 3, 3, 1, 1,
            1, 1, 5, 1, 1, 1, 1, 1, 1, 2, 1, 1]])




```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```

##### TF-IDF
<a id='tf_idf'></a>


```python

```

### Principal component analysis (PCA)
<a id='principal_component_analysis'></a>

##### High-level steps for principal component analysis (PCA)
<a id='high_level_steps_for_principal_component_analysis'></a>

Here is the high level modelling process we will follow:

Load data and split into X (features) and y (target)
Train test split
Fit scaler to X_train, transform X_train and X_test
Fit PCA object to scaled X_train, transform scaled X_train and X_test
Fit model on PCA-transformed X_train, score on transformed X_train and X_test.


```python
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
```


```python
# 1. Load data
cancer = load_breast_cancer()
X = cancer.data
y = cancer.target
print("There are", X.shape[0], 'data points, each with', X.shape[1], 'features.')
```


```python
# 2. Train and test split
split = 0.20
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split, random_state=17)
```


```python
# 3. Scale data
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
```

##### Steps for choosing the number of principal components
<a id='steps_for_choosing_the_number_of_principal_components'></a>

When applying PCA we need to choose the number of principal components we want to keep. A lower number means a larger reduction of the number of dimensions, however this will come at the cost of losing additional information from the components which are dropped.

The process for picking the number of PCs is as follows:

Apply PCA to the original k-dimensional data set.
Extract the explained variance ratios for the principal components.
Use one of the following methods to decide how many PCs can represent the data well enough:
Make a line plot of the explained variance ratios. Look for an elbow in the plot to decide how many PCs are sufficient, as that will tell you that adding more PCs is not substantially raising the explained variance.
Decide on a threshold for how much variance needs to be explained by the PCs (e.g. if I want to preserve 90% of the original dataset's variance, how many PCs do I need to include in my reduced dimension data?)
Based on your decision, re-fit a PCA object with the required number of PCs, and use that to create a lower-dimensional representation of your data for further analysis.


```python
# Default n_components will generate the same number of PCs as you have features 
my_PCA = PCA()
my_PCA.fit(X_train)
```


```python
# transform data 
X_train_PCA = my_PCA.transform(X_train)
X_test_PCA = my_PCA.transform(X_test)
```


```python
print(f"Variance captured by PC1: {my_PCA.explained_variance_[0]: 0.3f}")
print(f"Variance captured by PC2: {my_PCA.explained_variance_[1]: 0.3f}")

print(f"Proportion of variance captured by PC1: {my_PCA.explained_variance_ratio_[0]: 0.3f}")
print(f"Proportion of variance captured by PC2: {my_PCA.explained_variance_ratio_[1]: 0.3f}")
```


```python
my_PCA.explained_variance_ratio_
```

#### Method 1: Make a line plot of the explained variance_ratio_ attribute
<a id='method_one'></a>


```python
expl_var = my_PCA.explained_variance_ratio_
```


```python
plt.figure()
plt.plot(range(1,31),expl_var,marker='.')
plt.xlabel('Number of PCs')
plt.ylabel('Proportion of Variance Explained')
plt.xticks(range(1,31,2))
plt.show()
```

#### Method 2: Use a pre-set threshold for proportion of variance explained
<a id='method_two'></a>


```python
# Pull out the explained variance ratio
expl_var = my_PCA.explained_variance_ratio_

# Calculate the cumulative sum of this array using the 
cumulative_sum = np.cumsum(expl_var)

cumulative_sum
```


```python
# Plot out the cumulative sum graph

plt.figure()
plt.plot(range(1,31), cumulative_sum, marker='.')
plt.axhline(0.9, c='r', linestyle='--')
plt.xlabel('Number of PCs')
plt.ylabel('Cumulative Sum of Explained Variance')
plt.xticks(range(1,31,2))
plt.show()
```

#### Fitting a model
<a id='fitting_a_model'></a>


```python
from sklearn.linear_model import LogisticRegression
```


```python
# Let's use all the default parameters for now
my_logreg = LogisticRegression()

# Fitting to original data
my_logreg.fit(X_train,y_train)

# Scoring on original train and test sets
print(f'Train Score: {my_logreg.score(X_train, y_train)}')
print(f'Test Score: {my_logreg.score(X_test, y_test)}')
```


```python
# Do the same but fit on the PCA transformed data
my_logreg_PCA = LogisticRegression()

# Fitting to PCA data
my_logreg_PCA.fit(X_train_PCA,y_train)

# Scoring on PCA train and test sets
print(f'Train Score: {my_logreg_PCA.score(X_train_PCA, y_train)}')
print(f'Test Score: {my_logreg_PCA.score(X_test_PCA, y_test)}')
```

#### Get principal components
<a id='get_principal_components'></a>


```python
my_PCA.components_
```

#### Heatmap of principal components
<a id='heatmap_of_principal_components'></a>


```python
plt.figure(figsize=(10,8))

# Create a heatmap. The values are all contained in the .components_ attribute
ax = sns.heatmap(my_PCA.components_,
                 cmap='coolwarm',
                 yticklabels=[ "PC"+str(x) for x in range(1,my_PCA.n_components_+1)],
                 xticklabels=list(X.columns),
                 linewidths = 1,
                 annot = True,
                 vmin=-1,
                 vmax=1,
                 cbar_kws={"orientation": "vertical"})

plt.yticks(rotation=0)
plt.xticks(rotation=25)
ax.set_aspect("equal")
```

#### Plot principal components
<a id='plot_principal_components'></a>


```python
plt.figure(figsize=(10,5))
scatter = plt.scatter(X_train_PCA[:,0], X_train_PCA[:,1], c=y_train, cmap='tab10')

plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")

# Generate Legend
classes = ['Setosa', 'Virginica']
plt.legend(handles=scatter.legend_elements()[0], labels=classes)

plt.show()
```


```python

```


```python

```


```python

```

### Feature engineering
<a id='feature_engineering'></a>


```python
# Variance threshold
from sklearn.feature_selection import VarianceThreshold

# Instantiate the VarianceThresholder, we need to set a threshold variance
my_vt = VarianceThreshold(threshold=0.0004)

# Fit to the data and calculate the variances per column
my_vt.fit(df2_scaled)
```


```python
# Visualise the variance threshold
# Extract the variances per column
column_variances = my_vt.variances_

# Plot with the threshold
plt.figure()
plt.barh(np.flip(df2_scaled.columns), np.flip(column_variances))
plt.xlabel('Variance ($)')
plt.ylabel('Feature')
plt.axvline(0.0004, color='black', linestyle='--')
plt.show()
```


```python
# Get the columns which are retained
my_vt.get_support()
```


```python
# Apply the variance threshold to drop columns below the given variance
pd.DataFrame(my_vt.transform(df2_scaled), columns = df2.columns[my_vt.get_support()]).head()
```


```python
# Using KBest
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif

# Pull out features and target
X = retail_df[['quantity', 'shipping_cost']]
y = retail_df['margin_gt_15']

# Instantiate KBest feature selector and fit
my_KBest = SelectKBest(f_classif, k=1).fit(X, y)

my_KBest.get_support()

my_KBest.scores_
```


```python
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 111)

# Select 5 best for regression
my_K_best = SelectKBest(f_regression, k=5)
my_K_best.fit(X_train, y_train)
print(f"Variables chosen: {np.array(diabetes.feature_names)[my_K_best.get_support()]}")
X_train_selected = my_K_best.transform(X_train)
X_test_selected = my_K_best.transform(X_test)

# Fit linear regression
model = LinearRegression()
model.fit(X_train_selected, y_train)
print(model.score(X_train_selected, y_train))
print(model.score(X_test_selected, y_test))
```


    ---------------------------------------------------------------------------

    ValueError                                Traceback (most recent call last)

    /var/folders/3w/7b2j0crd0hvc3bb28cmq6j_40000gn/T/ipykernel_11049/448313915.py in <module>
          9 # Select 5 best for regression
         10 my_K_best = SelectKBest(f_regression, k=5)
    ---> 11 my_K_best.fit(X_train, y_train)
         12 print(f"Variables chosen: {np.array(diabetes.feature_names)[my_K_best.get_support()]}")
         13 X_train_selected = my_K_best.transform(X_train)


    ~/opt/anaconda3/lib/python3.9/site-packages/sklearn/feature_selection/_univariate_selection.py in fit(self, X, y)
        405             )
        406 
    --> 407         self._check_params(X, y)
        408         score_func_ret = self.score_func(X, y)
        409         if isinstance(score_func_ret, (list, tuple)):


    ~/opt/anaconda3/lib/python3.9/site-packages/sklearn/feature_selection/_univariate_selection.py in _check_params(self, X, y)
        602     def _check_params(self, X, y):
        603         if not (self.k == "all" or 0 <= self.k <= X.shape[1]):
    --> 604             raise ValueError(
        605                 "k should be >=0, <= n_features = %d; got %r. "
        606                 "Use k='all' to return all features." % (X.shape[1], self.k)


    ValueError: k should be >=0, <= n_features = 2; got 5. Use k='all' to return all features.



```python
# Try other values of K

from sklearn.feature_selection import f_regression
from sklearn.linear_model import LinearRegression

train_scores = []
test_scores = []

ks = list(range(1,6))

for k in ks:
    
    my_KBest = SelectKBest(f_regression,k=k).fit(X_train,y_train)
    
    X_train_selected = my_KBest.transform(X_train)
    X_test_selected = my_KBest.transform(X_test)
    
    model = LinearRegression()
    model.fit(X_train_selected,y_train)
    
    print(f"k = {k}, {np.array(diabetes.feature_names)[my_KBest.get_support()]}")
    
    train_scores.append(model.score(X_train_selected,y_train))
    test_scores.append(model.score(X_test_selected,y_test))a
```


      File "/var/folders/3w/7b2j0crd0hvc3bb28cmq6j_40000gn/T/ipykernel_11049/3257251367.py", line 24
        test_scores.append(model.score(X_test_selected,y_test))a
                                                               ^
    SyntaxError: invalid syntax




```python
# Lasso and Ridge regressions
# Imports
from sklearn.linear_model import Lasso, Ridge

# Instantiate
mylasso = Lasso()
myridge = Ridge()

# Fit models
mylasso.fit(X_train_s, y_train)
myridge.fit(X_train_s ,y_train)
```

#### One-Hot Encoding
<a id='one_hot_encoding'></a>


```python
from sklearn.preprocessing import OneHotEncoder

# Instantiate the OneHotEncoder
ohe = OneHotEncoder()

# Fit the OneHotEncoder to the subcategory column and transform
# It expects a 2D array, so we first convert the column into a DataFrame
subcategory = pd.DataFrame(retail_df['subcategory'])

encoded = ohe.fit_transform(subcategory)
encoded
```


```python
# Convert from sparse matrix to dense
dense_array = encoded.toarray()
dense_array
```


```python
# Get the categories
ohe.categories_
```


```python
# Put into a dataframe to get column names
encoded_df = pd.DataFrame(dense_array, columns=ohe.categories_, dtype=int)

# Add original back in (just to check)
encoded_df['subcategory'] = retail_df['subcategory']

# Show
encoded_df.head()
```

#### Ordinal encoding
<a id='ordinal_encoding'></a>


```python
from sklearn.preprocessing import OrdinalEncoder

oe = OrdinalEncoder(categories=[['Low','Medium','High','Critical']])
oe.fit_transform(pd.DataFrame(retail_df['order_priority']))
```


```python
oe.categories_
```

#### Label encoding
<a id='label_encoding'></a>


```python
from sklearn.preprocessing import LabelEncoder

# Instantiate the label encoder
le = LabelEncoder()

# Fit and transform the order priority column
le.fit_transform(retail_df['order_priority'])
```


```python
le.classes_
```

### Modelling functions
<a id='modelling_function'></a>


```python
# Load data
mydata = pd.read_csv('data.csv',index_col=0)
```


```python
# Assign X and y
X = mydata[['column1','column']] 
y = loans['column'] 

# Or, if the date is from a loaded library
X = mydata.data
y = mydata.target
```


```python
# Train-test-split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```


```python
# Scale and transform training data - you can do this by 'creating a scaled version of the data'
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

# Make a scaler
scaler = StandardScaler()
scaler = MinMaxScaler()

# Fit the scaler
scaler.fit(mydata[["column", "column"]])
scaler.fit(X_train)

# Scale and transform the training data
mydata_scaled = scaler.transform(mydata[["column", "column"]])
mydata_scaled = scaler.transform(loans_df[["column", "column"]])
X_train = scaler.transform(X_train)

# Transform the testing data
X_test = scaler.transform(X_test)
```


    ---------------------------------------------------------------------------

    TypeError                                 Traceback (most recent call last)

    /var/folders/3w/7b2j0crd0hvc3bb28cmq6j_40000gn/T/ipykernel_1516/1590578944.py in <module>
          8 
          9 # Fit the scaler
    ---> 10 scaler.fit(mydata[["column", "column"]])
         11 scaler.fit(X_train)
         12 


    TypeError: unhashable type: 'list'



```python
# Import model library
from sklearn.linear_model import LinearRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
```


```python
# Instantiate the model: my_model_instance = sklearn_model()
linear_regression_model = LinearRegression()
logistic_regression_model = LogisticRegression() #C = 0.1
KNN_model = KNeighborsClassifier() #n_neighbors=3
Decision_tree_model = DecisionTreeClassifier() #max_depth=1
bagofwords = CountVectorizer() # stop_words="english", min_df=5, max_features=1000
svc = SVC() #kernel='rbf'
linear_svc = LinearSVC()
SVM_model = LinearSVC() # C=1/(10**10)
nbmodel = MultinomialNB()
my_PCA = PCA() # n_components=2
```


```python
# Fit the model: model.fit(X, y)
linear_regression_model.fit()
KNN_model.fit()
Decision_tree_model.fit()
bagofwords.fit()
logistic_regression_model.fit()
svc.fit()
linear_svc.fit()
SVM_model.fit()
PCA.fit(X)
```


```python
# Test, score, predict with, and evaluate the model 

# Import scoring libraries
from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score

# Make predictions with the model
model_prediction = linear_regression_model.predict(X)
train_prediction = logistic_regression_model.predict(X_train)
test_predictions = KNN_model.predict(X_test)

# Score the model
linear_regression_model.score(X, y)
logistic_regression_model.score(X_train_transformed, y_train)
r2_score(y, model_prediction)
accuracy_score(y, model_prediction)
accuracy_score(train_prediction, y_train)
accuracy_score(test_prediction, y_test)
SVM_model.score(X_train, y_train)
knn.score(X_train_transformed, y_train)
dt.score(X_train_transformed, y_train)
linear_svc.score(X_train_transformed, y_train)
```

#### Model builder


```python
# Load data
mydata = pd.read_csv('data.csv',index_col=0)
```


```python
# Assign X and y
X = mydata[['column1','column']] 
y = loans['column'] 

# Or, if the date is from a loaded library
X = mydata.data
y = mydata.target
```


```python
# Train-test-split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```


```python
# Function for training and testing a model
def baseline_model_training_and_testing(X_train, X_test, y_train, y_test, select_scaler, sklearn_model):
    '''
    Function for training and testing a model. 
    
    Takes in the following: 
    
    - independent_variables
    - dependent_variable
    - select_test_size
    - select_random_state
    - select_scaler
    = sklearn_model
    
    Gives out a model score. 
    
    '''
    
    # Prints inputs
    print(f'You have chosen a {sklearn_model} with {select_scaler}.')
    
    print('The balance in the training and testing data is...')

    # Makes a scaler
    scaler = select_scaler 

    # Fits the scaler
    scaler.fit(X_train)

    # Scales and transform the training data
    X_train_scaled = scaler.transform(X_train)

    # Transforms the testing data
    X_test_scaled = scaler.transform(X_test)
    #display(X_test_scaled)

    # Instantiates the model
    instantiated_model = sklearn_model

    # Fits the model 
    instantiated_model.fit(X_train, y_train)

    # Scores the model
    instantiated_model_score = instantiated_model.score(X, y)
    
    # Prints empty line to separate outputs for readability
    print('')
    
    # Returns
    print(f'Your model scored {instantiated_model_score}.') 
```

#### Pipeline


```python
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
```


```python
from sklearn.datasets import load_breast_cancer
```


```python
mydata = load_breast_cancer()
```


```python
X = mydata.data
y = mydata.target
```


```python
# Train-test-split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```


```python
# Reduce dimensions
pca = PCA(n_components=3)
pca.fit(X_train)
X_train = pca.transform(X_train)
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    /var/folders/3w/7b2j0crd0hvc3bb28cmq6j_40000gn/T/ipykernel_1516/2652172143.py in <module>
          1 # Reduce dimensions
    ----> 2 pca = PCA(n_components=3)
          3 pca.fit(X_train)
          4 X_train = pca.transform(X_train)


    NameError: name 'PCA' is not defined



```python
baseline_model_training_and_testing(X_train, X_test, y_train, y_test, StandardScaler(), KNeighborsClassifier())
```

    You have chosen a KNeighborsClassifier() with StandardScaler().
    The balance in the training and testing data is...
    
    Your model scored 0.9437609841827768.


    /Users/lawrencekay/opt/anaconda3/lib/python3.9/site-packages/sklearn/neighbors/_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)



```python
baseline_model_training_and_testing(X_train, X_test, y_train, y_test, StandardScaler(), LinearRegression())
```

    You have chosen a LinearRegression() with StandardScaler().
    The balance in the training and testing data is...
    
    Your model scored 0.7685381178156124.



```python
from sklearn.linear_model import LogisticRegression
```


```python
baseline_model_training_and_testing(X_train, X_test, y_train, y_test, StandardScaler(), LogisticRegression())
```

    You have chosen a LogisticRegression() with StandardScaler().
    The balance in the training and testing data is...
    
    Your model scored 0.9560632688927944.


    /Users/lawrencekay/opt/anaconda3/lib/python3.9/site-packages/sklearn/linear_model/_logistic.py:814: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      n_iter_i = _check_optimize_result(



```python
model_hyperparameter_optimisation(X_train, X_test, y_train, y_test, StandardScaler(), 
                                  LinearRegression, 10)


```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    /var/folders/3w/7b2j0crd0hvc3bb28cmq6j_40000gn/T/ipykernel_1516/3465459521.py in <module>
    ----> 1 model_hyperparameter_optimisation(X_train, X_test, y_train, y_test, StandardScaler(), 
          2                                   LinearRegression, 10)
          3 


    NameError: name 'model_hyperparameter_optimisation' is not defined



```python

```


```python

```


```python

```

#### Model maker


```python
# Function for training and testing a model
def baseline_model_training_and_testing(independent_variables, dependent_variable, select_test_size, select_random_state, select_scaler, sklearn_model):
    '''
    Function for training and testing a model. 
    
    Takes in the following: 
    
    - independent_variables
    - dependent_variable
    - select_test_size
    - select_random_state
    - select_scaler
    = sklearn_model
    
    Gives out a model score. 
    
    '''
    
    # Prints inputs
    print(f'You have chosen a {sklearn_model}, {select_scaler}, a test size of {select_test_size}, and a random state of {select_random_state}.')
    
    print(f'The independent_variables have {len(independent_variables)}')
    
    print('The balance in the training and testing data is...')
    
    # Splits data into training and testing
    X_train, X_test, y_train, y_test = train_test_split(independent_variables, dependent_variable, test_size = select_test_size, random_state = select_random_state)

    # Makes a scaler
    scaler = select_scaler 

    # Fits the scaler
    scaler.fit(X_train)

    # Scales and transform the training data
    X_train_scaled = scaler.transform(X_train)

    # Transforms the testing data
    X_test_scaled = scaler.transform(X_test)
    #display(X_test_scaled)

    # Instantiates the model
    instantiated_model = sklearn_model

    # Fits the model 
    instantiated_model.fit(X, y)

    # Scores the model
    instantiated_model_score = instantiated_model.score(X, y)
    
    # Prints empty line to separate outputs for readability
    print('')
    
    # Returns
    print(f'Your model scored {instantiated_model_score}.') 
```


```python

```


```python

```


```python

```


```python

```

#### Hyperparameter optimisation


```python
# Function for optimising the hyperparameters of a model

def model_hyperparameter_optimisation(X_train, X_test, y_train, y_test, select_scaler, sklearn_model, hyperparameter_range):
    
    # Create list to loop through
    hyperparameter_list = (1, hyperparameter_range)

    # Creates list to take training and testing accuracy scores
    training_accuracy = []
    testing_accuracy = []

    # Loops through hyperparameter range, collecting train and test scores
    for hyperparameter in hyperparameter_list:
        # Gives the hyperparameter in the list to the instantiated model
        instantiated_model = sklearn_model(K = hyperparameter)
        # Fits the instantiated model with the given hyperparameter to the training data
        instantiated_model.fit(X_train, y_train)
        # Appends the score of the instantiated model with the given hyperparameter to the training accuracy list
        training_accuracy.append(instantiated_model.score(X_train, y_train))
        # Appends the score of the instantiated model with the given hyperparameter to the testing accuracy list
        testing_accuracy.append(instantiated_model.score(X_test, y_test))

    # Plots the training and testing accuracy scores      
    plt.figure()
    # Plots the list of hyperparameters and the training scores on them
    plt.plot(hyperparameter_list, training_accuracy, label = 'Training')
    # Plots the list of hyperparameters and the training scores on them
    plt.plot(hyperparameter_list, testing_accuracy, label = 'Testing')
    # Creates a title
    plt.title('Accuracy and hyperparameter iterations')
    # Creates a legend 
    plt.legend()
    # Creates labels on the hyperparameter iterations
    plt.xlabel('Number of hyperparameters')
    # Creates labels on accuracy per hyperparameter iteration
    plt.ylabel("Accuracy per hyperparameter iteration")
    # Shows the plot
    plot = plt.show()
    
    # Returns
    return plot
    
```


```python
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
```


```python
from sklearn.datasets import load_breast_cancer
```


```python
baseline_model_training_and_testing(cancer.data, cancer.target, 0.2, 42, StandardScaler(), KNeighborsClassifier)
```

    You have chosen a <class 'sklearn.neighbors._classification.KNeighborsClassifier'> with StandardScaler().
    The balance in the training and testing data is...



    ---------------------------------------------------------------------------

    ValueError                                Traceback (most recent call last)

    /var/folders/3w/7b2j0crd0hvc3bb28cmq6j_40000gn/T/ipykernel_11049/1791125949.py in <module>
    ----> 1 baseline_model_training_and_testing(cancer.data, cancer.target, 0.2, 42, StandardScaler(), KNeighborsClassifier)
    

    /var/folders/3w/7b2j0crd0hvc3bb28cmq6j_40000gn/T/ipykernel_11049/3555095622.py in baseline_model_training_and_testing(X_train, X_test, y_train, y_test, select_scaler, sklearn_model)
         32 
         33     # Transforms the testing data
    ---> 34     X_test_scaled = scaler.transform(X_test)
         35     #display(X_test_scaled)
         36 


    ~/opt/anaconda3/lib/python3.9/site-packages/sklearn/preprocessing/_data.py in transform(self, X, copy)
        971 
        972         copy = copy if copy is not None else self.copy
    --> 973         X = self._validate_data(
        974             X,
        975             reset=False,


    ~/opt/anaconda3/lib/python3.9/site-packages/sklearn/base.py in _validate_data(self, X, y, reset, validate_separately, **check_params)
        564             raise ValueError("Validation should be done on X, y or both.")
        565         elif not no_val_X and no_val_y:
    --> 566             X = check_array(X, **check_params)
        567             out = X
        568         elif no_val_X and not no_val_y:


    ~/opt/anaconda3/lib/python3.9/site-packages/sklearn/utils/validation.py in check_array(array, accept_sparse, accept_large_sparse, dtype, order, copy, force_all_finite, ensure_2d, allow_nd, ensure_min_samples, ensure_min_features, estimator)
        767             # If input is 1D raise error
        768             if array.ndim == 1:
    --> 769                 raise ValueError(
        770                     "Expected 2D array, got 1D array instead:\narray={}.\n"
        771                     "Reshape your data either using array.reshape(-1, 1) if "


    ValueError: Expected 2D array, got 1D array instead:
    array=[0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 1. 1. 0. 0.
     0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0.
     1. 1. 1. 1. 1. 0. 0. 1. 0. 0. 1. 1. 1. 1. 0. 1. 0. 0. 1. 1. 1. 1. 0. 1.
     0. 0. 1. 0. 1. 0. 0. 1. 1. 1. 0. 0. 1. 0. 0. 0. 1. 1. 1. 0. 1. 1. 0. 0.
     1. 1. 1. 0. 0. 1. 1. 1. 1. 0. 1. 1. 0. 1. 1. 1. 1. 1. 1. 1. 1. 0. 0. 0.
     1. 0. 0. 1. 1. 1. 0. 0. 1. 0. 1. 0. 0. 1. 0. 0. 1. 1. 0. 1. 1. 0. 1. 1.
     1. 1. 0. 1. 1. 1. 1. 1. 1. 1. 1. 1. 0. 1. 1. 1. 1. 0. 0. 1. 0. 1. 1. 0.
     0. 1. 1. 0. 0. 1. 1. 1. 1. 0. 1. 1. 0. 0. 0. 1. 0. 1. 0. 1. 1. 1. 0. 1.
     1. 0. 0. 1. 0. 0. 0. 0. 1. 0. 0. 0. 1. 0. 1. 0. 1. 1. 0. 1. 0. 0. 0. 0.
     1. 1. 0. 0. 1. 1. 1. 0. 1. 1. 1. 1. 1. 0. 0. 1. 1. 0. 1. 1. 0. 0. 1. 0.
     1. 1. 1. 1. 0. 1. 1. 1. 1. 1. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
     0. 0. 1. 1. 1. 1. 1. 1. 0. 1. 0. 1. 1. 0. 1. 1. 0. 1. 0. 0. 1. 1. 1. 1.
     1. 1. 1. 1. 1. 1. 1. 1. 1. 0. 1. 1. 0. 1. 0. 1. 1. 1. 1. 1. 1. 1. 1. 1.
     1. 1. 1. 1. 1. 0. 1. 1. 1. 0. 1. 0. 1. 1. 1. 1. 0. 0. 0. 1. 1. 1. 1. 0.
     1. 0. 1. 0. 1. 1. 1. 0. 1. 1. 1. 1. 1. 1. 1. 0. 0. 0. 1. 1. 1. 1. 1. 1.
     1. 1. 1. 1. 1. 0. 0. 1. 0. 0. 0. 1. 0. 0. 1. 1. 1. 1. 1. 0. 1. 1. 1. 1.
     1. 0. 1. 1. 1. 0. 1. 1. 0. 0. 1. 1. 1. 1. 1. 1. 0. 1. 1. 1. 1. 1. 1. 1.
     0. 1. 1. 1. 1. 1. 0. 1. 1. 0. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 0. 1.
     0. 0. 1. 0. 1. 1. 1. 1. 1. 0. 1. 1. 0. 1. 0. 1. 1. 0. 1. 0. 1. 1. 1. 1.
     1. 1. 1. 1. 0. 0. 1. 1. 1. 1. 1. 1. 0. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 0.
     1. 1. 1. 1. 1. 1. 1. 0. 1. 0. 1. 1. 0. 1. 1. 1. 1. 1. 0. 0. 1. 0. 1. 0.
     1. 1. 1. 1. 1. 0. 1. 1. 0. 1. 0. 1. 0. 0. 1. 1. 1. 0. 1. 1. 1. 1. 1. 1.
     1. 1. 1. 1. 1. 0. 1. 0. 0. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.
     1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 0. 0. 0. 0. 0. 0. 1.].
    Reshape your data either using array.reshape(-1, 1) if your data has a single feature or array.reshape(1, -1) if it contains a single sample.



```python
import warnings
warnings.filterwarnings('ignore')
```


```python
model_hyperparameter_optimisation(cancer.data, cancer.target, 0.2, 42, StandardScaler(), LogisticRegression, 20)
    
    
```


    ---------------------------------------------------------------------------

    ValueError                                Traceback (most recent call last)

    /var/folders/3w/7b2j0crd0hvc3bb28cmq6j_40000gn/T/ipykernel_11049/255501125.py in <module>
    ----> 1 model_hyperparameter_optimisation(cancer.data, cancer.target, 0.2, 42, StandardScaler(), LogisticRegression, 20)
          2 
          3 


    /var/folders/3w/7b2j0crd0hvc3bb28cmq6j_40000gn/T/ipykernel_11049/1001966764.py in model_hyperparameter_optimisation(X_train, X_test, y_train, y_test, select_scaler, sklearn_model, hyperparameter_range)
         15         instantiated_model = sklearn_model(C = hyperparameter)
         16         # Fits the instantiated model with the given hyperparameter to the training data
    ---> 17         instantiated_model.fit(X_train, y_train)
         18         # Appends the score of the instantiated model with the given hyperparameter to the training accuracy list
         19         training_accuracy.append(instantiated_model.score(X_train, y_train))


    ~/opt/anaconda3/lib/python3.9/site-packages/sklearn/linear_model/_logistic.py in fit(self, X, y, sample_weight)
       1506             _dtype = [np.float64, np.float32]
       1507 
    -> 1508         X, y = self._validate_data(
       1509             X,
       1510             y,


    ~/opt/anaconda3/lib/python3.9/site-packages/sklearn/base.py in _validate_data(self, X, y, reset, validate_separately, **check_params)
        579                 y = check_array(y, **check_y_params)
        580             else:
    --> 581                 X, y = check_X_y(X, y, **check_params)
        582             out = X, y
        583 


    ~/opt/anaconda3/lib/python3.9/site-packages/sklearn/utils/validation.py in check_X_y(X, y, accept_sparse, accept_large_sparse, dtype, order, copy, force_all_finite, ensure_2d, allow_nd, multi_output, ensure_min_samples, ensure_min_features, y_numeric, estimator)
        977     )
        978 
    --> 979     y = _check_y(y, multi_output=multi_output, y_numeric=y_numeric)
        980 
        981     check_consistent_length(X, y)


    ~/opt/anaconda3/lib/python3.9/site-packages/sklearn/utils/validation.py in _check_y(y, multi_output, y_numeric)
        991         )
        992     else:
    --> 993         y = column_or_1d(y, warn=True)
        994         _assert_all_finite(y)
        995         _ensure_no_complex_data(y)


    ~/opt/anaconda3/lib/python3.9/site-packages/sklearn/utils/validation.py in column_or_1d(y, warn)
       1036         return np.ravel(y)
       1037 
    -> 1038     raise ValueError(
       1039         "y should be a 1d array, got an array of shape {} instead.".format(shape)
       1040     )


    ValueError: y should be a 1d array, got an array of shape () instead.



```python

```


```python
from sklearn.linear_model import LogisticRegression

validation_scores = []
train_scores = []

C_range = np.array([.00000001,.0000001,.000001,.00001,.0001,.001,0.1,\
                1,10,100,1000,10000,100000,1000000,10000000,100000000,1000000000])

for c in C_range:
    my_logreg = LogisticRegression(C = c,random_state=1)
    my_logreg.fit(X_train,y_train)
    
    # train on traning set
    train_scores.append(my_logreg.score(X_train,y_train))
    # score on validation set
    validation_scores.append(my_logreg.score(X_validation,y_validation))
```


```python

```


```python

```

#### Get variance
<a id='get_variance'></a>


```python
df2.var()
```

#### Get SD
<a id='get_sd'></a>


```python
df2.std()
```


```python

```

#### Convert to an array
<a id='convert_to_an_array'></a>


```python
toarray()
```


```python

```

#### Bins from minus to plus infinity, with bins in between
<a id='Bins_from_minus_to_plus_infinity_with_bins_in_between'></a>


```python
[-np.inf, 0, 15, 30, np.inf])
```


```python

```

### Model evaluation
<a id='model_evaluation'></a>

We can calculate our precision and recall using the formulas from above:


```python
tp = cf_matrix[1, 1]
predicted_fraud = cf_matrix[:, 1].sum()

precision = tp/predicted_fraud

print(f"Precision = {tp}/{predicted_fraud} = {round(precision*100, 2)}%")
```


```python
tp = cf_matrix[1, 1]
true_fraud = cf_matrix[1, :].sum()

recall = tp/true_fraud

print(f"Recall = {tp}/{true_fraud} = {round(recall*100, 2)}%")
```

The same calculation is carried out by the `precision_score` and `recall_score` functions which only require the true and predicted labels:


```python
# Precision 
from sklearn.metrics import precision_score

# precision_score(true labels, predicted labels)
precision_score(y_test, y_pred)
```


```python
# Recall
from sklearn.metrics import recall_score

# recall_score(true labels, predicted labels)
recall_score(y_test, y_pred)
```

The F1 score is the [harmonic mean](https://en.wikipedia.org/wiki/Harmonic_mean) of the precision and recall scores.
- The F1 score is always between the precision and recall score.
- An F1 score reaches its best value at 1 which corresponds to both perfect precision and perfect recall.
- The worst F1 score is 0 which occurs if either precision or recall becomes zero; this is one advantage of using the harmonic mean over a simple average of precision and recall.

We use the F1 score in order to try and maximize the precision and recall scores simultaneously.

For our example, where we had a 99% accuracy, we get an F1 score of:


```python
from sklearn.metrics import f1_score

f1_score(y_test, y_pred)
```

Finally, the very handy `classification_report` function in `sklearn.metrics` will compute precision, recall, and $F_1$ score for both the positive class (this is how we introduced precision and recall previously) and also the negative class.


```python
from sklearn.metrics import classification_report

report_initial = classification_report(y_test, y_pred)
print(report_initial)
```

#### Optimising hyperparameters with a KNN model
<a id='optimising_hyperparameters_with_a_knn_model'></a>


```python
# 1) Load data
#sonar = 
X = sonar.data
y = sonar.target
```


```python
# 2) Import models, libraries, plotting
from sklearn.datasets import fetch_openml
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
```


```python
# 3) Scale the data
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_validation = scaler.transform(X_validation)
```


```python
# 4) Hyperparameter tuning
```


```python
# 5) Select range of K values, which have been arbitrarily chosen
K_list = range(1, 40, 2)

# Lists for scores
validation_scores = []
train_scores = []
```


```python
# 6) Iterate over each K
for k in K_list:
    # 7) Fit model with K value
    KNN_model = KNeighborsClassifier(n_neighbors = k)
    KNN_model.fit(X_train, y_train)
    # 8) Calculate training and validation accuracy
    train_scores.append(KNN_model.score(X_train,y_train))
    validation_scores.append(KNN_model.score(X_validation,y_validation))
```


```python
# Plot training and validation curves
plt.figure()
plt.plot(K_list, train_scores, label='Train')
plt.plot(K_list, validation_scores, label='Validation')
plt.xlabel('n_neighbours')
plt.ylabel('accuracy')
plt.title('KNN model')
plt.legend(bbox_to_anchor = (1.05, 1), loc='upper left')
plt.show()
```


```python
# Find the best K
(np.abs(np.array(train_scores) - np.array(validation_scores)))
```


```python
# Continued find the best K
K_list[best_idx]
```


```python
# Finalise the model
final_KNN = KNeighborsClassifier(n_neighbors = K_list[best_idx])
final_KNN.fit(X_train, y_train)

final_KNN.score(X_train,y_train)
final_KNN.score(X_test,y_test)
```


```python

```


```python

```


```python

```

## Visualisation
<a id='visualisation'></a>

### Seaborn with Matplotlib
<a id='seaborn'></a>

#### Loop for plot creation
<a id='Pairplot'></a>


```python
# Use a for loop to generate the plots and use subplots this time

# Visualise the 3 numeric cols using a histogram and boxplot
for col in ['Age','BodyweightKg', 'TotalKg']:
    plt.subplots(1, 2, figsize=(20, 5))

    # Plot out the histogram
    plt.subplot(1, 2, 1)
    plt.hist(powerlifting_df[col], bins=60, alpha=0.5)
    plt.title('Histogram')
    plt.xlabel(f'{col}')
    plt.ylabel('frequency')

    # Plot the boxplot. We can use the seaborn boxplot code for this.
    plt.subplot(1, 2, 2)
    sns.boxplot(x=powerlifting_df[col], color="steelblue")
    plt.title('Boxplot')


    plt.show()
```

#### Vertical lines plot
<a id='Pairplot'></a>


```python
# Plot
sns.histplot(x = time_spent, alpha = 0.3)
plt.vlines(120, 0, 25, color = 'red', label = 'Competitor avg')
plt.vlines(time_spent.mean(), 0, 25, color = 'blue', label = 'our avg')

plt.title('User Engagement')
plt.xlabel('Time spent (s)')
plt.legend()
plt.show()
```

#### Boxplot
<a id='Pairplot'></a>


```python
# Boxplot
sns.boxplot(data=df_cleaned, x=column, y=df_cleaned["registered"])
plt.title(column)
plt.show()
```

#### Kernel density plot
<a id='Pairplot'></a>


```python
# Kernel density plot

plt.figure(figsize=(10, 5))
sns.kdeplot(x=powerlifting_df['Age'], y=powerlifting_df['TotalKg'], shade=True, thresh=0.05)
plt.xlabel('Age')
plt.ylabel('TotalKg')
plt.show()
```

#### Pairplot
<a id='Pairplot'></a>


```python
#Seems to plot all of the variables
sns.pairplot(df_cleaned_num)
```

#### Histogram
<a id='Heatmap'></a>


```python
sns.histplot(data = clean_data, x = 'age', hue = 'registered')
plt.show()
```

#### Heatmap
<a id='Heatmap'></a>


```python
# Heatmap
sns.heatmap(diamonds.corr(), cmap='coolwarm')
plt.show()
```


```python
# Correlation of the variables in a heatmap
plt.figure(figsize=(20, 10))
matrix = np.triu(X.corr()) #Gives only one half, to avoid repetition
sns.heatmap(X.corr(), annot=True, mask=matrix, cmap='coolwarm')
plt.show()
```


```python
# correlation
df_cleaned_num_corr = df_cleaned_num.corr().round(2)

plt.figure(figsize=(10, 6))
sns.heatmap(df_cleaned_num_corr, vmin=-1, vmax=1, cmap="BuPu", annot=True)
plt.show()
```

##### Multi-collinearity heatmap
<a id='multi_collinearity_heatmap'></a>


```python
# Heatmap
corr_df = X.corr()

# TRIANGLE MASK
mask = np.triu(corr_df)
# heatmap
plt.figure(figsize = (20, 20))
sns.heatmap(corr_df.round(2), annot = True, vmax = 1, vmin = -1, center = 0, cmap = 'Spectral', mask = mask)
plt.show()
```


```python

```


```python

```

#### Countplot
<a id='Countplot'></a>


```python
# Countplot
plt.figure()
sns.countplot(x=powerlifting_df['Sex'])
plt.title("Bar Plot (Count)")
plt.show()
```

### Matplotlib
<a id='matplotlib'></a>

#### Bar plot with groupby
<a id='bar_plot_with_groupby'></a>


```python
train_weather.groupby(['month'])['wnvpresent'].sum().plot.bar(color='r')
plt.title("West Nile Virus Detection by months", fontsize=20)
plt.xlabel("Months", fontsize =14)
plt.ylabel("West Nile Virus Detections", fontsize=14)
plt.xticks(rotation=0)
```


```python
# Varioable count by years and month
train.groupby(['year','month']).nummosquitos.sum().unstack(fill_value=0).plot.bar(figsize=(10,7))
plt.xticks(rotation=0, fontsize=14)
plt.yticks(fontsize=14)
plt.title('Mosquitos Count by Year & Month', fontsize=14)
plt.ylabel('Count', fontsize=14)
plt.xlabel('Year',fontsize=14)
plt.legend(fontsize=14)
plt.show()
```


```python
#Plots the above
clean_data.groupby('registered')['last_contact_duration'].agg(['mean', 'median']).plot(kind = 'bar')
plt.show()
```

#### Stacked bar chart
<a id='matplotlib'></a>


```python
rate_data.unstack()[['yes', 'no']].sort_values('yes').plot(kind = 'barh', stacked = True)
plt.title(f"Registration rate per {column} type")
plt.show()
```

#### Scatter graph
<a id='matplotlib'></a>


```python
# using the size parameter to help readability
plt.scatter(powerlifting_df['Age'], powerlifting_df['TotalKg'], s=2, )
plt.xlabel('Age')
plt.ylabel('TotalKg')
plt.show()
```

#### Scatter graph
<a id='scatter_graph'></a>


```python
#Plots the above arrays on a scatter graph
plt.figure()
plt.scatter(apples, pears)
plt.xlabel('Apples prices')
plt.ylabel('Pears prices')
plt.title('Prices for apples and pears')
plt.show()
```

#### Lines for variables
<a id='lines_for_variables'></a>


```python
# add lines for each column
fig = px.line(air_traffic, x=air_traffic.index, y=air_traffic.columns,)
```

#### Slider
<a id='slider'></a>


```python
# activate slider
fig.update_xaxes(rangeslider_visible=True)
```

#### Seasonal plot
<a id='seasonal_plot'></a>


```python
# Seasonal plot
from statsmodels.graphics.tsaplots import month_plot

plt.figure(figsize=(15, 5))

# create the seasonal plot
month_plot(air_traffic_monthly["Revenue Passenger Miles"], ax=plt.gca())

plt.title("Seasonal Revenue Passenger Miles per Month")
sns.despine()
plt.show()
```

#### Add decision boundaries
<a id='add_decision_boundaries'></a>


```python
#Add some decision boundaries
divide_three, = ax.plot([7,-7], [-10,10], dashes=[6, 2], color = 'green')

plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, s=20, edgecolor='k')
plt.show()
```

#### Three dimensional data
<a id='three_dimensional_data'></a>


```python
# Three dimensional graph
import plotly.graph_objects as go

scatter_data = go.Scatter3d(x=X[:, 0], y=X[:, 1], z=x3, mode="markers", marker={"color": y, "size":0.5})


fig = go.Figure(data=[scatter_data])
fig.update_layout(scene = dict(xaxis_title='x1',
                               yaxis_title='x2',
                               zaxis_title='x3'))
fig.show()
```

#### Vertical lines across axes
<a id='three_dimensional_data'></a>


```python
#Shows the distribution of the last contact duration column
plt.figure()

#Gives lines on respective stats
plt.axvline(clean_data['last_contact_duration'].mean(), color = 'red', label = 'Mean')
plt.axvline(clean_data['last_contact_duration'].median(), color = 'yellow', label = 'Median')

#Gives labelling
plt.xlim(0, 2500) #Limits to 2,500
plt.xlabel('Last contact duration')
plt.ylabel('Count')
plt.title('Most calls are short')

#Shows with select bins
plt.hist(clean_data['last_contact_duration'], bins = 30)
plt.show()

#Perhaps shows an exponential distribution
```

#### Move plot legend
<a id='move_plot_legend'></a>


```python
# Put the legend in a plot at any place
plt.legend(bbox_to_anchor = (1.05, 1), loc='upper left')
```

#### Subplot grid
<a id='subplot_grid'></a>


```python
# Set up a subplot grid
plt.subplots(2,2, figsize=(10,8))

# Iterate over the datasets and their indices
for i, dataset in enumerate(['I','II','III','IV']):

    # Plot each dataset
    plt.subplot(2,2,i+1)
    plt.scatter(
        anscombe[anscombe['Dataset'] == dataset]['X'],
        anscombe[anscombe['Dataset'] == dataset]['Y']
    )
    plt.title(f'Dataset {dataset}')

plt.show()
```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python
# Ignore futurewarnings
import warnings
warnings.filterwarnings('ignore')
```


```python

```


```python

```


```python

```

## Miscellanea
<a id='miscellanea'></a>


```python
# Mark's code

confirmed_deaths_ts_transposed = confirmed_deaths_ts.drop(['country_name', 'region_name', 'Unnamed: 0', 'jurisdiction'], axis = 1).set_index(['country_code', 'region_code']).T
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    /var/folders/3w/7b2j0crd0hvc3bb28cmq6j_40000gn/T/ipykernel_2474/3744389534.py in <module>
          1 # Mark's code
          2 
    ----> 3 confirmed_deaths_ts_transposed = confirmed_deaths_ts.drop(['country_name', 'region_name', 'Unnamed: 0', 'jurisdiction'], axis = 1).set_index(['country_code', 'region_code']).T
    

    NameError: name 'confirmed_deaths_ts' is not defined



```python
# Sets the latitude and longitude entries for all of the hotels missing such information

# Sets 'lat' and 'lng' for 20 Rue De La Ga t 14th arr 75014 Paris France
lat_lon_missing.loc[lat_lon_missing['Hotel_Address'] == '20 Rue De La Ga t 14th arr 75014 Paris France', 'lat'] = 48.832760
lat_lon_missing.loc[lat_lon_missing['Hotel_Address'] == '20 Rue De La Ga t 14th arr 75014 Paris France', 'lng'] = 2.324740

# Sets 'lat' and 'lng' for 23 Rue Damr mont 18th arr 75018 Paris France
lat_lon_missing.loc[lat_lon_missing['Hotel_Address'] == '23 Rue Damr mont 18th arr 75018 Paris France', 'lat'] = 48.892399
lat_lon_missing.loc[lat_lon_missing['Hotel_Address'] == '23 Rue Damr mont 18th arr 75018 Paris France', 'lng'] = 2.344990

# Sets 'lat' and 'lng' for 4 rue de la P pini re 8th arr 75008 Paris France
lat_lon_missing.loc[lat_lon_missing['Hotel_Address'] == '4 rue de la P pini re 8th arr 75008 Paris France', 'lat'] = 48.877708
lat_lon_missing.loc[lat_lon_missing['Hotel_Address'] == '4 rue de la P pini re 8th arr 75008 Paris France', 'lng'] = 2.316550

# Sets 'lat' and 'lng' for Bail n 4 6 Eixample 08010 Barcelona Spain
lat_lon_missing.loc[lat_lon_missing['Hotel_Address'] == 'Bail n 4 6 Eixample 08010 Barcelona Spain', 'lat'] = 41.391420
lat_lon_missing.loc[lat_lon_missing['Hotel_Address'] == 'Bail n 4 6 Eixample 08010 Barcelona Spain', 'lng'] = 2.175210

# Sets 'lat' and 'lng' for Gr nentorgasse 30 09 Alsergrund 1090 Vienna Austria
lat_lon_missing.loc[lat_lon_missing['Hotel_Address'] == 'Gr nentorgasse 30 09 Alsergrund 1090 Vienna Austria', 'lat'] = 48.224480
lat_lon_missing.loc[lat_lon_missing['Hotel_Address'] == 'Gr nentorgasse 30 09 Alsergrund 1090 Vienna Austria', 'lng'] = 16.354010

# Sets 'lat' and 'lng' for Hasenauerstra e 12 19 D bling 1190 Vienna Austria
lat_lon_missing.loc[lat_lon_missing['Hotel_Address'] == 'Hasenauerstra e 12 19 D bling 1190 Vienna Austria', 'lat'] = 48.240460
lat_lon_missing.loc[lat_lon_missing['Hotel_Address'] == 'Hasenauerstra e 12 19 D bling 1190 Vienna Austria', 'lng'] = 16.348710

# Sets 'lat' and 'lng' for Josefst dter Stra e 10 12 08 Josefstadt 1080 Vienna Austria
lat_lon_missing.loc[lat_lon_missing['Hotel_Address'] == 'Josefst dter Stra e 10 12 08 Josefstadt 1080 Vienna Austria', 'lat'] = 48.210430
lat_lon_missing.loc[lat_lon_missing['Hotel_Address'] == 'Josefst dter Stra e 10 12 08 Josefstadt 1080 Vienna Austria', 'lng'] = 16.344380

# Sets 'lat' and 'lng' for Josefst dter Stra e 22 08 Josefstadt 1080 Vienna Austria
lat_lon_missing.loc[lat_lon_missing['Hotel_Address'] == 'Josefst dter Stra e 22 08 Josefstadt 1080 Vienna Austria', 'lat'] = 48.210430
lat_lon_missing.loc[lat_lon_missing['Hotel_Address'] == 'Josefst dter Stra e 22 08 Josefstadt 1080 Vienna Austria', 'lng'] = 16.344380

# Sets 'lat' and 'lng' for Landstra er G rtel 5 03 Landstra e 1030 Vienna Austria
lat_lon_missing.loc[lat_lon_missing['Hotel_Address'] == 'Landstra er G rtel 5 03 Landstra e 1030 Vienna Austria', 'lat'] = 48.201740
lat_lon_missing.loc[lat_lon_missing['Hotel_Address'] == 'Landstra er G rtel 5 03 Landstra e 1030 Vienna Austria', 'lng'] = 16.391590

# Sets 'lat' and 'lng' for Paragonstra e 1 11 Simmering 1110 Vienna Austria
lat_lon_missing.loc[lat_lon_missing['Hotel_Address'] == 'Paragonstra e 1 11 Simmering 1110 Vienna Austria', 'lat'] = 48.174280
lat_lon_missing.loc[lat_lon_missing['Hotel_Address'] == 'Paragonstra e 1 11 Simmering 1110 Vienna Austria', 'lng'] = 16.416430

# Sets 'lat' and 'lng' for Pau Clar s 122 Eixample 08009 Barcelona Spain
lat_lon_missing.loc[lat_lon_missing['Hotel_Address'] == 'Pau Clar s 122 Eixample 08009 Barcelona Spain', 'lat'] = 41.393820
lat_lon_missing.loc[lat_lon_missing['Hotel_Address'] == 'Pau Clar s 122 Eixample 08009 Barcelona Spain', 'lng'] = 2.169310

# Sets 'lat' and 'lng' for Savoyenstra e 2 16 Ottakring 1160 Vienna Austria
lat_lon_missing.loc[lat_lon_missing['Hotel_Address'] == 'Savoyenstra e 2 16 Ottakring 1160 Vienna Austria', 'lat'] = 48.213200
lat_lon_missing.loc[lat_lon_missing['Hotel_Address'] == 'Savoyenstra e 2 16 Ottakring 1160 Vienna Austria', 'lng'] = 16.310830

# Sets 'lat' and 'lng' for Sep lveda 180 Eixample 08011 Barcelona Spain
lat_lon_missing.loc[lat_lon_missing['Hotel_Address'] == 'Sep lveda 180 Eixample 08011 Barcelona Spain', 'lat'] = 41.384270
lat_lon_missing.loc[lat_lon_missing['Hotel_Address'] == 'Sep lveda 180 Eixample 08011 Barcelona Spain', 'lng'] = 2.159630

# Sets 'lat' and 'lng' for Sieveringer Stra e 4 19 D bling 1190 Vienna Austria
lat_lon_missing.loc[lat_lon_missing['Hotel_Address'] == 'Sieveringer Stra e 4 19 D bling 1190 Vienna Austria', 'lat'] = 48.240461
lat_lon_missing.loc[lat_lon_missing['Hotel_Address'] == 'Sieveringer Stra e 4 19 D bling 1190 Vienna Austria', 'lng'] = 16.348711

# Sets 'lat' and 'lng' for Taborstra e 8 A 02 Leopoldstadt 1020 Vienna Austria
lat_lon_missing.loc[lat_lon_missing['Hotel_Address'] == 'Taborstra e 8 A 02 Leopoldstadt 1020 Vienna Austria', 'lat'] = 48.218021
lat_lon_missing.loc[lat_lon_missing['Hotel_Address'] == 'Taborstra e 8 A 02 Leopoldstadt 1020 Vienna Austria', 'lng'] = 16.390360

# Sets 'lat' and 'lng' for W hringer Stra e 12 09 Alsergrund 1090 Vienna Austria
lat_lon_missing.loc[lat_lon_missing['Hotel_Address'] == 'W hringer Stra e 12 09 Alsergrund 1090 Vienna Austria', 'lat'] = 48.224481
lat_lon_missing.loc[lat_lon_missing['Hotel_Address'] == 'W hringer Stra e 12 09 Alsergrund 1090 Vienna Austria', 'lng'] = 16.354011

# Sets 'lat' and 'lng' for W hringer Stra e 33 35 09 Alsergrund 1090 Vienna Austria
lat_lon_missing.loc[lat_lon_missing['Hotel_Address'] == 'W hringer Stra e 33 35 09 Alsergrund 1090 Vienna Austria', 'lat'] = 48.224482
lat_lon_missing.loc[lat_lon_missing['Hotel_Address'] == 'W hringer Stra e 33 35 09 Alsergrund 1090 Vienna Austria', 'lng'] = 16.354012
```
