# End-to-end-ML-Data-Modelling-Pipelines
## Define Problem:
Here we will build a predictive model that answers the question: “what sorts of people were more likely to survive?” using passenger data (ie name, age, gender, socio-economic class, etc).
## Step 1: Load all required Libraries
```python
import pandas as pd
import numpy as np
from pandas import set_option
from pandas.plotting import scatter_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import Perceptron
from xgboost import XGBClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import SGDClassifier
from matplotlib import pyplot
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn import linear_model
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
import featuretools as ft
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import ExtraTreesRegressor
```

## Step 2: Loading Data into Dataframe
Machine learning algorithms need data. You can load your own data from business database tables or CSV / excel files.
* Load data from Netezza DB, using below standard steps.
		1 Input password for netezza db connection and encrypt it.
	```python
	def PasswordEncrypt():
    '''
    Used to encrypt password and store in user home folder.  getpass used to
    mask password on entry (note: this does not mask on all environments, such
    as QtConsole)
    '''
    import getpass

    myPassword = getpass.getpass('Enter the password you wish to encrypt: ')
    myPassword = myPassword.encode('utf-8')  # must be bytes
    

    # for writing text file to home folder
    from pathlib import Path
    home = str(Path.home())

    # first, generate a random key, save to file
    from cryptography.fernet import Fernet
    key = Fernet.generate_key()
    with open(home + '/PythonKey.bin', 'wb') as file_object:
        file_object.write(key)

    # use the key above to encrypt/decrypt your password
    cipher_suite = Fernet(key)
    ciphered_text = cipher_suite.encrypt(myPassword)

    # write encrypted password to binary file
    with open(home + '/PythonPWD.bin', 'wb') as file_object:
        file_object.write(ciphered_text)

	# execute function
	PasswordEncrypt()
	```
	2.Netezza DB connection creation.
	```python
	# Import libraries necessary for this project
	import numpy as np
	import pandas as pd
	from time import time
	from IPython.display import display # Allows the use of display() for DataFrames
	#Netezza database connection setup
	import pyodbc
	import math
	print (pyodbc.dataSources())
	# copy this block to connect to an Insight database
	# Specify Insight DB to connect to below: DM_RISKS, DM_CUSTOMER, DM_RMS
	database = 'BI_ANALYTICS'
	# ********* Please provide the user ID ********** #
	user_id = ''
	home = str(Path.home())
	   # retreive encrypted password
	   with open(home + '/PythonPWD.bin', 'rb') as file_object:
	       for line in file_object:
	           encryptedpwd = line
	   # retrieve key
	   with open(home + '/PythonKey.bin', 'rb') as file_object:
	       for line in file_object:
	           key = line
	   # decrypt
	   cipher_suite = Fernet(key)
	   uncipher_text = (cipher_suite.decrypt(encryptedpwd))
	   password = bytes(uncipher_text).decode('utf-8') #convert to string
	   # Create connection to DB
	   db_conn_str = 'DRIVER={NetezzaSQL};SERVER='';DATABASE=' + database + ';UID=' + user_id + ';PWD=' + password
	   db_connection = pyodbc.connect(db_conn_str)
	   c = db_connection.cursor()    
	   return (db_connection, c)
	cnxn, cursor = connectInsight(database)  
	```	
* Data Pull from csv / excel file using CSV.reader() / numpy.loadtxt() / pandas.read csv() function
* Data Summary [Model Cohort]
```python
import pandas as pd
df=pd.read_csv("titanic.csv",header=0)
```
### Attribute Information: 
*PassengerId* : Sequential Id assigned to the Passenger
*Survived* :    (0/1) 0 => Passenger did not survived ,1 => Passenger survived   <Target Variable>
*Pclass* : Class of the Passenger
*Name* :    Name of the Passenger
*Sex* :    Geneder of the Passenger
*Age* :    Age of the Passenger
*SibSp* :    Sibling and Spouse Count of the Passenger
*Parch* :    Parent and Child Count of the Passenger
*Ticket* :    Ticket number of the passenger
*Fare* :    Ticket Fare of the Passsenger
*Cabin* :    Type of Cabin assigned to the passenger
*Embarked* :  Port of Embarkation (C = Cherbourg; Q = Queenstown; S = Southampton)

### Insight
-   **‘Survived’**  is the target variable, which we will predict once our preprocessing of our data is done. So, we retain that column.
-   Only the columns such as  **‘Age’, ‘Cabin’ and ‘Embarked’**  has missing values.
-   **‘PassengerId’, ‘Name’ and ‘Ticket’** doesn’t add much value in predicting the target variable.
-   **‘ParCh’**(Parent/Children) **and ‘SibSp’**(Siblings/Spouse) details are related to family, so we can derive a new column named  **‘Size of the family’**
-   **‘Sex’, ‘Cabin’ and ‘Embarked’**  are the categorical data that needs to be encoded to numerical values.

## Step 2. Dropping of columns
In this step, we are going to drop columns with the least priority. The column such as ‘PassengerId’, 'Name' and ‘Ticket’ comes under this category.
```python
df.drop(columns = ['PassengerId','Name','Ticket'],axis=1,inplace=True)
````
### Target variable distribution
```python
c = df.Survived.value_counts(dropna=False)
p = df.Survived.value_counts(dropna=False, normalize=True)
pd.concat([c,p], axis=1, keys=['counts', '%']).to_excel("Target_Variable_Distribution.xlsx", header=True)
```
## Creating new interaction variables
```python
df['Dependent_ind']='0'
df['Parent_child_ind']='0'
df['Sibling_Spouse_ind']='0'
df['Child_ind']='0'
df['Embarked_ind']='0'

df.loc[df['Parch']>0,'Dependent_ind']='1'
df.loc[df['SibSp']>0,'Dependent_ind']='1'
df.loc[df['Parch']>0,'Parent_child_ind']='1'
df.loc[df['SibSp']>0,'Sibling_Spouse_ind']='1'
df.loc[df['Age']<16,'Child_ind']='1'
df.loc[df['Embarked']=='C','Embarked_ind']='1'

df = df[['Survived','Pclass','Sex','Age','Child_ind','SibSp','Parch','Dependent_ind','Parent_child_ind','Sibling_Spouse_ind','Fare','Cabin','Embarked','Embarked_ind']]
````
## Splitting Continuous and Categorical types variables
```python
cols=df.columns
num_columns=df._get_numeric_data().columns
cat_columns=list(set(cols) - set(num_columns))
print("Total Columns : " + str(len(cols)))
df_categorical = df[cat_columns]
df_continuous = df[num_columns]
````
## Step 3: Analyze Data
Once you have loaded your data into Python you need to be able to understand it. The better
you can understand your data, the better and more accurate the models that you can build.
The rst step to understanding your data is to use descriptive statistics.
Helper functions provided on the Pandas Data Frame.

 Understand your data using the head() function to look at the rst few rows.

 Review the dimensions of your data with the shape property.

 Look at the data types for each attribute with the dtypes property.

 Review the distribution of your data with the describe() function.

 Calculate pairwise correlation between your variables using the corr() function.

### Descriptive Statistics
**Univariate Statistics of continuous variables**
Count,Mean, Median, Standard Deviation, Quartile, Maximum,Minimum, % of missing values, 
```python
''' Univariate Statistic '''
df_univariate = df_continuous.drop(target_var, axis=1).describe(percentiles=[.25,.5,.75,.9,.99]).T

''' Correlation - Target versus individual continous features '''
df_corr = pd.DataFrame(df_continuous.drop(target_var, axis=1).apply(lambda x: x.corr(df_continuous.eval(target_var))))

''' Missing Value Information '''
df_nmis = pd.DataFrame(df_continuous.drop(target_var, axis=1).isnull().sum())

''' Concatenate Univariate,Missing and Correlation Information '''
df_univariate = pd.concat([df_univariate,df_nmis,df_corr], axis=1).reset_index()
df_univariate.columns=['variable','obs','mean','stdv','min','p25','median','p75','p90','p99','max','nmiss','corr']

''' Calculating Missing Percentage '''
df_univariate['missing %']=df_univariate['nmiss']/len(df_continuous)

''' Write to excel '''
select_metrics=['variable','obs','missing %','min','max','mean','median','stdv','p25','p75','p90','p99','corr']
df_univariate[select_metrics].round(2).to_excel(out_file,index = None, header=True)
```
**Weight of Evidence(WOE) and Information Value(IV) Calculation**
Weight of Evidence (WoE) describes the relationship between a predictor and a binary dependent variable. Information Value (IV) is the measurement of that relationship’s power. Based on its role, IV can be used as a base for attributes selection.
The weight of evidence tells the predictive power of an independent variable in relation to the dependent variable.
Since it evolved from credit scoring world, it is generally described as a measure of the separation of good and bad customers. "Bad Customers" refers to the customers who defaulted on a loan. and "Good 
Customers" refers to the customers who paid back loan.

![](https://i.imgur.com/hrTnEXv.jpg)

The concept of WOE in terms of events and non-events. It is calculated by taking the natural logarithm (log to base e) of division of % of non-events and % of events.

![](https://i.imgur.com/dfosGgc.jpg)

**Weight of Evidence (WOE)**  helps to transform a continuous independent variable into a set of groups or bins based on similarity of dependent variable distribution i.e. number of events and non-events.

**For continuous independent variables**  : First, create bins (categories / groups) for a continuous independent variable and then combine categories with similar WOE values and replace categories with WOE values. Use WOE values rather than input values in your model.

**For categorical independent variables**  : Combine categories with similar WOE and then create new categories of an independent variable with continuous WOE values. In other words, use WOE values rather than raw categories in your model. The transformed variable will be a continuous variable with WOE values. It is same as any continuous variable.

#### Why combine categories with similar WOE?[](https://www.kaggle.com/pavansanagapati/weight-of-evidence-woe-information-value-iv#Why-combine-categories-with-similar-WOE?)

It is because the categories with similar WOE have almost same proportion of events and non-events. In other words, the behavior of both the categories is same.

#### Rules related to WOE[](https://www.kaggle.com/pavansanagapati/weight-of-evidence-woe-information-value-iv#Rules-related-to-WOE)

-   Each category (bin) should have at least 5% of the observations.
-   Each category (bin) should be non-zero for both non-events and events.
-   The WOE should be distinct for each category. Similar groups should be aggregated.
-   The WOE should be monotonic, i.e. either growing or decreasing with the groupings.
-   Missing values are binned separately.

#### Number of Bins (Groups)[](https://www.kaggle.com/pavansanagapati/weight-of-evidence-woe-information-value-iv#Number-of-Bins-(Groups))

In general, 10 or 20 bins are taken. Ideally, each bin should contain at least 5% cases. The number of bins determines the amount of smoothing - the fewer bins, the more smoothing. If someone asks you ' "why not to form 1000 bins?" The answer is the fewer bins capture important patterns in the data, while leaving out noise. Bins with less than 5% cases might not be a true picture of the data distribution and might lead to model instability.

#### Handle Zero Event/ Non-Event[](https://www.kaggle.com/pavansanagapati/weight-of-evidence-woe-information-value-iv#Handle-Zero-Event/-Non-Event)

If a particular bin contains no event or non-event, you can use the formula below to ignore missing WOE. We are adding 0.5 to the number of events and non-events in a group.

AdjustedWOE = ln (((Number of non-events in a group + 0.5) / Number of non-events)) / ((Number of events in a group + 0.5) / Number of events))

#### How to check correct binning with WOE[](https://www.kaggle.com/pavansanagapati/weight-of-evidence-woe-information-value-iv#How-to-check-correct-binning-with-WOE)

1.  The WOE should be monotonic i.e. either growing or decreasing with the bins. You can plot WOE values and check linearity on the graph.
    
2.  Perform the WOE transformation after binning. Next, we run logistic regression with 1 independent variable having WOE values. If the slope is not 1 or the intercept is not ln(% of non-events / % of events) then the binning algorithm is not good.
    

Both dummy coding and WOE transformation give the similar results. The choice which one to use mainly depends on personal preferences.

In general optimal binning, dummy coding and weight of evidence transformation are, when carried out manually, time-consuming processes.

**WOE Advantage**:

The advantages of WOE transformation are

**1. Handles missing values**

**2. Handles outliers**

**3. The transformation is based on logarithmic value of distributions. This is aligned with the logistic regression output function**

**4. No need for dummy variables**

**5. By using proper binning technique, it can establish monotonic relationship (either increase or decrease) between the independent and dependent variable**

**WOE Disadvantage**:

The main disadvantage of WOE transformation is

**- in only considering the relative risk of each bin, without considering the proportion of accounts in each bin. The information value can be utilised instead to assess the relative contribution of each bin.**

### Information Value (IV)
Information value is one of the most useful technique to select important variables in a predictive model. It helps to rank variables on the basis of their importance. The IV is calculated using the following formula :
![](https://i.imgur.com/r6ACeFN.jpg)

**IV statistic can be interpreted as follows.**

![](https://i.imgur.com/cZx3taD.jpg)

If the IV statistic is:

-   Less than 0.02, then the predictor is not useful for modeling (separating the Goods from the Bads)
-   0.02 to 0.1, then the predictor has only a weak relationship to the Goods/Bads odds ratio
-   0.1 to 0.3, then the predictor has a medium strength relationship to the Goods/Bads odds ratio
-   0.3 to 0.5, then the predictor has a strong relationship to the Goods/Bads odds ratio.
-   0.5, suspicious relationship (Check once)

#### Binning of Continuous/Categorical Variables and generate Weight of Evidence and Information Value
**create_volume_group()** This User defined function creates bins on the basis of parameter n_bin (number of bins) provided. This algorithm creates almost eqi_volume groups with unique values in groups :

1. It calculates the Average Bin Volume by dividing the total volume of data by number of bins.
2. It sorts the data based on the value of the continuous variables.
3. It directly moves to index (I1) having value Average Bin Volume (ABV).
4. It checks the value of continous variable at the index position decided in previous step.
5. It finds the index(I2) of last position of the value identified at previous step 4.
6. It concludes the data of the First Bin within the range (0 to I2).
7. The Index I1 is again calculated as I1 = I2 + ABV and step 4-6 is repeated
8. This Process is continued till the desired number of bins are created
9. Seperate Bin is created if the continuous variable is having missing value

Note : qcut() does provide equi-volume groups but does not provide unique values in the groups.
hence,qcut() is not used in binning the data. 

**cont_bin_Miss()** Creates BINS for continous variables having some Missing values. 'The Missing Values have been grouped together as a seperate bin - "Missing".

**cont_bin_NO_Miss()** Creates BINS for continuous variables having no missing values'.

**cat_bin_trend()** Creates the bins / groups on the 'Bins' of the Categorical Columns
1. Event -> Target = 1
2. Non-Event -> Target = 0
3. ALong with the Bins of the categorical Columns, A summary record is also created with header "Total".

**bin_data()** This User Defined Function performs the following :
4. Replace the NAN value with "Missing" value.
5. Binning is done on the unique Labels of the Categorical columns.
6. For Missing Values, it has been treated as seperate Label - "Missing".
7. Calculates Weight of Evidence (WOE) of each bin/label of Categorical Variables.
8. Calculates Information Vale (IV) for each Categorical Variables.

**Automate_woe_population()** Creates a new field with suffix "_WOE" and gets populated with Weight of Evidence as obtained for each 'Bins' of the Categorical Variables.

**calc_iv** Calculates the Wieght of Evidence (WOE) and Information Value(IV) for Categorical fields.

### Descriptive statistics with PLOT
Binning of Continuous/Categorical Variables and generate Rank and Plot of the same
performs Binning of Continuous/Categorical Variables and generate Rank and Plot of the same.

**plot_stat()** Do below 2 steps.
1. Line Plot showing Volume% and Event% against Bins/Bins.
2. Scatter Plot showing Target against Bins/Bins.

**add_table_plot()** Adds Line Plot and Scatter Plot in an excel
just on the beside of the binning data of the continuous/categorical Fields. This gives better readability in analyzing the data.

 Use the hist() function to create a histogram of each attribute.

 Use the plot(kind='box') function to create box and whisker plots of each attribute.

 Use the pandas.scatter matrix() function to create pairwise scatter plots of all at-
tributes.

A fundamental task in many statistical analyses is to characterize the location and variability of a data set. A further characterization of the data includes  **skewness and kurtosis**.

**Skewness**  is a measure of symmetry, or more precisely, the lack of symmetry. A distribution, or data set, is symmetric if it looks the same to the left and right of the center point.

**Kurtosis**  is a measure of whether the data are heavy-tailed or light-tailed relative to a normal distribution. That is, data sets with high kurtosis tend to have heavy tails, or outliers. Data sets with low kurtosis tend to have light tails, or lack of outliers. A uniform distribution would be the extreme case.
```python
from scipy.stats import skew
from scipy.stats import kurtosis
def plotBarCat(df,feature,target):
    
    
    
    x0 = df[df[target]==0][feature]
    x1 = df[df[target]==1][feature]

    trace1 = go.Histogram(
        x=x0,
        opacity=0.75
    )
    trace2 = go.Histogram(
        x=x1,
        opacity=0.75
    )

    data = [trace1, trace2]
    layout = go.Layout(barmode='overlay',
                      title=feature,
                       yaxis=dict(title='Count'
        ))
    fig = go.Figure(data=data, layout=layout)

    py.iplot(fig, filename='overlaid histogram')
    
    def DescribeFloatSkewKurt(df,target):
        """
 A fundamental task in many statistical analyses is to characterize
 the location and variability of a data set. A further
 characterization of the data includes skewness and kurtosis.
 Skewness is a measure of symmetry, or more precisely, the lack
 of symmetry. A distribution, or data set, is symmetric if it
 looks the same to the left and right of the center point.
 Kurtosis is a measure of whether the data are heavy-tailed
 or light-tailed relative to a normal distribution. That is,
 data sets with high kurtosis tend to have heavy tails, or
 outliers. Data sets with low kurtosis tend to have light
 tails, or lack of outliers. A uniform distribution would
 be the extreme case
 """
        print('-*-'*25)
        print("{0} mean : ".format(target), np.mean(df[target]))
        print("{0} var  : ".format(target), np.var(df[target]))
        print("{0} skew : ".format(target), skew(df[target]))
        print("{0} kurt : ".format(target), kurtosis(df[target]))
        print('-*-'*25)
    
    DescribeFloatSkewKurt(df,target)
```
## Prepare For Modeling by Pre-Processing Data

Sometimes you need to pre-process your data in order to best present the inherent structure of the problem in your data to the modeling algorithms. The scikit-learn library provides two standard idioms for transforming data. Each are useful inherent circumstances: Fit and Multiple Transform and Combined Fit-And-Transform.

 Standardize numerical data (e.g. mean of 0 and standard deviation of 1) using the scale
and center options.

 Normalize numerical data (e.g. to a range of 0-1) using the range option.

 Explore more advanced feature engineering such as Binarizing.

### 1. Rescale Data
Two of the most popular ways to rescale data are data normalization and data standardization.
**Normalization**: Data normalization in machine learning consists of rescaling the values of all features such that they lie in a range between 0 and 1  and have a maximum length of one. This serves the purpose of equating attributes of different scales.

The following equation allows you to normalize the values of a feature:

![](https://static.packt-cdn.com/products/9781789803556/graphics/B12714_01_15.jpg)
Here, _zi_ corresponds to the _ith_ normalized value and _x_ represents all values.
Using the **age** variable that was created in the first exercise of this notebook, normalize the data using the preceding formula and store it in a new variable called **age_normalized**. Print out the top 10 values:
```python
age_normalized = (age - age.min())/(age.max()-age.min())
age_normalized.head(10)
```
![](https://static.packt-cdn.com/products/9781789803556/graphics/B12714_01_17.jpg)

**Standardization**: This is a rescaling technique that transforms the  data into a Gaussian distribution with a mean equal to 0 and a standard deviation equal to 1.

One simple way of standardizing a feature is shown in the following equation:

![](https://static.packt-cdn.com/products/9781789803556/graphics/B12714_01_16.jpg)
Here, _zi_ corresponds to the _ith_ standardized value, and _x_ represents all values.
```python
age_standardized = (age - age.mean())/age.std()
age_standardized.head(10)
```
Different than normalization, in standardization, the values distribute normally around zero.
### 2. Binarize Data (Make Binary): 
You can transform your data using a binary threshold. All values above the threshold are marked 1 and all equal to or below are marked as 0. It can be useful when you have probabilities that you want to make crisp values.
```python
# binarization
from sklearn.preprocessing import Binarizer
binarizer  =  Binarizer(threshold=0.0).fit(X)
binaryX  =  binarizer.transform(X)
```
### Missing Treatment
**Number of Missing Values in each column**
To get the number of missing values in each column, we can use pandas [**_isnull()_**](http://pandas.pydata.org/pandas-docs/version/0.24/reference/api/pandas.DataFrame.isnull.html#pandas.DataFrame.isnull) or [**_notnull()_**](http://pandas.pydata.org/pandas-docs/version/0.24/reference/api/pandas.DataFrame.notnull.html#pandas.DataFrame.notnull) methods. **isnull()** method returns an boolean value **TRUE** when null value is present whereas **notnull()** method returns an boolean value **TRUE**, if there is no null value for every observation in the data. We will add them so that it gives the no of null or non-null values in both cases.
```python
df_missing=0
def missing_values_table(df_phase3):
    mis_val = df_phase3.isnull().sum()
    mis_val_percent = 100*df_phase3.isnull().sum()/len(df_phase3)
    mis_val_table = pd.concat([mis_val, mis_val_percent], axis = 'columns')
    mis_val_table_ren_columns = mis_val_table.rename(columns ={0: 'Count of Missing Values', 1: '% of Total Values'} )

    #Sort the table by percent of missing values descending
    mis_val_table_ren_columns = mis_val_table_ren_columns[mis_val_table_ren_columns.iloc[:, 1] != 0 ].sort_values(by='% of Total Values', ascending = False).round(1)
    mis_val_table_ren_columns = mis_val_table_ren_columns.reset_index()
    #print summary
    #print("The dataset has " + str(df_phase3.shape[0]) + " rows and " + str(df_phase3.shape[1]) + " columns." )
    #print(str(mis_val_table_ren_columns.shape[0]) )

    return mis_val_table_ren_columns;
    
df_missing=missing_values_table(X)    
```
![](https://miro.medium.com/max/479/1*93aVbjW7Kg9tpmPwSxtY9Q.png)
### **_Numerical Data_**
Missing values are present in **_Age_**, **_Cabin_** and **_Embarked_** columns. **Age** is of **Numerical** Datatype and **_Cabin_**, **_Embarked_** are of **Object** Datatype which may be string or a character. Now we are going to look how to handle Missing Values in colums which are of Numerical Datatype.
**Imputing with Mean/Median/Mode/Backward fill/ Forward fill**
Null values are replaced with mean/median.mode in this method. This is the statistical method of handling Null values.
The mean of the numerical column data is used to replace null values when the data is normally distributed. Median is used if the data comprised of outliers. Mode is used when the data having more occurrences of a particular value or more frequent value.
```python
#Replace Null Values (np.nan) with mean
df['Age'] = df['Age'].replace(np.nan, df['Age'].mean())
#Alternate way to fill null values with mean
df['Age'] = df['Age'].fillna(df['Age'].mean())
#Checking for null values in Age column
df['Age'].isnull().sum()
#In the same way we can impute using median and mode
df['Age'] = df['Age'].replace(np.nan, df['Age'].median())
df['Age'] = df['Age'].replace(np.nan, df['Age'].mode())
#Alternate ways to impute null values with median and mode
df['Age'] = df['Age'].fillna(df['Age'].median())
df['Age'] = df['Age'].fillna(df['Age'].mode())
```
we can also fill null values by it’s previous value in the column which is called **Backward fill** or next occurring value in the column which is called **Forward fill**.
```python
#Backward fill or Forward fill can be used to impute the previous or next values
#Backward fill
df['Age'] = df['Age'].fillna(method='bfill')
#Forward fill
df['Age'] = df['Age'].fillna(method='ffill')
```
Replacing null values with mean using **SciKit Learn’s** [_SimpleImputer_](https://scikit-learn.org/stable/modules/generated/sklearn.impute.SimpleImputer.html) class.
___

```python
# Replacing the null values in the Age column with Mean
from sklearn.impute import SimpleImputerimputer = SimpleImputer(missing_values=np.nan, strategy='mean')
# Fit and transform to the parameters
new_df['Age'] = imputer.fit_transform(new_df[['Age']])
# Checking for any null values
new_df['Age'].isnull().sum()
# Alternatively, we can replace null values with median, most frequent value and also with an constant
# Replace with Median
imputer = SimpleImputer(missing_values=np.nan, strategy='median')new_df['Age'] = imputer.fit_transform(new_df[['Age']])
```
