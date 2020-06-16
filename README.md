# End-to-end-ML-Data-Modelling-Pipelines
## Define Problem:
Here we will build a predictive model that answers the question: “what sorts of people were more likely to survive?” using passenger data (ie name, age, gender, socio-economic class, etc).
# Part 1: [Importing Necessary Libraries and datasets](https://www.kaggle.com/masumrumi/a-statistical-analysis-ml-workflow-of-titanic#Part-1:-Importing-Necessary-Libraries-and-datasets)
## 1a. Loading libraries
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

## 1b. Loading Datasets
Machine learning algorithms need data. You can load your own data from business database tables or CSV / excel files.
* Load data from Netezza DB, using below 2 steps.
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
	database = ''
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
## 1c. A Glimpse of the Datasets
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
### Types of attributes
**Categorical:**
-   **Nominal**(variables that have two or more categories, but which do not have an intrinsic order.)
    
    > -   **Cabin**
    > -   **Embarked**(Port of Embarkation)
    >     
    >     ```
    >         C(Cherbourg)
    >         Q(Queenstown) 
    >         S(Southampton)
    >     ```
    >     
    
-   **Dichotomous**(Nominal variable with only two categories)
    
    > -   **Sex**
    >     
    >     ```
    >         Female
    >         Male
    >     ```
    >     
    
-   **Ordinal**(variables that have two or more categories just like nominal variables. Only the categories can also be ordered or ranked.)
    
    > -   **Pclass**  (A proxy for socio-economic status (SES))
    >     
    >     ```
    >         1(Upper)
    >         2(Middle) 
    >         3(Lower)
    >     ```
    >     
    

----------

**Numeric:**

-   **Discrete**
    
    > -   **Passenger ID**(Unique identifing # for each passenger)
    > -   **SibSp**
    > -   **Parch**
    > -   **Survived**  (Our outcome or dependent variable)
    >     
    >     ```
    >        0
    >        1
    >     ```
    >     
    
-   **Continous**
    
    > -   **Age**
    > -   **Fare**
    

----------

**Text Variable**

> -   **Ticket**  (Ticket number for passenger.)
> -   **Name**( Name of the passenger.)

## 1d. Insight of Dataset
-   **‘Survived’**  is the target variable, which we will predict once our preprocessing of our data is done. So, we retain that column.
-   Only the columns such as  **‘Age’, ‘Cabin’ and ‘Embarked’**  has missing values.
-   **‘PassengerId’, ‘Name’ and ‘Ticket’** doesn’t add much value in predicting the target variable.
-   **‘ParCh’**(Parent/Children) **and ‘SibSp’**(Siblings/Spouse) details are related to family, so we can derive a new column named  **‘Size of the family’**
-   **‘Sex’, ‘Cabin’ and ‘Embarked’**  are the categorical data that needs to be encoded to numerical values.

## 1e.  Dropping of unnecessary attributes
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
### Creating new interaction variables
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
## 1f. Splitting Continuous and Categorical types variables
```python
cols=df.columns
num_columns=df._get_numeric_data().columns
cat_columns=list(set(cols) - set(num_columns))
print("Total Columns : " + str(len(cols)))
df_categorical = df[cat_columns]
df_continuous = df[num_columns]
````
# Part 2: Overview and Cleaning the Data
## Step 2a: Analyze Data
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
### Correlation Matrix and Heatmap
#### Correlations
```python
pd.DataFrame(abs(train.corr()['Survived']).sort_values(ascending = False))
## get the most important variables. 
#Squaring the correlation feature not only gives on positive correlations but also amplifies the relationships.
corr = df.corr()**2
corr.Survived.sort_values(ascending=False)
```
**Heatmeap to see the correlation between features.** 
```python
# Generate a mask for the upper triangle (taken from seaborn example gallery)
import numpy as np
mask = np.zeros_like(train.corr(), dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
sns.set_style('whitegrid')
plt.subplots(figsize = (15,12))
sns.heatmap(train.corr(), 
            annot=True,
            mask = mask,
            cmap = 'RdBu', ## in order to reverse the bar replace "RdBu" with "RdBu_r"
            linewidths=.9, 
            linecolor='white',
            fmt='.2g',
            center = 0,
            square=True)
plt.title("Correlations Among Features", y = 1.03,fontsize = 20, pad = 40);
```
![](https://www.kaggleusercontent.com/kf/34502374/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..FWSErvF5JrIqNq4M_ZJjHA.zUVO6cbsCcrhaBX2ZI2k2skzMqT7CxPus62WMGsjP9wDak5wWEazHpLkEeJX9zOQtnIvCj73Imj4sLgWRQQRPVOGYqyIJ1KIGHLHB1zijNfj4AsltYONLAIVQimxeIPpzdSlrwbgojLY0aRHVqI-BvWxrKj8G-_kI1P_mVv7og-Rexeh6Tc_1xjkxSFn7Nhv7-svxBG004IvleJff3d6v8ekjKLEtSGG-yzcOvng1xX9TiFTaONTikW-5H80d_ZFjSWyDImeSzDXhUliwduCteq-1W5zFvNsMhfCglBY2sREt6ryOlRgHmddJlLkuAmW86PlpvcDi_l-NE2sNl1_jx4jf1OO6KQKcVzweJ2jft98LczHNWI3t15HSvyAeoQVyIGF8Y3Y0xF2nkQSkQN6Cdj2rXHTFdMTJNZFDua5SKiE6q6eE5R7Y4-tPLuwnof2Y7ZlWmnE3cI1_aX020D7B9YINXNdKMQa1vmJNQXP22NNWcoxNT_sVSVL6uwVsYFuvBN679psyrsMxUc4NGGS-AoSq1EPo359bGaUZ7FzbKNMEiJn0Rx7nIrhKDPtZ2_fFdpkaUlHEw56I6BYYLzn3lecwsoFTOoyMG3gIFYNXW_hPj_rhVemK7xTLUFHeQm9Xn4_WOgPR_kVlor9x4XbKSLizDu1Jymobo89nGmcXW7e54oCOvK7pQSqXegG8KL0.d525aI6-T3k9DlhBKiV7rg/__results___files/__results___107_0.png)
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
## 2b. Dealing with Missing values
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
## **Categorical Data**
**Cabin** and **Embarked** columns are of **_Categorical_** datatype. Now let’s see how to handle null values which are of Categorical type. View first few rows of these two columns to know which type of values these two columns comprised of in the dataframe.

```python
# Handling Missing values in Categorical data
df[['Cabin','Embarked']].head()
```
![](https://miro.medium.com/max/421/1*YuADT0O3fee2MFtZhtnvjA.png)
Replacing those null values with most frequent value among them.
```python
#Number of Missing values in both the columns
df[['Cabin', 'Embarked']].isnull().sum()
# Most frequent values in the Embarked column data
df['Embarked'].value_counts()
# Replacing the null values with the most frequent value
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].value_counts().index[0])
# Replacing null values in Embarked with most frequent value
imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
new_df['Embarked'] = imputer.fit_transform(new_df[['Embarked']])
# Value counts for Embarked column
new_df['Embarked'].value_counts()
```
Another way of handling null values in the column which is of categorical type, is to add an **_Unknown_** class and replace those null values with that class.
```python
# Replacing null values with Unknown Class
df['Cabin'] = df['Cabin'].fillna('Unknown')
# Value counts for Cabin Column
df['Cabin'].value_counts()
# Checking for null values in Cabin column
df['Cabin'].isnull().sum()
# Replacing null values in Cabin with Unknown class
imputer = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value='Unknown')
new_df['Cabin'] = imputer.fit_transform(new_df[['Cabin']])
# Checking for null values in the Cabin columnnew_
df['Cabin'].isnull().sum()
```
# Part 3. Visualization and Feature Relations
Before we dive into finding relations between independent variables and our dependent variable(survivor), let us create some assumptions about how the relations may turn-out among features.
**Assumptions:**
-   Gender: More female survived than male
-   Pclass: Higher socio-economic status passenger survived more than others.
-   Age: Younger passenger survived more than other passengers.
-   Fare: Passenger with higher fare survived more that other passengers. This can be quite correlated with Pclass.
Now, let's see how the features are related to each other by creating some visualizations.
## 3a. Gender and Survived
```python
import seaborn as sns
pal = {'male':"green", 'female':"Pink"}
sns.set(style="darkgrid")
plt.subplots(figsize = (15,8))
ax = sns.barplot(x = "Sex", 
                 y = "Survived", 
                 data=df, 
                 palette = pal,
                 linewidth=5,
                 order = ['female','male'],
                 capsize = .05,

                )

plt.title("Survived/Non-Survived Passenger Gender Distribution", fontsize = 25,loc = 'center', pad = 40)
plt.ylabel("% of passenger survived", fontsize = 15, )
plt.xlabel("Sex",fontsize = 15)
```
![](https://www.kaggleusercontent.com/kf/34502374/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..FWSErvF5JrIqNq4M_ZJjHA.zUVO6cbsCcrhaBX2ZI2k2skzMqT7CxPus62WMGsjP9wDak5wWEazHpLkEeJX9zOQtnIvCj73Imj4sLgWRQQRPVOGYqyIJ1KIGHLHB1zijNfj4AsltYONLAIVQimxeIPpzdSlrwbgojLY0aRHVqI-BvWxrKj8G-_kI1P_mVv7og-Rexeh6Tc_1xjkxSFn7Nhv7-svxBG004IvleJff3d6v8ekjKLEtSGG-yzcOvng1xX9TiFTaONTikW-5H80d_ZFjSWyDImeSzDXhUliwduCteq-1W5zFvNsMhfCglBY2sREt6ryOlRgHmddJlLkuAmW86PlpvcDi_l-NE2sNl1_jx4jf1OO6KQKcVzweJ2jft98LczHNWI3t15HSvyAeoQVyIGF8Y3Y0xF2nkQSkQN6Cdj2rXHTFdMTJNZFDua5SKiE6q6eE5R7Y4-tPLuwnof2Y7ZlWmnE3cI1_aX020D7B9YINXNdKMQa1vmJNQXP22NNWcoxNT_sVSVL6uwVsYFuvBN679psyrsMxUc4NGGS-AoSq1EPo359bGaUZ7FzbKNMEiJn0Rx7nIrhKDPtZ2_fFdpkaUlHEw56I6BYYLzn3lecwsoFTOoyMG3gIFYNXW_hPj_rhVemK7xTLUFHeQm9Xn4_WOgPR_kVlor9x4XbKSLizDu1Jymobo89nGmcXW7e54oCOvK7pQSqXegG8KL0.d525aI6-T3k9DlhBKiV7rg/__results___files/__results___63_0.png)
-   As we suspected, female passengers have survived at a much better rate than male passengers.
-   It seems about right since females and children were the priority.
## 3b. Pclass and Survived
```python
temp = df[['Pclass', 'Survived', 'PassengerId']].groupby(['Pclass', 'Survived']).count().reset_index()
temp_df = pd.pivot_table(temp, values = 'PassengerId', index = 'Pclass',columns = 'Survived')
names = ['No', 'Yes']
temp_df.columns = names
r = [0,1,2]
totals = [i+j for i, j in zip(temp_df['No'], temp_df['Yes'])]
No_s = [i / j * 100 for i,j in zip(temp_df['No'], totals)]
Yes_s = [i / j * 100 for i,j in zip(temp_df['Yes'], totals)]
## Plotting
plt.subplots(figsize = (15,10))
barWidth = 0.60
names = ('Upper', 'Middle', 'Lower')
# Create green Bars
plt.bar(r, No_s, color='Red', edgecolor='white', width=barWidth)
# Create orange Bars
plt.bar(r, Yes_s, bottom=No_s, color='Green', edgecolor='white', width=barWidth)

 
# Custom x axis
plt.xticks(r, names)
plt.xlabel("Pclass")
plt.ylabel('Percentage')
 
# Show graphic
plt.show()
```
![](https://www.kaggleusercontent.com/kf/34502374/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..FWSErvF5JrIqNq4M_ZJjHA.zUVO6cbsCcrhaBX2ZI2k2skzMqT7CxPus62WMGsjP9wDak5wWEazHpLkEeJX9zOQtnIvCj73Imj4sLgWRQQRPVOGYqyIJ1KIGHLHB1zijNfj4AsltYONLAIVQimxeIPpzdSlrwbgojLY0aRHVqI-BvWxrKj8G-_kI1P_mVv7og-Rexeh6Tc_1xjkxSFn7Nhv7-svxBG004IvleJff3d6v8ekjKLEtSGG-yzcOvng1xX9TiFTaONTikW-5H80d_ZFjSWyDImeSzDXhUliwduCteq-1W5zFvNsMhfCglBY2sREt6ryOlRgHmddJlLkuAmW86PlpvcDi_l-NE2sNl1_jx4jf1OO6KQKcVzweJ2jft98LczHNWI3t15HSvyAeoQVyIGF8Y3Y0xF2nkQSkQN6Cdj2rXHTFdMTJNZFDua5SKiE6q6eE5R7Y4-tPLuwnof2Y7ZlWmnE3cI1_aX020D7B9YINXNdKMQa1vmJNQXP22NNWcoxNT_sVSVL6uwVsYFuvBN679psyrsMxUc4NGGS-AoSq1EPo359bGaUZ7FzbKNMEiJn0Rx7nIrhKDPtZ2_fFdpkaUlHEw56I6BYYLzn3lecwsoFTOoyMG3gIFYNXW_hPj_rhVemK7xTLUFHeQm9Xn4_WOgPR_kVlor9x4XbKSLizDu1Jymobo89nGmcXW7e54oCOvK7pQSqXegG8KL0.d525aI6-T3k9DlhBKiV7rg/__results___files/__results___68_0.png)
It looks like ...

-   ~ 63% first class passenger survived titanic tragedy, while
-   ~ 48% second class and
-   ~ only 24% third class passenger survived.
The first class passengers had the upper hand during the tragedy.
## 3c. Fare and Survived
```python
# Kernel Density Plot
fig = plt.figure(figsize=(15,8),)
ax=sns.kdeplot(train.loc[(train['Survived'] == 0),'Fare'] , color='gray',shade=True,label='not survived')
ax=sns.kdeplot(train.loc[(train['Survived'] == 1),'Fare'] , color='g',shade=True, label='survived')
plt.title('Fare Distribution Survived vs Non Survived', fontsize = 25, pad = 40)
plt.ylabel("Frequency of Passenger Survived", fontsize = 15, labelpad = 20)
plt.xlabel("Fare", fontsize = 15, labelpad = 20);
```
![](https://www.kaggleusercontent.com/kf/34502374/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..FWSErvF5JrIqNq4M_ZJjHA.zUVO6cbsCcrhaBX2ZI2k2skzMqT7CxPus62WMGsjP9wDak5wWEazHpLkEeJX9zOQtnIvCj73Imj4sLgWRQQRPVOGYqyIJ1KIGHLHB1zijNfj4AsltYONLAIVQimxeIPpzdSlrwbgojLY0aRHVqI-BvWxrKj8G-_kI1P_mVv7og-Rexeh6Tc_1xjkxSFn7Nhv7-svxBG004IvleJff3d6v8ekjKLEtSGG-yzcOvng1xX9TiFTaONTikW-5H80d_ZFjSWyDImeSzDXhUliwduCteq-1W5zFvNsMhfCglBY2sREt6ryOlRgHmddJlLkuAmW86PlpvcDi_l-NE2sNl1_jx4jf1OO6KQKcVzweJ2jft98LczHNWI3t15HSvyAeoQVyIGF8Y3Y0xF2nkQSkQN6Cdj2rXHTFdMTJNZFDua5SKiE6q6eE5R7Y4-tPLuwnof2Y7ZlWmnE3cI1_aX020D7B9YINXNdKMQa1vmJNQXP22NNWcoxNT_sVSVL6uwVsYFuvBN679psyrsMxUc4NGGS-AoSq1EPo359bGaUZ7FzbKNMEiJn0Rx7nIrhKDPtZ2_fFdpkaUlHEw56I6BYYLzn3lecwsoFTOoyMG3gIFYNXW_hPj_rhVemK7xTLUFHeQm9Xn4_WOgPR_kVlor9x4XbKSLizDu1Jymobo89nGmcXW7e54oCOvK7pQSqXegG8KL0.d525aI6-T3k9DlhBKiV7rg/__results___files/__results___74_0.png)
This plot shows something impressive..

-   The spike in the plot under 100 dollar represents that a lot of passengers who bought the ticket within that range did not survive.
-   When fare is approximately more than 280 dollars, there is no gray shade which means, either everyone passed that fare point survived or maybe there is an outlier that clouds our judgment. Let's check...
## 3e. Combined Feature Relation
```python
pal = {1:"seagreen", 0:"gray"}
g = sns.FacetGrid(df,size=5, col="Sex", row="Survived", margin_titles=True, hue = "Survived",
                  palette=pal)
g = g.map(plt.hist, "Age", edgecolor = 'white');
g.fig.suptitle("Survived by Sex and Age", size = 25)
plt.subplots_adjust(top=0.90)
```
![](https://www.kaggleusercontent.com/kf/34502374/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..FWSErvF5JrIqNq4M_ZJjHA.zUVO6cbsCcrhaBX2ZI2k2skzMqT7CxPus62WMGsjP9wDak5wWEazHpLkEeJX9zOQtnIvCj73Imj4sLgWRQQRPVOGYqyIJ1KIGHLHB1zijNfj4AsltYONLAIVQimxeIPpzdSlrwbgojLY0aRHVqI-BvWxrKj8G-_kI1P_mVv7og-Rexeh6Tc_1xjkxSFn7Nhv7-svxBG004IvleJff3d6v8ekjKLEtSGG-yzcOvng1xX9TiFTaONTikW-5H80d_ZFjSWyDImeSzDXhUliwduCteq-1W5zFvNsMhfCglBY2sREt6ryOlRgHmddJlLkuAmW86PlpvcDi_l-NE2sNl1_jx4jf1OO6KQKcVzweJ2jft98LczHNWI3t15HSvyAeoQVyIGF8Y3Y0xF2nkQSkQN6Cdj2rXHTFdMTJNZFDua5SKiE6q6eE5R7Y4-tPLuwnof2Y7ZlWmnE3cI1_aX020D7B9YINXNdKMQa1vmJNQXP22NNWcoxNT_sVSVL6uwVsYFuvBN679psyrsMxUc4NGGS-AoSq1EPo359bGaUZ7FzbKNMEiJn0Rx7nIrhKDPtZ2_fFdpkaUlHEw56I6BYYLzn3lecwsoFTOoyMG3gIFYNXW_hPj_rhVemK7xTLUFHeQm9Xn4_WOgPR_kVlor9x4XbKSLizDu1Jymobo89nGmcXW7e54oCOvK7pQSqXegG8KL0.d525aI6-T3k9DlhBKiV7rg/__results___files/__results___82_0.png)
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
## 5c. Feature Scaling
Here  **Age**  and  **Calculated_fare**  is much higher in magnitude compared to others machine learning features. This can create problems as many machine learning models will get confused thinking  **Age**  and  **Calculated_fare**  have higher weight than other features. Therefore, we need to do feature scaling to get a better result. There are multiple ways to do feature scaling.

-   **MinMaxScaler**-Scales the data using the max and min values so that it fits between 0 and 1.
-   **StandardScaler**-Scales the data so that it has mean 0 and variance of 1.
-   **RobustScaler**-Scales the data similary to Standard Scaler, but makes use of the median and scales using the interquertile range so as to aviod issues with large outliers.
```python
headers = df.columns 
df.head()
# Feature Scaling
## We will be using standardscaler to transform
from sklearn.preprocessing import StandardScaler
st_scale = StandardScaler()
## transforming "df"
df= st_scale.fit_transform(df)
#After Scaling
pd.DataFrame(df, columns=headers).head()
```
## Variance Inflation Factor
```python
from statsmodels.stats.outliers_influence import variance_inflation_factor
vif = pd.DataFrame()
vif["features"] = X.columns
vif["VIF Factor"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
```
## Variable Clustering using VARCLUSHI
```python  
from  varclushi  import  VarClusHi  
df_cluster=VarClusHi(X,maxeigval2=1,maxclus=12)  
df_cluster.varclus()
```
## Step Wise Logistic Regression
```python
import pandas as pd
import numpy as np
import statsmodels.api as sm
#from sklearn.feature_Selection import RFE
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression(fit_intercept=True,tol=0.0001,random_state=10)
sel_var=['AVG_CALL_COST_180D','AVG_CALL_COST_60D','AVG_CALL_COST_90D','AVG_CALL_COST_ALL']

X = df[sel_var]
y = df['TARGET_30D_PAYMENT_RELATED']

def stepwise_selection(X,y,initial_list=[],threshold_in=0.0001,threshold_out=0.0001,verbose=True):
    included = list(initial_list)
    while True:
        changed=False
        excluded = list(set(X.columns)-set(included))
        new_pval=pd.Series(index=excluded)
        for new_column in excluded:
            #model=sm.OLS(y,sm.add_constant(pd.DataFrame(X[included+[new_column]]))).fit()
            model=sm.Logit(y,sm.add_constant(pd.DataFrame(X[included+[new_column]]))).fit()
            #model=logreg.fit(pd.DataFrame(X[included+[new_column]]),y)
            print(model.summary())
            new_pval[new_column]=model.pvalues[new_column]
        best_pval = new_pval.min()
        if best_pval < threshold_in:
            best_feature = new_pval.argmin()
            included.append(best_feature)
            changed=True
            if verbose:
                print('Add {:50} with p_value {:.6}'.format(best_feature,best_pval))
                
        #backward step
        #model=sm.OLS(y,sm.add_constant(pd.DataFrame(X[included]))).fit()
        model=sm.Logit(y,sm.add_constant(pd.DataFrame(X[included]))).fit()
        #model=logreg.fit(pd.DataFrame(X[included]),y)
        print(model.summary())
        pvalues=model.pvalues.iloc[1:]
        worst_pval = pvalues.max()
        if worst_pval > threshold_out:
            changed=True
            worst_feature = pvalues.argmax()
            included.remove(worst_feature)
            if verbose:
                print('Drop {:50} with p_value {:.6}'.format(worst_feature,worst_pval))
        if not changed:
            break
    return included
result=stepwise_selection(X,y)
```

# Part 6: Pre-Modeling Tasks
## 6a. Separating dependent and independent variables
Our dependent variable or target variable is something that we are trying to find, and our independent variable is the features we use to find the dependent variable. The way we use machine learning algorithm in a dataset is that we train our machine learning model by specifying independent variables and dependent variable. 
## 6b. Splitting the training data
The dataset used to train a machine learning algorithm is called a training dataset. The dataset
used to train an algorithm cannot be used to give you reliable estimates of the accuracy of the
model on new data. This is a big problem because the whole idea of creating the model is to
make predictions on new data. You can use statistical methods called resampling methods to
split your training dataset into subsets, some are used to train the model and others are held
back and used to estimate the accuracy of the model on unseen data.
There are multiple ways of splitting data. They are...
-   train_test_split.
-   cross_validation.
Splits the train data into 4 parts,  **X_train**,  **X_test**,  **y_train**,  **y_test**.
-   **X_train**  and  **y_train**  first used to train the algorithm.
-   then,  **X_test**  is used in that trained algorithms to predict  **outcomes.**
-   Once we get the  **outcomes**, we compare it with  **y_test**
```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size = .33, random_state=0)
```
By comparing the  **outcome**  of the model with  **y_test**, we can determine whether our algorithms are performing well or not. As we compare we use confusion matrix to determine different aspects of model performance.
# Part 7: Modeling the Data
```python
# import LogisticRegression model in python.
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_absolute_error, accuracy_score
## call on the model object
logreg = LogisticRegression(solver='liblinear',
                            penalty= 'l1',random_state = 42                                
                            )
## fit the model with "train_x" and "train_y"
logreg.fit(X_train,y_train)
## Once the model is trained we want to find out how well the model is performing, so we test the model. 
## we use "X_test" portion of the data(this data was not used to fit the model) to predict model outcome. 
y_pred = logreg.predict(X_test)
## Once predicted we save that outcome in "y_pred" variable.
## Then we compare the predicted value( "y_pred") and actual value("test_y") to see how well our model is performing.
```
## Evaluating a classification model

There are multiple ways to evaluate a classification model.
-   Confusion Matrix.
-   ROC Curve
-   AUC Curve.

### Confusion Matrix
**Confusion matrix**, a table that  **describes the performance of a classification model**. Confusion Matrix tells us how many our model predicted correctly and incorrectly in terms of binary/multiple outcome classes by comparing actual and predicted cases. For example, in terms of this dataset, our model is a binary one and we are trying to classify whether the passenger survived or not survived. we have fit the model using  **X_train**  and  **y_train**  and predicted the outcome of  **X_test**  in the variable  **y_pred**. So, now we will use a confusion matrix to compare between  **y_test**  and  **y_pred**. Let's do the confusion matrix.
```python
from sklearn.metrics import classification_report, confusion_matrix
# printing confision matrix
pd.DataFrame(confusion_matrix(y_test,y_pred),\
            columns=["Predicted Not-Survived", "Predicted Survived"],\
            index=["Not-Survived","Survived"] )
```
After all it is a matrix and we have some terminologies to call these statistics more specifically.
-   **True Positive(TP)**: values that the model predicted as yes(survived) and is actually yes(survived).
-   **True Negative(TN)**: values that model predicted as no(not-survived) and is actually no(not-survived)
-   **False Positive(or Type I error)**: values that model predicted as yes(survived) but actually no(not-survived)
-   **False Negative(or Type II error)**: values that model predicted as no(not-survived) but actually yes(survived)
 #### Accuracy
- Accuracy is the measure of how often the model is correct.
>  (TP + TN)/total
```python
from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_pred)
```
**Misclassification Rate:**  Misclassification Rate is the measure of how often the model is wrong**

-   Misclassification Rate and Accuracy are opposite of each other.
-   Missclassification is equivalent to 1 minus Accuracy.
-   Misclassification Rate is also known as "Error Rate".
> (FP + FN)/Total
>
**True Positive Rate/Recall/Sensitivity:**  How often the model predicts yes(survived) when it's actually yes(survived)?
> TP/(TP+FN)
```python
from sklearn.metrics import recall_score
recall_score(y_test, y_pred)
```
**False Positive Rate:**  How often the model predicts yes(survived) when it's actually no(not-survived)?
> FP/(FP+TN)

**True Negative Rate/Specificity:**  How often the model predicts no(not-survived) when it's actually no(not-survived)?
True Negative Rate is equivalent to 1 minus False Positive Rate.
> TN/(TN+FP)

**Precision:**  How often is it correct when the model predicts yes.
> TP/(TP+FP) 
```python
from sklearn.metrics import precision_score
precision_score(y_test, y_pred)
```
```python
from sklearn.metrics import classification_report, balanced_accuracy_score
print(classification_report(y_test, y_pred))
```
## AUC & ROC Curve
```python
from sklearn.metrics import roc_curve, auc
#plt.style.use('seaborn-pastel')
y_score = logreg.decision_function(X_test)

FPR, TPR, _ = roc_curve(y_test, y_score)
ROC_AUC = auc(FPR, TPR)
print (ROC_AUC)

plt.figure(figsize =[11,9])
plt.plot(FPR, TPR, label= 'ROC curve(area = %0.2f)'%ROC_AUC, linewidth= 4)
plt.plot([0,1],[0,1], 'k--', linewidth = 4)
plt.xlim([0.0,1.0])
plt.ylim([0.0,1.05])
plt.xlabel('False Positive Rate', fontsize = 18)
plt.ylabel('True Positive Rate', fontsize = 18)
plt.title('ROC for Titanic survivors', fontsize= 18)
plt.show()
```
![](https://www.kaggleusercontent.com/kf/34502374/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..FWSErvF5JrIqNq4M_ZJjHA.zUVO6cbsCcrhaBX2ZI2k2skzMqT7CxPus62WMGsjP9wDak5wWEazHpLkEeJX9zOQtnIvCj73Imj4sLgWRQQRPVOGYqyIJ1KIGHLHB1zijNfj4AsltYONLAIVQimxeIPpzdSlrwbgojLY0aRHVqI-BvWxrKj8G-_kI1P_mVv7og-Rexeh6Tc_1xjkxSFn7Nhv7-svxBG004IvleJff3d6v8ekjKLEtSGG-yzcOvng1xX9TiFTaONTikW-5H80d_ZFjSWyDImeSzDXhUliwduCteq-1W5zFvNsMhfCglBY2sREt6ryOlRgHmddJlLkuAmW86PlpvcDi_l-NE2sNl1_jx4jf1OO6KQKcVzweJ2jft98LczHNWI3t15HSvyAeoQVyIGF8Y3Y0xF2nkQSkQN6Cdj2rXHTFdMTJNZFDua5SKiE6q6eE5R7Y4-tPLuwnof2Y7ZlWmnE3cI1_aX020D7B9YINXNdKMQa1vmJNQXP22NNWcoxNT_sVSVL6uwVsYFuvBN679psyrsMxUc4NGGS-AoSq1EPo359bGaUZ7FzbKNMEiJn0Rx7nIrhKDPtZ2_fFdpkaUlHEw56I6BYYLzn3lecwsoFTOoyMG3gIFYNXW_hPj_rhVemK7xTLUFHeQm9Xn4_WOgPR_kVlor9x4XbKSLizDu1Jymobo89nGmcXW7e54oCOvK7pQSqXegG8KL0.d525aI6-T3k9DlhBKiV7rg/__results___files/__results___189_1.png)
```python
from sklearn.metrics import precision_recall_curve

y_score = logreg.decision_function(X_test)

precision, recall, _ = precision_recall_curve(y_test, y_score)
PR_AUC = auc(recall, precision)

plt.figure(figsize=[11,9])
plt.plot(recall, precision, label='PR curve (area = %0.2f)' % PR_AUC, linewidth=4)
plt.xlabel('Recall', fontsize=18)
plt.ylabel('Precision', fontsize=18)
plt.title('Precision Recall Curve for Titanic survivors', fontsize=18)
plt.legend(loc="lower right")
plt.show()
```
![](https://www.kaggleusercontent.com/kf/34502374/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..FWSErvF5JrIqNq4M_ZJjHA.zUVO6cbsCcrhaBX2ZI2k2skzMqT7CxPus62WMGsjP9wDak5wWEazHpLkEeJX9zOQtnIvCj73Imj4sLgWRQQRPVOGYqyIJ1KIGHLHB1zijNfj4AsltYONLAIVQimxeIPpzdSlrwbgojLY0aRHVqI-BvWxrKj8G-_kI1P_mVv7og-Rexeh6Tc_1xjkxSFn7Nhv7-svxBG004IvleJff3d6v8ekjKLEtSGG-yzcOvng1xX9TiFTaONTikW-5H80d_ZFjSWyDImeSzDXhUliwduCteq-1W5zFvNsMhfCglBY2sREt6ryOlRgHmddJlLkuAmW86PlpvcDi_l-NE2sNl1_jx4jf1OO6KQKcVzweJ2jft98LczHNWI3t15HSvyAeoQVyIGF8Y3Y0xF2nkQSkQN6Cdj2rXHTFdMTJNZFDua5SKiE6q6eE5R7Y4-tPLuwnof2Y7ZlWmnE3cI1_aX020D7B9YINXNdKMQa1vmJNQXP22NNWcoxNT_sVSVL6uwVsYFuvBN679psyrsMxUc4NGGS-AoSq1EPo359bGaUZ7FzbKNMEiJn0Rx7nIrhKDPtZ2_fFdpkaUlHEw56I6BYYLzn3lecwsoFTOoyMG3gIFYNXW_hPj_rhVemK7xTLUFHeQm9Xn4_WOgPR_kVlor9x4XbKSLizDu1Jymobo89nGmcXW7e54oCOvK7pQSqXegG8KL0.d525aI6-T3k9DlhBKiV7rg/__results___files/__results___190_0.png)

## Using Cross-validation
Pros:
-   Helps reduce variance.
-   Expends models predictability
### Grid Search on Logistic Regression
**Gridsearch**  is a simple concept but effective technique in Machine Learning. The word  **GridSearch**  stands for the fact that we are searching for optimal parameter/parameters over a "grid." These optimal parameters are also known as  **Hyperparameters**.  **The Hyperparameters are model parameters that are set before fitting the model and determine the behavior of the model.**. For example, when we choose to use linear regression, we may decide to add a penalty to the loss function such as Ridge or Lasso. These penalties require specific alpha (the strength of the regularization technique) to set beforehand. The higher the value of alpha, the more penalty is being added. GridSearch finds the optimal value of alpha among a range of values provided by us, and then we go on and use that optimal value to fit the model and get sweet results. It is essential to understand those model parameters are different from models outcomes, for example,  **coefficients**  or model evaluation metrics such as  **accuracy score**  or  **mean squared error**  are model outcomes and different than hyperparameters.

```python
from sklearn.model_selection import GridSearchCV, StratifiedKFold
## C_vals is the alpla value of lasso and ridge regression(as alpha increases the model complexity decreases,)
## remember effective alpha scores are 0<alpha<infinity 
C_vals = [0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,2,3,4,5,6,7,8,9,10,12,13,14,15,16,16.5,17,17.5,18]
## Choosing penalties(Lasso(l1) or Ridge(l2))
penalties = ['l1','l2']
## Choose a cross validation strategy. 
cv = StratifiedShuffleSplit(n_splits = 10, test_size = .25)

## setting param for param_grid in GridSearchCV. 
param = {'penalty': penalties, 'C': C_vals}

logreg = LogisticRegression(solver='liblinear')
## Calling on GridSearchCV object. 
grid = GridSearchCV(estimator=LogisticRegression(), 
                           param_grid = param,
                           scoring = 'accuracy',
                            n_jobs =-1,
                           cv = cv
                          )
## Fitting the model
grid.fit(X, y)
```
```python
## Getting the best of everything. 
print (grid.best_score_)
print (grid.best_params_)
print(grid.best_estimator_)
```
#### Using the best parameters from the grid-search.
```python
### Using the best parameters from the grid-search.
logreg_grid = grid.best_estimator_
logreg_grid.score(X,y)
```
### Confusion Matrix: Under-fitting & Over-fitting:
![](https://cdncontribute.geeksforgeeks.org/wp-content/uploads/fittings.jpg)
As you see in the chart above. **Underfitting** is when the model fails to capture important aspects of the data and therefore introduces more bias and performs poorly. On the other hand, **Overfitting** is when the model performs too well on the training data but does poorly in the validation set or test sets. This situation is also known as having less bias but more variation and perform poorly as well. Ideally, we want to configure a model that performs well not only in the training data but also in the test data. This is where **bias-variance tradeoff** comes in. When we have a model that overfits, meaning less biased and more of variance, we introduce some bias in exchange of having much less variance. One particular tactic for this task is regularization models (Ridge, Lasso, Elastic Net). These models are built to deal with the bias-variance tradeoff.
![](http://scott.fortmann-roe.com/docs/docs/BiasVariance/biasvariance.png)
Ideally, we want to pick a sweet spot where the model performs well in training set, validation set, and test set. As the model gets complex, bias decreases, variance increases. However, the most critical part is the error rates. We want our models to be at the bottom of that  **U**  shape where the error rate is the least. That sweet spot is also known as  **Optimum Model Complexity(OMC).**
Now that we know what we want in terms of under-fitting and over-fitting, let's talk about how to combat them.
How to combat over-fitting?
-   Simplify the model by using less parameters.
-   Simplify the model by changing the hyper-parameters`.
-   Introducing regularization models.
-   Use more training data.
-   Gather more data ( and gather better quality data).
```python
from sklearn.linear_model import Ridge,RidgeCV,ElasticNet,Lasso,LassoCV,LassoLarsCV,LassoLarsIC
from sklearn.model_selection import cross_val_score
import numpy as np
import matplotlib.pyplot as plt
import time
import pickle
import pandas as pd
# LassoLinear
clf=Lasso(alpha=0.00001,normalize=True)
clf.fit(X,y)
clf.score(X,y)
#clf.coef_
coef=pd.Series(clf.coef_,index=X.columns)
print("Lasso linear")
print(coef)
```
# Spot-Check Algorithms
## 7a. K-Nearest Neighbor classifier(KNN)
```python
## Importing the model. 
from sklearn.neighbors import KNeighborsClassifier
## calling on the model oject. 
knn = KNeighborsClassifier(metric='minkowski', p=2)
## knn classifier works by doing euclidian distance 

## doing 10 fold staratified-shuffle-split cross validation 
cv = StratifiedShuffleSplit(n_splits=10, test_size=.25, random_state=2)

accuracies = cross_val_score(knn, X,y, cv = cv, scoring='accuracy')
print ("Cross-Validation accuracy scores:{}".format(accuracies))
print ("Mean Cross-Validation accuracy score: {}".format(round(accuracies.mean(),3)))
```
#### Manually find the best possible k value for KNN
```python
## Search for an optimal value of k for KNN.
k_range = range(1,31)
k_scores = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X,y, cv = cv, scoring = 'accuracy')
    k_scores.append(scores.mean())
print("Accuracy scores are: {}\n".format(k_scores))
print ("Mean accuracy score: {}".format(np.mean(k_scores)))
```
```python
from matplotlib import pyplot as plt
plt.plot(k_range, k_scores)
```
### Grid search on KNN classifier
```python
## trying out multiple values for k
k_range = range(1,31)
## 
weights_options=['uniform','distance']
# 
param = {'n_neighbors':k_range, 'weights':weights_options}
## Using startifiedShufflesplit. 
cv = StratifiedShuffleSplit(n_splits=10, test_size=.30, random_state=15)
# estimator = knn, param_grid = param, n_jobs = -1 to instruct scikit learn to use all available processors. 
grid = GridSearchCV(KNeighborsClassifier(), param,cv=cv,verbose = False, n_jobs=-1)
## Fitting the model. 
grid.fit(X,y)
```
```python
print(grid.best_score_)
print(grid.best_params_)
print(grid.best_estimator_)
```
#### Using best estimator from grid search using KNN
```python
### Using the best parameters from the grid-search.
knn_grid= grid.best_estimator_
knn_grid.score(X,y)
```
#### Using RandomizedSearchCV
Randomized search is a close cousin of grid search. It doesn't always provide the best result but its fast.
```python
from sklearn.model_selection import RandomizedSearchCV
## trying out multiple values for k
k_range = range(1,31)
## 
weights_options=['uniform','distance']
# 
param = {'n_neighbors':k_range, 'weights':weights_options}
## Using startifiedShufflesplit. 
cv = StratifiedShuffleSplit(n_splits=10, test_size=.30)
# estimator = knn, param_grid = param, n_jobs = -1 to instruct scikit learn to use all available processors. 
## for RandomizedSearchCV, 
grid = RandomizedSearchCV(KNeighborsClassifier(), param,cv=cv,verbose = False, n_jobs=-1, n_iter=40)
## Fitting the model. 
grid.fit(X,y)
```
```python
print (grid.best_score_)
print (grid.best_params_)
print(grid.best_estimator_)
```
```python
### Using the best parameters from the grid-search.
knn_ran_grid = grid.best_estimator_
knn_ran_grid.score(X,y)
```
## Gaussian Naive Bayes
```python
# Gaussian Naive Bayes
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
gaussian = GaussianNB()
gaussian.fit(X, y)
y_pred = gaussian.predict(X_test)
gaussian_accy = round(accuracy_score(y_pred, y_test), 3)
print(gaussian_accy)
```
## Support Vector Machines(SVM)
```python
from sklearn.svm import SVC
Cs = [0.001, 0.01, 0.1, 1,1.5,2,2.5,3,4,5, 10] ## penalty parameter C for the error term. 
gammas = [0.0001,0.001, 0.01, 0.1, 1]
param_grid = {'C': Cs, 'gamma' : gammas}
cv = StratifiedShuffleSplit(n_splits=10, test_size=.30, random_state=15)
grid_search = GridSearchCV(SVC(kernel = 'rbf', probability=True), param_grid, cv=cv) ## 'rbf' stands for gaussian kernel
grid_search.fit(X,y)
```
```python
print(grid_search.best_score_)
print(grid_search.best_params_)
print(grid_search.best_estimator_)
```
```python
# using the best found hyper paremeters to get the score. 
svm_grid = grid_search.best_estimator_
svm_grid.score(X,y)
```
## Decision Tree Classifier
```python
from sklearn.tree import DecisionTreeClassifier
max_depth = range(1,30)
max_feature = [21,22,23,24,25,26,28,29,30,'auto']
criterion=["entropy", "gini"]

param = {'max_depth':max_depth, 
         'max_features':max_feature, 
         'criterion': criterion}
grid = GridSearchCV(DecisionTreeClassifier(), 
                                param_grid = param, 
                                 verbose=False, 
                                 cv=StratifiedKFold(n_splits=20, random_state=15, shuffle=True),
                                n_jobs = -1)
grid.fit(X, y)
```
```python
print( grid.best_params_)
print (grid.best_score_)
print (grid.best_estimator_)
```
#### Let's look at the feature importance from decision tree grid
```python
## feature importance
feature_importances = pd.DataFrame(dectree_grid.feature_importances_,
                                   index = column_names,
                                    columns=['importance'])
feature_importances.sort_values(by='importance', ascending=False).head(10)
```
## Random Forest Classifier(RF)
RF is an ensemble method (combination of many decision trees) which is where the "forest" part comes in. One crucial details about Random Forest is that while using a forest of decision trees, RF model **takes random subsets of the original dataset(bootstrapped)** and **random subsets of the variables(features/columns)**. Using this method, the RF model creates 100's-1000's(the amount can be menually determined) of a wide variety of decision trees. This variety makes the RF model more effective and accurate. We then run each test data point through all of these 100's to 1000's of decision trees or the RF model and take a vote on the output.
```python
from sklearn.model_selection import GridSearchCV, StratifiedKFold, StratifiedShuffleSplit
from sklearn.ensemble import RandomForestClassifier
n_estimators = [140,145,150,155,160];
max_depth = range(1,10);
criterions = ['gini', 'entropy'];
cv = StratifiedShuffleSplit(n_splits=10, test_size=.30, random_state=15)

parameters = {'n_estimators':n_estimators,
              'max_depth':max_depth,
              'criterion': criterions
              
        }
grid = GridSearchCV(estimator=RandomForestClassifier(max_features='auto'),
                                 param_grid=parameters,
                                 cv=cv,
                                 n_jobs = -1)
grid.fit(X,y)
```python
print (grid.best_score_)
print (grid.best_params_)
print (grid.best_estimator_)
```
```python
rf_grid = grid.best_estimator_
rf_grid.score(X,y)
```
### Feature Importance
```python
## feature importance
feature_importances = pd.DataFrame(rf_grid.feature_importances_,
                                   index = column_names,
                                    columns=['importance'])
feature_importances.sort_values(by='importance', ascending=False).head(10)
```
#### Hyper Tuning : Randomized Search CV (Random Forest)
```python
import numpy as np

#Number of tress in Random Forest
n_estimators = [int(x) for x in np.linspace(start = 200,stop = 1000, num = 5)]

#Number of features to consider while splitting
max_features = ['auto', 'sqrt']

#Maximum number of levels in the tree
max_depth = [int(x) for x in np.linspace(start = 7, stop = 10, num = 4)]

#Minimum # of samples required to split the node
min_samples_split = [10,15]

#Minimum # of samples required at each leaf node
min_samples_leaf = [3,5]

#Method of selecting samples from training each tree
bootstrap = [True, False]

from sklearn.metrics import accuracy_score,make_scorer,precision_score,recall_score,roc_auc_score,f1_score
scoring={'AUC' : make_scorer(roc_auc_score) , 
         'Accuracy':make_scorer(accuracy_score), 
         'Recall':make_scorer(recall_score), 
         'Precision':make_scorer(precision_score),
         'F1 Score':make_scorer(f1_score)}

#Create the random grid:
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf ,
               'bootstrap': bootstrap                
              }
from sklearn.ensemble import RandomForestClassifier
RFcl = RandomForestClassifier(random_state = 0, n_jobs = -1) 

from sklearn.model_selection import RandomizedSearchCV
CV_rfc = RandomizedSearchCV(estimator=RFcl, param_distributions =random_grid, n_jobs = -1, cv= 10,scoring=scoring,refit='Recall',return_train_score=True,n_iter=10)
CV_rfc.fit(X_train, y_train)
print(CV_rfc.cv_results_)
```
### Hyper Tuning : Grid Search CV (Random Forest)
```python
def format_grid_search_result(res):
    global df_gs_result
    gs_results=res
    
    gs_model=gs_results['params']
    
    # Grid Search : AUC Metrics
    gs_mean_test_AUC=pd.Series(gs_results['mean_test_AUC'])
    gs_std_test_AUC=pd.Series(gs_results['std_test_AUC'])
    gs_rank_test_AUC=pd.Series(gs_results['rank_test_AUC'])
    
    # Grid Search : Accuracy Metrics
    gs_mean_test_Accuracy=pd.Series(gs_results['mean_test_Accuracy'])
    gs_std_test_Accuracy=pd.Series(gs_results['std_test_Accuracy'])
    gs_rank_test_Accuracy=pd.Series(gs_results['rank_test_Accuracy'])
    
    # Grid Search : Recall Metrics
    gs_mean_test_Recall=pd.Series(gs_results['mean_test_Recall'])
    gs_std_test_Recall=pd.Series(gs_results['std_test_Recall'])
    gs_rank_test_Recall=pd.Series(gs_results['rank_test_Recall'])

    # Grid Search : Precision Metrics
    gs_mean_test_Precision=pd.Series(gs_results['mean_test_Precision'])
    gs_std_test_Precision=pd.Series(gs_results['std_test_Precision'])
    gs_rank_test_Precision=pd.Series(gs_results['rank_test_Precision'])
    
    # Grid Search : F1-Score Metrics
    gs_mean_test_F1_Score=pd.Series(gs_results['mean_test_F1 Score'])
    gs_std_test_F1_Score=pd.Series(gs_results['std_test_F1 Score'])
    gs_rank_test_F1_Score=pd.Series(gs_results['rank_test_F1 Score'])   

    
    gs_model_split=str(gs_model).replace("[{","").replace("}]","").split('}, {')
    df_gs_result=pd.DataFrame(gs_model_split,index=None,columns=['Model_attributes'])
    df_gs_result=pd.concat([df_gs_result,gs_mean_test_AUC,gs_std_test_AUC,gs_rank_test_AUC,gs_mean_test_Accuracy,gs_std_test_Accuracy,gs_rank_test_Accuracy,gs_mean_test_Recall,gs_std_test_Recall,gs_rank_test_Recall,gs_mean_test_Precision,gs_std_test_Precision,gs_rank_test_Precision,gs_mean_test_F1_Score,gs_std_test_F1_Score,gs_rank_test_F1_Score],axis=1)
    
    df_gs_result.columns=['Model_attributes','mean_test_AUC','std_test_AUC','rank_test_AUC','mean_test_Accuracy','std_test_Accuracy','rank_test_Accuracy','mean_test_Recall','std_test_Recall','rank_test_Recall','mean_test_Precision','std_test_Precision','rank_test_Precision','mean_test_F1_Score','std_test_F1_Score','rank_test_F1_Score']  
```
```python
import numpy as np

#Number of tress in Random Forest
n_estimators = [int(x) for x in np.linspace(start = 200,stop = 1000, num = 5)]

#Number of features to consider while splitting
max_features = ['auto', 'sqrt']

#Maximum number of levels in the tree
max_depth = [int(x) for x in np.linspace(start = 7, stop = 10, num = 4)]

#Minimum # of samples required to split the node
min_samples_split = [10,15]

#Minimum # of samples required at each leaf node
min_samples_leaf = [3,5]

#Method of selecting samples from training each tree
bootstrap = [True, False]

from sklearn.metrics import accuracy_score,make_scorer,precision_score,recall_score,roc_auc_score,f1_score
scoring={'AUC' : make_scorer(roc_auc_score) , 
         'Accuracy':make_scorer(accuracy_score), 
         'Recall':make_scorer(recall_score), 
         'Precision':make_scorer(precision_score),
         'F1 Score':make_scorer(f1_score)}

#Create the grid:
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf ,
               'bootstrap': bootstrap                
              }
   
from sklearn.ensemble import RandomForestClassifier
RFcl = RandomForestClassifier(random_state = 0, n_jobs = -1) 

from sklearn.model_selection import cross_val_score, GridSearchCV
GS_rfc = GridSearchCV(estimator=RFcl, param_grid=random_grid, cv= 10, n_jobs = -1,scoring=scoring,refit='Recall',return_train_score=True)
GS_rfc.fit(X_train, y_train)
print(GS_rfc.best_score_)
#print(GS_rfc.cv_results_)
    
format_grid_search_result(GS_rfc.cv_results_)
df_gs_result.to_excel('Random_forest_Grid_Search_24feb.xlsx')
```
----------

## Introducing Ensemble Learning

In statistics and machine learning, ensemble methods use multiple learning algorithms to obtain better predictive performance than could be obtained from any of the constituent learning algorithms alone.

There are two types of ensemple learnings.

**Bagging/Averaging Methods**

> In averaging methods, the driving principle is to build several estimators independently and then to average their predictions. On average, the combined estimator is usually better than any of the single base estimator because its variance is reduced.

**Boosting Methods**

> The other family of ensemble methods are boosting methods, where base estimators are built sequentially and one tries to reduce the bias of the combined estimator. The motivation is to combine several weak models to produce a powerful ensemble.

## 7f. Bagging Classifier
[Bagging Classifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.BaggingClassifier.html)(Bootstrap Aggregating) is the ensemble method that involves manipulating the training set by resampling and running algorithms on it. Let's do a quick review:

-   Bagging classifier uses a process called bootstrapped dataset to create multiple datasets from one original dataset and runs algorithm on each one of them. Here is an image to show how bootstrapped dataset works.
![](https://uc-r.github.io/public/images/analytics/bootstrap/bootstrap.png)
#### Resampling from original dataset to bootstrapped datasets
-   After running a learning algorithm on each one of the bootstrapped datasets, all models are combined by taking their average. the test data/new data then go through this averaged classifier/combined classifier and predict the output.

Here is an image to make it clear on how bagging works,
![](https://prachimjoshi.files.wordpress.com/2015/07/screen_shot_2010-12-03_at_5-46-21_pm.png)
```python
from sklearn.ensemble import BaggingClassifier
n_estimators = [10,30,50,70,80,150,160, 170,175,180,185];
cv = StratifiedShuffleSplit(n_splits=10, test_size=.30, random_state=15)

parameters = {'n_estimators':n_estimators,
              
        }
grid = GridSearchCV(BaggingClassifier(base_estimator= None, ## If None, then the base estimator is a decision tree.
                                      bootstrap_features=False),
                                 param_grid=parameters,
                                 cv=cv,
                                 n_jobs = -1)
grid.fit(X,y)
```
```
print (grid.best_score_)
print (grid.best_params_)
print (grid.best_estimator_)
```
```python
bagging_grid = grid.best_estimator_
bagging_grid.score(X,y)
```
### Why use Bagging? (Pros and cons)

Bagging works best with strong and complex models(for example, fully developed decision trees). However, don't let that fool you to thinking that similar to a decision tree, bagging also overfits the model. Instead, bagging reduces overfitting since a lot of the sample training data are repeated and used to create base estimators. With a lot of equally likely training data, bagging is not very susceptible to overfitting with noisy data, therefore reduces variance. However, the downside is that this leads to an increase in bias.

#### Random Forest VS. Bagging Classifier

If some of you are like me, you may find Random Forest to be similar to Bagging Classifier. However, there is a fundamental difference between these two which is  **Random Forests ability to pick subsets of features in each node.**  I will elaborate on this in a future update.
## 7g. AdaBoost Classifier[](https://www.kaggle.com/masumrumi/a-statistical-analysis-ml-workflow-of-titanic#7h.-AdaBoost-Classifier)
AdaBoost is another  **ensemble model**  and is quite different than Bagging. Let's point out the core concepts.

> AdaBoost combines a lot of "weak learners"(they are also called stump; a tree with only one node and two leaves) to make classifications.
> 
> This base model fitting is an iterative process where each stump is chained one after the other;  **It cannot run in parallel.**

> **Some stumps get more say in the final classifications than others.**  The models use weights that are assigned to each data point/raw indicating their "importance." Samples with higher weight have a higher influence on the total error of the next model and gets more priority. The first stump starts with uniformly distributed weight which means, in the beginning, every datapoint have an equal amount of weights.
> 
> **Each stump is made by talking the previous stump's mistakes into account.**  After each iteration weights gets re-calculated in order to take the errors/misclassifications from the last stump into consideration.
> 
> The final prediction is typically constructed by a weighted vote where weights for each base model depends on their training errors or misclassification rates.

To illustrate what we have talked about so far let's look at the following visualization.
![](https://cdn-images-1.medium.com/max/1600/0*paPv7vXuq4eBHZY7.png)
Let's dive into each one of the nitty-gritty stuff about AdaBoost:

----------

> **First**, we determine the best feature to split the dataset using Gini index(basics from decision tree). The feature with the lowest Gini index becomes the first stump in the AdaBoost stump chain(the lower the Gini index is, the better unmixed the label is, therefore, better split).
> 
> ----------
> 
> **Secondly**, we need to determine how much say a stump will have in the final classification and how we can calculate that.
> 
> -   We learn how much say a stump has in the final classification by calculating how well it classified the samples (aka calculate the total error of the weight).
> -   The  **Total Error**  for a stump is the sum of the weights associated with the incorrectly classified samples. For example, lets say, we start a stump with 10 datasets. The first stump will uniformly distribute an weight amoung all the datapoints. Which means each data point will have 1/10 weight. Let's say once the weight is distributed we run the model and find 2 incorrect predicitons. In order to calculate the total erorr we add up all the misclassified weights. Here we get 1/10 + 1/10 = 2/10 or 1/5. This is our total error. We can also think about it

$$ \\epsilon_t = \\frac{\\text{misclassifications}_t}{\\text{observations}_t} $$

-   Since the weight is uniformly distributed(all add up to 1) among all data points, the total error will always be between 0(perfect stump) and 1(horrible stump).
-   We use the total error to determine the amount of say a stump has in the final classification using the following formula

$$ \\alpha_t = \\frac{1}{2}ln \\left(\\frac{1-\\epsilon_t}{\\epsilon_t}\\right) \\text{where } \\epsilon_t < 1$$

Where  ϵtϵt  is the misclassification rate for the current classifier:

$$ \\epsilon_t = \\frac{\\text{misclassifications}_t}{\\text{observations}_t} $$

Here...

* $\\alpha_t$ = Amount of Say
* $\\epsilon_t$ = Total error

We can draw a graph to determine the amount of say using the value of total error(0 to 1)
![](http://chrisjmccormick.files.wordpress.com/2013/12/adaboost_alphacurve.png)
-   The blue line tells us the amount of say for  **Total Error(Error rate)**  between 0 and 1.
-   When the stump does a reasonably good job, and the  **total error**  is minimal, then the  **amount of say(Alpha)**  is relatively large, and the alpha value is positive.
-   When the stump does an average job(similar to a coin flip/the ratio of getting correct and incorrect ~50%/50%), then the  **total error**  is ~0.5. In this case the  **amount of say**  is  **0**.
-   When the error rate is high let's say close to 1, then the  **amount of say**  will be negative, which means if the stump outputs a value as "survived" the included weight will turn that value into "not survived."
- P.S. If the  **Total Error**  is 1 or 0, then this equation will freak out. A small amount of error is added to prevent this from happening.
> **Third**, We need to learn how to modify the weights so that the next stump will take the errors that the current stump made into account. The pseducode for calculating the new sample weight is as follows.
$$ New Sample Weight = Sample Weight + e^{\\alpha_t}$$

Here the $\\alpha_t(AmountOfSay)$ can be positive or negative depending whether the sample was correctly classified or misclassified by the current stump. We want to increase the sample weight of the misclassified samples; hinting the next stump to put more emphasize on those. Inversely, we want to decrease the sample weight of the correctly classified samples; hinting the next stump to put less emphasize on those.     
The following equation help us to do this calculation. 

$$ D_{t+1}(i) = D_t(i) e^{-\\alpha_t y_i h_t(x_i)} $$

Here, 
* $D_{t+1}(i)$ = New Sample Weight. 
* $D_t(i)$ = Current Sample weight.
* $\\alpha_t$ = Amount of Say, alpha value, this is the coefficient that gets updated in each iteration and 
* $y_i h_t(x_i)$ = place holder for 1 if stump correctly classified, -1 if misclassified. 

Finally, we put together the combined classifier, which is 

$$ AdaBoost(X) = sign\\left(\\sum_{t=1}^T\\alpha_t h_t(X)\\right) $$ 

Here, 
$AdaBoost(X)$ is the classification predictions for $y$ using predictor matrix $X$
$T$ is the set of \weak learners\
$\\alpha_t$ is the contribution weight for weak learner $t$
$h_t(X)$ is the prediction of weak learner $t$
and $y$ is binary **with values -1 and 1**

P.S. Since the stump barely captures essential specs about the dataset, the model is highly biased in the beginning. However, as the chain of stumps continues and at the end of the process, AdaBoost becomes a strong tree and reduces both bias and variance.
```python
from sklearn.ensemble import AdaBoostClassifier
n_estimators = [100,140,145,150,160, 170,175,180,185];
cv = StratifiedShuffleSplit(n_splits=10, test_size=.30, random_state=15)
learning_r = [0.1,1,0.01,0.5]

parameters = {'n_estimators':n_estimators,
              'learning_rate':learning_r
              
        }
grid = GridSearchCV(AdaBoostClassifier(base_estimator= None, ## If None, then the base estimator is a decision tree.
                                     ),
                                 param_grid=parameters,
                                 cv=cv,
                                 n_jobs = -1)
grid.fit(X,y)
```
```python
print (grid.best_score_)
print (grid.best_params_)
print (grid.best_estimator_)
```
```python
adaBoost_grid = grid.best_estimator_
adaBoost_grid.score(X,y)
```
## Pros and cons of boosting
### Pros[](https://www.kaggle.com/masumrumi/a-statistical-analysis-ml-workflow-of-titanic#Pros)

-   Achieves higher performance than bagging when hyper-parameters tuned properly.
-   Can be used for classification and regression equally well.
-   Easily handles mixed data types.
-   Can use "robust" loss functions that make the model resistant to outliers.

### Cons[](https://www.kaggle.com/masumrumi/a-statistical-analysis-ml-workflow-of-titanic#Cons)

-   Difficult and time consuming to properly tune hyper-parameters.
-   Cannot be parallelized like bagging (bad scalability when huge amounts of data).
-   More risk of overfitting compared to bagging.

## 7h. Gradient Boosting Classifier
```python
# Gradient Boosting Classifier
from sklearn.ensemble import GradientBoostingClassifier

gradient_boost = GradientBoostingClassifier()
gradient_boost.fit(X, y)
y_pred = gradient_boost.predict(X_test)
gradient_accy = round(accuracy_score(y_pred, y_test), 3)
print(gradient_accy)
```
## 7i. XGBClassifier
```python
from xgboost import XGBClassifier
XGBClassifier = XGBClassifier()
XGBClassifier.fit(X, y)
y_pred = XGBClassifier.predict(X_test)
XGBClassifier_accy = round(accuracy_score(y_pred, y_test), 3)
print(XGBClassifier_accy)
```
## 7j. Extra Trees Classifier
```
from sklearn.ensemble import ExtraTreesClassifier
ExtraTreesClassifier = ExtraTreesClassifier()
ExtraTreesClassifier.fit(X, y)
y_pred = ExtraTreesClassifier.predict(X_test)
extraTree_accy = round(accuracy_score(y_pred, y_test), 3)
print(extraTree_accy)
```
## 7k. Gaussian Process Classifier
```python
from sklearn.gaussian_process import GaussianProcessClassifier
GaussianProcessClassifier = GaussianProcessClassifier()
GaussianProcessClassifier.fit(X, y)
y_pred = GaussianProcessClassifier.predict(X_test)
gau_pro_accy = round(accuracy_score(y_pred, y_test), 3)
print(gau_pro_accy)
```
## 7l. Voting Classifier
```python
from sklearn.ensemble import VotingClassifier

voting_classifier = VotingClassifier(estimators=[
    ('lr_grid', logreg_grid),
    ('svc', svm_grid),
    ('random_forest', rf_grid),
    ('gradient_boosting', gradient_boost),
    ('decision_tree_grid',dectree_grid),
    ('knn_classifier', knn_grid),
    ('XGB_Classifier', XGBClassifier),
    ('bagging_classifier', bagging_grid),
    ('adaBoost_classifier',adaBoost_grid),
    ('ExtraTrees_Classifier', ExtraTreesClassifier),
    ('gaussian_classifier',gaussian),
    ('gaussian_process_classifier', GaussianProcessClassifier)
],voting='hard')

#voting_classifier = voting_classifier.fit(train_x,train_y)
voting_classifier = voting_classifier.fit(X,y
```
```python
y_pred = voting_classifier.predict(X_test)
voting_accy = round(accuracy_score(y_pred, y_test), 3)
print(voting_accy)
```
##  Gain Chart
```python

import pickle
import pandas as pd
from sklearn import metrics
import numpy as np
from sklearn.ensemble import RandomForestClassifier

select_var=['Pclass','Child_ind','Fare','Sex_WOE','Dependent_ind_WOE','Embarked_ind_WOE']

X_train=X_train[select_var]
X_test=X_test[select_var]

#clf=pd.read_pickle('decision_tree_24feb2020.pkl')
#clf=pd.read_pickle('random_forest_model_25feb2020_optimized.pkl')
clf=pd.read_pickle('gbm_model_24feb2020_optimized.pkl')

y_pred = clf.predict(X_train)
y_train_score = clf.predict_proba(X_train)
random_forest_model_accuracy = metrics.accuracy_score(y_train,y_pred)
                
print("====== Classification Metrics - Development ======")
print(" Accuracy : "  + str(metrics.accuracy_score(y_train,y_pred)))
print(" Recall : "  + str(metrics.recall_score(y_train,y_pred)))
print(" Precision : "  + str(metrics.precision_score(y_train,y_pred)))
print(" F1_Score : "  + str(metrics.f1_score(y_train,y_pred)))
print(" Confusion_metrics : "  + str(metrics.confusion_matrix(y_train,y_pred)))
print(" ")

y_train_score_df = pd.DataFrame(y_train_score, index=range(y_train_score.shape[0]),columns=range(y_train_score.shape[1]))
y_train_score_df['Actual'] = pd.Series(y_train.values, index=y_train_score_df.index)
y_train_score_df['Predicted'] = pd.Series(y_pred, index=y_train_score_df.index)
y_train_score_df['Decile'] = pd.qcut(y_train_score_df[1],10,duplicates='drop')

lift_tbl = pd.DataFrame([y_train_score_df.groupby('Decile')[1].min(),
                                                 y_train_score_df.groupby('Decile')[1].max(),
                                                 y_train_score_df[(y_train_score_df['Actual'] == 1)].groupby('Decile')[1].count(),
                                                 y_train_score_df[(y_train_score_df['Actual'] == 0)].groupby('Decile')[1].count(),
                                                 y_train_score_df.groupby('Decile')[1].count()]).T
lift_tbl.columns = ["MIN","MAX","Event","Non-Event","TOTAL"]
lift_tbl = lift_tbl.sort_values("MIN", ascending=False)
lift_tbl = lift_tbl.reset_index()

list_vol_pct=[]
list_event_pct=[]

for i in range(len(lift_tbl.Event)):
    list_vol_pct.append(lift_tbl['TOTAL'][i]/lift_tbl['TOTAL'].sum())
    list_event_pct.append(lift_tbl['Event'][i]/lift_tbl['TOTAL'][i])

lift_tbl = pd.concat([lift_tbl,pd.Series(list_vol_pct),pd.Series(list_event_pct)],axis=1)


lift_tbl = lift_tbl[["Decile","MIN","MAX","Event","Non-Event","TOTAL",0,1]]        
lift_tbl = lift_tbl.rename(columns={lift_tbl.columns[len(lift_tbl.keys())-2]: "Volume(%)"})
lift_tbl = lift_tbl.rename(columns={lift_tbl.columns[len(lift_tbl.keys())-1]: "Event(%)"})

lift_tbl["Cumm_Event"] = lift_tbl["Event"].cumsum()
lift_tbl["Cumm_Event_Pct"] = lift_tbl["Cumm_Event"] / lift_tbl["Event"].sum()
#lift_tbl
lift_tbl.to_excel("Titanic_Lift_Chart_optimized_gbm_25feb2020.xlsx", index = None, header=True)
lift_tbl
```
