###################################
# Data Science 2019 Final Project #
###########################################################################################
# Note: This is the python code that was used in the Jupyter Notebook                     #
#       Therefore it will not run properly here.                                          #
#       Each section was run individually, and will require some rework to work as before #
###########################################################################################

# Import Libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
%pylab inline

# Plotly Tools
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.offline as offline
offline.init_notebook_mode()
from plotly import tools
import plotly.tools as tls
init_notebook_mode(connected=True)

# Import cufflinks and offline mode to get plotly working
import cufflinks as cf
cf.go_offline()

# Import the formula tools for prediction/modeling
import statsmodels.formula.api as smf

# Include SciPy Packages
from scipy import stats
from scipy.stats import reciprocal
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit

# Sklearn splitting/modeling data
from sklearn import model_selection, metrics
from sklearn.model_selection import train_test_split

#Import CSV as a Pandas Dataframe
fp = pd.read_csv("gun-violence-data_01-2013_03-2018.csv")

#Confirm that dataset was properly loaded
print(fp.head())

################################################### Section Divisor ###################################################

print(fp.shape)

################################################### Section Divisor ###################################################

#List of Columns
print(fp.columns)

################################################### Section Divisor ###################################################

# Clean the Dataset
#Removing unnecessary columns
fp_clean = fp.loc[:,['date','state','city_or_county','n_killed','n_injured','n_guns_involved']]

#Replace NaNs in "n_guns_involved" with 1, because at least 1 gun was involved
fp_clean["n_guns_involved"] = fp_clean["n_guns_involved"].fillna(1)

#Replace NaN's in the columns with 0
fp_clean = fp_clean.fillna(0)

#Convert # of guns from floating point to integer
fp_clean['n_guns_involved'] = fp_clean['n_guns_involved'].astype('int')

# Create some additional features - make it into a time series
fp_clean['injured_killed'] = fp_clean['n_killed'] + fp_clean['n_injured']

fp_clean['date'] = pd.to_datetime(fp_clean['date'])
fp_clean['year'] = fp_clean['date'].dt.year
fp_clean['month'] = fp_clean['date'].dt.month
fp_clean['weekday'] = fp_clean['date'].dt.weekday

print(fp_clean.head())

################################################### Section Divisor ###################################################

# Numerical Information on some of the Data
print(fp_clean.describe())

################################################### Section Divisor ###################################################

# Interesting Data Analysis
# Plot Cleaned Data
temp = fp_clean["state"].value_counts().head(30)
temp.iplot(kind='bar', xTitle = 'State name', yTitle = "Number of Incidents",
           title = 'Top 30 States with highest number of Gun Violence', filename='Bar')

################################################### Section Divisor ###################################################

# Pairplot Graph
# Checking for any relationship on the data
sns.pairplot(fp_clean)

################################################### Section Divisor ###################################################

# Split Data into Testing and Training Sets
# Split data into testing set and training set from Sklearn
train_set,test_set = train_test_split(fp_clean,test_size = 0.3,random_state = 0)

# Confirming ~30% for test - Note: 239677 total incidents, so 239677*0.3 = ~71,903
print(train_set.shape)
print(test_set.shape)

plt.scatter(train_set['n_guns_involved'], train_set['injured_killed'], color='green')
plt.scatter(test_set['n_guns_involved'], test_set['injured_killed'], color='violet')
plt.title('Number of Guns vs. Number of People Killed and Injured')
plt.xlabel('Number of Guns')
plt.ylabel('Sum of Killed and Injured')

################################################### Section Divisor ###################################################

# Different Prediction Models
#Let's start with a simple linear prediction model:
result = smf.ols(data = train_set, formula = "n_guns_involved ~ injured_killed").fit()
print(result.summary())

################################################### Section Divisor ###################################################

#Try with an exponential function:
result2 = smf.ols(data = train_set, formula = "n_guns_involved ~ np.exp(-injured_killed)").fit()
print(result2.summary())

################################################### Section Divisor ###################################################

#let's see what Logarithmic function is like
#Note: Had to remove 0s since log(0)=infinity
trainNo0=train_set[train_set.injured_killed > 0]

result3 = smf.ols(data = trainNo0, formula = "n_guns_involved ~ np.log(1/injured_killed)").fit()
print(result3.summary())

################################################### Section Divisor ###################################################

# Plot the Prediction Graphs vs. the Original Graph
#ik_temp = pd.DataFrame({"injured_killed": np.arange(0,60.0,0.01)})
test_set['Linear_prediction'] = result.predict(test_set)
test_set['Exponential_prediction'] = result2.predict(test_set)
test_set['Logarithmic_prediction'] = result3.predict(test_set)

plt.scatter(train_set['n_guns_involved'], train_set['injured_killed'], color='black')
plt.plot(test_set['injured_killed'], test_set['Linear_prediction'], color = 'red')
plt.plot(test_set['injured_killed'], test_set['Exponential_prediction'], color = 'orange')
plt.plot(test_set['injured_killed'], test_set['Logarithmic_prediction'], color = 'green')
plt.title('Number of Guns vs. Number of People Killed and Injured')
plt.xlabel('Number of Guns')
plt.ylabel('Sum of Killed and Injured')
plt.legend()

################################################### Section Divisor ###################################################

# Plot Reciprocal Function vs Original Data
# Creating the reciprocal function (1/x) from:
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.reciprocal.html#scipy.stats.reciprocal
a, b = 0.0022, 450
r = reciprocal.rvs(a, b, size=1000)

x = np.linspace(reciprocal.ppf(0.01, a, b),reciprocal.ppf(0.99, a, b), 100)

plt.scatter(train_set['n_guns_involved'], train_set['injured_killed'], color='black')
plt.plot(x, reciprocal.pdf(x, a, b),'r-', alpha=0.6, label='reciprocal pdf', color = 'orange')
plt.title('Number of Guns vs. Number of People Killed and Injured')
plt.xlabel('Number of Guns')
plt.ylabel('Sum of Killed and Injured')

################################################### Section Divisor ###################################################
# Try a Polynomial Fit
from scipy.interpolate import interp1d

int_q, sl_q = np.polynomial.polynomial.polyfit(train_set['n_guns_involved'], train_set['injured_killed'], 1)
p0, p1, p2 = np.polynomial.polynomial.polyfit(train_set['n_guns_involved'], train_set['injured_killed'], 2)

x_r = np.arange(0,5,0.01)

scatter(train_set['n_guns_involved'], train_set['injured_killed'],color='green', label='Train Set')
scatter(test_set['n_guns_involved'], test_set['injured_killed'],color='blue', label='Test Set')
plt.plot(int_q + sl_q*x_r , color='red',label='Linear');
plt.plot(p0 + p1*x_r + p2*(x_r**2), color='orange',label='Quadratic');
plt.title('Number of Guns vs. Number of People Killed and Injured')
plt.xlabel('Number of Guns')
plt.ylabel('Sum of Killed and Injured')
plt.legend()
plt.grid()

################################################### Section Divisor ###################################################
# Try Linear Regression using Different Model
slope, intercept, r_value, p_value, slope_std_error = \
    stats.linregress(train_set['n_guns_involved'],train_set['injured_killed'])

train_regr_1 = intercept + slope * train_set['n_guns_involved']

scatter(train_set['n_guns_involved'],train_set['injured_killed'],label='Train Data', color='green')
scatter(test_set['n_guns_involved'],test_set['injured_killed'],color='blue',label='Test Data')
plt.plot(x,intercept + slope*x, color='red',label='Regression Model')
plt.title('Number of Guns vs. Number of People Killed and Injured')
plt.xlabel('Number of Guns')
plt.ylabel('Sum of Killed and Injured')
plt.legend()

################################################### Section Divisor ###################################################
#Compute R^2 and MAE on TRAINING
print(metrics.r2_score(train_set['injured_killed'], train_regr_1))
print(metrics.mean_absolute_error(train_set['injured_killed'], train_regr_1))

#Compute R^2 and MAE on TEST Set
print(metrics.r2_score(test_set['injured_killed'], intercept + slope*test_set['n_guns_involved']))
print(metrics.mean_absolute_error(test_set['injured_killed'], intercept + slope*test_set['n_guns_involved']))

################################################### Section Divisor ###################################################
# Last attempt - trying with Gaussian Function from:
# https://stackoverflow.com/questions/46497892/non-linear-regression-in-seaborn-python?rq=1
# Using Test set here instead of Train, because train caused "Max number of calls reached" error
model = lambda x, A, x0, sigma, offset:  offset+A*np.exp(-((x-x0)/sigma)**2)
popt, pcov = curve_fit(model, test_set["n_guns_involved"].values, test_set["injured_killed"].values, p0=[1,0,2,0])

x = np.linspace(test_set["n_guns_involved"].values.min(),test_set["n_guns_involved"].values.max(),250)

#Model 2
model2 = lambda x, sigma:  model(x,1,0,sigma,0)
popt2, pcov2 = curve_fit(model2, test_set["n_guns_involved"].values,test_set["injured_killed"].values, p0=[2])

x2 = np.linspace(test_set["n_guns_involved"].values.min(),test_set["n_guns_involved"].values.max(),250)

plt.scatter(test_set['n_guns_involved'], test_set['injured_killed'],color='black')
plt.plot(x,model(x,*popt), label="Model 1", color='orange')
plt.plot(x2,model2(x2,*popt2), label="Model 2", color='green')
plt.title('Number of Guns vs. Number of People Killed and Injured')
plt.xlabel('Number of Guns')
plt.ylabel('Sum of Killed and Injured')
plt.legend()

################################################### Section Divisor ###################################################
# Conclusion:
#  - Reject the Null Hypothesis
#  - No correlation between the number of guns involved and the number of people who were injured and killed