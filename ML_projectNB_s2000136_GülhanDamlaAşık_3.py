# Machine Learning Homework
# Gülhan Damla AŞIK - 2000136

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import itertools

from sklearn.model_selection import train_test_split, GridSearchCV, LeaveOneOut
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.utils import shuffle
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedShuffleSplit
from scipy.stats import norm, gaussian_kde
from sklearn.neighbors import KernelDensity
from sklearn.metrics import classification_report
from scipy import stats
from statsmodels.distributions.mixture_rvs import mixture_rvs
import statsmodels.api as sm

import os
os.getcwd()
# os.chdir()

### READ DATA
abaloneDF = pd.DataFrame(np.loadtxt("C:/Users/user/Desktop/BAU Lessons/2- Machine Learning and Pattern Recognition/Homeworks/abalone_dataset.txt", skiprows=1, dtype=str))
abaloneDF.shape[0]
# 4176 rows

abaloneDF.info()

### PREPARE DATA
abaloneDF.columns = ["Sex", "Length", "Diameter", "Height", "Whole weight", "Shucked weight","Viscera weight", "Shell weight", "Group"]
abaloneDF["Length"] = pd.to_numeric(abaloneDF["Length"])
abaloneDF["Diameter"] = pd.to_numeric(abaloneDF["Diameter"])
abaloneDF["Height"] = pd.to_numeric(abaloneDF["Height"])
abaloneDF["Whole weight"] = pd.to_numeric(abaloneDF["Whole weight"])
abaloneDF["Shucked weight"] = pd.to_numeric(abaloneDF["Shucked weight"])
abaloneDF["Viscera weight"] = pd.to_numeric(abaloneDF["Viscera weight"])
abaloneDF["Shell weight"] = pd.to_numeric(abaloneDF["Shell weight"])
abaloneDF["Group"] = pd.to_numeric(abaloneDF["Group"])

abaloneDF.groupby("Sex")["Group"].count()
# Sex
# F    1307
# I    1342
# M    1527

abaloneDF.describe()
abaloneDF.info()

# 1) Assume gaussian distribution for continuous features.
abaloneDF2 = abaloneDF
le = preprocessing.LabelEncoder()
abaloneDF2["Sex"] = le.fit_transform(abaloneDF2["Sex"])

x = abaloneDF2.drop(["Group"], axis = 1)
y = abaloneDF2.iloc[:, -1]
size = 100
splitter=StratifiedShuffleSplit(n_splits=5,test_size=(round(1-size/abaloneDF2.shape[0], 5)) ,random_state=109) 

for train,test in splitter.split(x,y):
    x_train = x.iloc[train]
    y_train = y.iloc[train]
    x_test = x.iloc[test]
    y_test = y.iloc[test]
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
# (100, 8) (4076, 8) (100,) (4076,)

gnb = GaussianNB()
a = gnb.fit(x_train,y_train)
b_pred = gnb.predict(x_test)
print("Accuracy:",accuracy_score(y_test, b_pred))
# Accuracy: 0.6263493621197253

confusion = confusion_matrix(y_test, b_pred)
print('Confusion Matrix\n')
print(confusion)
# [[548  87   3]
# [326 944 539]
# [ 45 330 355]]

print("Total misclassification errors:")
c_matrix1 = confusion_matrix(y_test, b_pred)
classification_errors = c_matrix1[0][1]  + c_matrix1[0][2] + c_matrix1[1][0] + c_matrix1[1][2] + c_matrix1[2][0] + c_matrix1[2][1]
print(classification_errors)
# Total misclassification errors:
# 1523

print('Micro Precision: {:.2f}'.format(precision_score(y_test, b_pred, average='micro')))
print('Micro Recall: {:.2f}'.format(recall_score(y_test, b_pred, average='micro')))
print('Micro F1-score: {:.2f}\n'.format(f1_score(y_test, b_pred, average='micro')))
# Micro : It does not consider each class individually, It calculates the metrics globally.

print('Macro Precision: {:.2f}'.format(precision_score(y_test, b_pred, average='macro')))
print('Macro Recall: {:.2f}'.format(recall_score(y_test, b_pred, average='macro')))
print('Macro F1-score: {:.2f}\n'.format(f1_score(y_test, b_pred, average='macro')))
# Macro: It calculates metrics for each class individually and then takes unweighted mean of the measures.

print('\nClassification Report\n')
print(classification_report(y_test, b_pred, target_names=['1', '2', '3']))

size = 1000
splitter=StratifiedShuffleSplit(n_splits=5,test_size=(round(1-size/abaloneDF2.shape[0], 5)) ,random_state=109) 

for train,test in splitter.split(x,y):
    x_train = x.iloc[train]
    y_train = y.iloc[train]
    x_test = x.iloc[test]
    y_test = y.iloc[test]
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
# (999, 8) (3177, 8) (999, 1) (3177, 1)

gnb = GaussianNB()
a = gnb.fit(x_train,y_train)
b_pred = gnb.predict(x_test)
print("Accuracy:",accuracy_score(y_test, b_pred))
# Accuracy: 0.5810513062637709

confusion = confusion_matrix(y_test, b_pred)
print('Confusion Matrix\n')
print(confusion)
# [[546  89   3]
# [321 943 545]
# [ 37 336 357]]

print("Total misclassification errors:")
c_matrix1 = confusion_matrix(y_test, b_pred)
classification_errors = c_matrix1[0][1]  + c_matrix1[0][2] + c_matrix1[1][0] + c_matrix1[1][2] + c_matrix1[2][0] + c_matrix1[2][1]
print(classification_errors)
# Total misclassification errors:
# 1331

print('Micro Precision: {:.2f}'.format(precision_score(y_test, b_pred, average='micro')))
print('Micro Recall: {:.2f}'.format(recall_score(y_test, b_pred, average='micro')))
print('Micro F1-score: {:.2f}\n'.format(f1_score(y_test, b_pred, average='micro')))
# Micro : It does not consider each class individually, It calculates the metrics globally.

print('Macro Precision: {:.2f}'.format(precision_score(y_test, b_pred, average='macro')))
print('Macro Recall: {:.2f}'.format(recall_score(y_test, b_pred, average='macro')))
print('Macro F1-score: {:.2f}\n'.format(f1_score(y_test, b_pred, average='macro')))
# Macro: It calculates metrics for each class individually and then takes unweighted mean of the measures.

print('\nClassification Report\n')
print(classification_report(y_test, b_pred, target_names=['1', '2', '3']))


# 2) Use Naive Estimator for each of the continuous feature.
def plot_confusion_matrix(cm, groups,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(groups))
    plt.xticks(tick_marks, groups, rotation=45)
    plt.yticks(tick_marks, groups)

    fmt = 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
def split_data(df,split_size):
    size = (str(round(1-split_size/df.shape[0], 5)))
    Stratified = StratifiedShuffleSplit(n_splits=5, test_size=float(size), random_state=109)
    Stratified.get_n_splits(df)
    for train_index, test_index in Stratified.split(df,df["Group"]):
        train = df.loc[train_index]
        validation = df.loc[test_index]
    return train,validation

def gaussian_distribution(train,index,value,distinct_class):
    mean = np.mean(train[(train["Group"] == distinct_class)][index])
    std = np.std(train[(train["Group"] == distinct_class)][index])
    return norm.cdf(value, mean, std)

def calculate_class_probablity(train,distinct_class):
    prob = train[train["Group"]==distinct_class].shape[0]/train["Group"].shape[0]
    return prob

def calculate_category_probability(train,index,value,distinct_class):
    sub_count = train[(train["Group"]==distinct_class) & (train[index]==value)].shape[0]
    count = train[train["Group"]==distinct_class]["Group"].shape[0]
    prob = sub_count/count
    return prob

def calculate_probability(train,val_row):
    max_class = -1
    max_value = -1
    for distinct_class in np.sort(train.Group.unique()):
        vector = 1
        for index,val in val_row.iteritems():
            if train[index].dtypes == np.float64:
                vector = vector * gaussian_distribution(train,index,val,distinct_class)
            elif train[index].dtypes == np.int64:
                vector = vector * calculate_class_probablity(train,distinct_class)
            else:
                vector = vector * calculate_category_probability(train,index,val,distinct_class)
        if vector > max_value:
            max_class = distinct_class
            max_value = vector
    return(max_class)

def calculate_likelihoods(train,validation):
    y_pred = []
    for i in range(len(validation.index)):
        y_pred.append(calculate_probability(train,validation.iloc[i]))
    return pd.Series(y_pred) 

def naive_bayes(df,size):
    train,validation = split_data(df,size)
    y_pred = calculate_likelihoods(train,validation)
    c_matrix = confusion_matrix(validation["Group"], y_pred)
    print("Confusion Matrix:")
    print(c_matrix)
    
    print("Total misclassification errors:")
    classification_errors = c_matrix[0][1]  + c_matrix[0][2] + c_matrix[1][0] + c_matrix[1][2] + c_matrix[2][0] + c_matrix[2][1]
    print(classification_errors)

    print("Accuracy:")
    accuracy = ((c_matrix[0][0] + c_matrix[1][1] + c_matrix[2][2]) / (len(df))) * 100
    print("%s%%" % accuracy)

    plt.figure()
    plot_confusion_matrix(c_matrix, groups=["young", "middle-aged", "old"],title='Confusion Matrix')
    plt.show()

naive_bayes(abaloneDF,100)
# Accuracy: 35.32088122605364%

naive_bayes(abaloneDF,1000)
# Accuracy: 28.59195402298851%


# 3) Use Kernel Estimator for each of the continuous feature.

abaloneNP = abaloneDF2.to_numpy()
kde = KernelDensity(kernel='gaussian', bandwidth=0.2).fit(abaloneNP)
kde.score_samples(abaloneNP)
# array([1.87628001, 3.18354209, 3.13416969, ..., 3.62789912, 3.52970297,  0.9263579 ])

# optimal bandwidth selection
grid = GridSearchCV(KernelDensity(),
                    {'bandwidth': np.linspace(0.1, 1.0, 30)},
                    cv=20) # 20-fold cross-validation
grid.fit(y[:, None])
bandwidth = grid.best_params_
print(bandwidth)
# {'bandwidth': 0.1}


#### Plot 1 - Kernel   
# Location, scale and weight for the two distributions (kernel&hist)
dist1_loc, dist1_scale, weight1 = -1 , .5, .25
dist2_loc, dist2_scale, weight2 = 1 , .5, .75

# Sample from a mixture of distributions
obs_dist = mixture_rvs(prob=[weight1, weight2], size=100,
dist=[stats.norm, stats.norm],
kwargs = (dict(loc=dist1_loc, scale=dist1_scale),
dict(loc=dist2_loc, scale=dist2_scale)))
kde = sm.nonparametric.KDEUnivariate(obs_dist)
kde.fit() # Estimate the densities
fig = plt.figure(figsize=(12, 5))
ax = fig.add_subplot(111)

# Plot the histrogram
ax.hist(obs_dist, bins=20, density=True, label='Histogram from samples',
        zorder=5, edgecolor='k', alpha=0.5)

# Plot the KDE as fitted using the default arguments
ax.plot(kde.support, kde.density, lw=3, label='KDE from samples', zorder=10)

# Plot the true distribution
true_values = (stats.norm.pdf(loc=dist1_loc, scale=dist1_scale, x=kde.support)*weight1
              + stats.norm.pdf(loc=dist2_loc, scale=dist2_scale, x=kde.support)*weight2)
ax.plot(kde.support, true_values, lw=3, label='True distribution', zorder=15)

# Plot the samples
ax.scatter(obs_dist, np.abs(np.random.randn(obs_dist.size))/40,
           marker='x', color='red', zorder=20, label='Samples', alpha=0.5)

ax.legend(loc='best')
ax.grid(True, zorder=-5)


#### Plot 2 # bandwidth comparison
# Location, scale and weight for the two distributions
dist1_loc, dist1_scale, weight1 = -1 , .5, .25
dist2_loc, dist2_scale, weight2 = 1 , .5, .75

# Sample from a mixture of distributions
obs_dist = mixture_rvs(prob=[weight1, weight2], size=100,
dist=[stats.norm, stats.norm],
kwargs = (dict(loc=dist1_loc, scale=dist1_scale),
dict(loc=dist2_loc, scale=dist2_scale)))
kde = sm.nonparametric.KDEUnivariate(obs_dist)
kde.fit() # Estimate the densities
fig = plt.figure(figsize=(12, 5))
ax = fig.add_subplot(111)

# Plot the histrogram
ax.hist(obs_dist, bins=25, label='Histogram from samples',
zorder=5, edgecolor='k', density=True, alpha=0.5)

# Plot the KDE for various bandwidths
for bandwidth in [0.1, 0.2, 0.4]:
    kde.fit(bw=bandwidth) # Estimate the densities
    ax.plot(kde.support, kde.density, '--', lw=2, color='k', zorder=10,
            label='KDE from samples, bw = {}'.format(round(bandwidth, 2)))

# Plot the true distribution
true_values = (stats.norm.pdf(loc=dist1_loc, scale=dist1_scale, x=kde.support)*weight1
+ stats.norm.pdf(loc=dist2_loc, scale=dist2_scale, x=kde.support)*weight2)
ax.plot(kde.support, true_values, lw=3, label='True distribution', zorder=15)

# Plot the samples
ax.scatter(obs_dist, np.abs(np.random.randn(obs_dist.size))/50,
marker='x', color='red', zorder=20, label='Data samples', alpha=0.5)

ax.legend(loc='best')
ax.set_xlim([-3, 3])
ax.grid(True, zorder=-5)


# Some info about Naive Estimator :It is an estimate of the slope merely by joining the first and last observations, 
# and by dividing the increase in the height by the horizontal distance between them.
# A Gaussian distribution can be summarized using only two numbers: the mean and the standard deviation. 
# This piece of math is called a Gaussian Probability Distribution Function (or Gaussian PDF) and can be calculated as:
# f(x) = (1 / sqrt(2 * PI) * sigma) * exp(-((x-mean)^2 / (2 * sigma^2)))
# Where sigma is the standard deviation for x, mean is the mean for x and PI is the value of pi.
# ADVANTAGES:
    # 1) It is easy and fast to predict the class of the test data set. It also performs well in multi-class prediction.
    # 2) When assumption of independence holds, model performs better compare to other models like logistic regression 
    # and you need less training data.
# DISADVANTAGES:
    # 1) If categorical variable has a category (in test data set), which was not observed in training data set, 
    #   then model will assign a 0 (zero) probability and will be unable to make a prediction. 
    #   This is often known as Zero Frequency. To solve this, we can use the smoothing technique. One of the simplest 
    #   smoothing techniques is called Laplace estimation.
    # 2) Another limitation of Naive Bayes is the assumption of independent predictors. 
    #   In real life, it is almost impossible that we get a set of predictors which are completely independent.
# When to use: Text Classification, when dataset is huge, when you have small training set.


# Some info about Kernel Estimator : All atributes ave some impact with different weights (If sample is around the 
# given sample, it should make a higher contribution). Only for numerical variables. Also called Gaussian Kernel. 
# If complexcity increases, bias decreases. 
# The bandwidth here acts as a smoothing parameter, controlling the tradeoff between bias and variance in the result. 
# A large bandwidth leads to a very smooth (i.e. high-bias) density distribution. A small bandwidth leads to an unsmooth 
# (i.e. high-variance) density distribution.
# HOW IT WORKS: The Kernel Density Estimation works by plotting out the data and beginning to create a curve of the 
# distribution. The curve is calculated by weighing the distance of all the points in each specific location along the 
# distribution. If there are more points grouped locally, the estimation is higher as the probability of seeing a point 
# at that location increases. The kernel function is the specific mechanism used to weigh the points across the data set. 
# The bandwidth of the kernel changes its shape. A lower bandwidth limits the scope of the function and leads to the 
# estimate curve looking rough and jagged. By tweaking the parameters of the kernel function (bandwidth and amplitude), 
# one changes the size and shape of the estimate.

