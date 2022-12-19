import sys
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.patches as patches
from scipy import stats
from statistics import mode
from sklearn.metrics import mean_squared_error, mean_absolute_error
import random
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier


def std_scale(data,f):
    data = data.copy()
    scaler = StandardScaler()
    data.iloc[:,0:f] = scaler.fit_transform(data.iloc[:,0:f])
    # self.scaler = scaler
    return data, scaler



def box_dim(data, K, f):

    x_max = np.zeros((K,f+1))
    x_min = np.zeros((K,f+1))

    for i in range(K):

        x_min[i,:] = np.min(data[data['model'] == i+1].iloc[:,0:f+1], axis = 0)
        x_max[i,:] = np.max(data[data['model'] == i+1].iloc[:,0:f+1], axis = 0)
        
    
    
    return x_min,x_max


def centroids(data,K,f,with_y = False):

    centroid = []

    if with_y:
        for i in range(K):
            centroid.append(np.average(data[data['model'] == i+1].iloc[:, 0:f+1],axis=0))
    else:
        for i in range(K):
            centroid.append(np.average(data[data['model'] == i+1].iloc[:, 0:f],axis=0))

    return centroid


def medians(data,K,f,with_y = False):

    centroid = []

    if with_y:
        for i in range(K):
            centroid.append(np.median(data[data['model'] == i+1].iloc[:, 0:f+1],axis=0))
    else:
        for i in range(K):
            centroid.append(np.median(data[data['model'] == i+1].iloc[:, 0:f],axis=0))

    return centroid


def features_centroid(data,K,f):
    

    centroid = centroids(data, K, f, with_y = True)

    features = list(data.columns[:f+1])

    feature_centroid = pd.DataFrame(features,columns = ['feature'])

    for i in range(K):
        feature_centroid['K='+str(i+1)] = np.around(centroid[i],decimals=2)

    return feature_centroid


def features_boundingbox(data, K, f):


    features = list(data.columns[:f+1])

    feature_box = pd.DataFrame(features,columns = ['Features'])

    X_MIN,X_MAX = box_dim(data,K,f)

    for i in range(K):
        bound_str = list()
        for j in range(f+1):
            bound_str.append(str(np.around(X_MIN[i,j],2)) + ' - ' + str(np.around(X_MAX[i,j],2)) )
        feature_box['Cluster '+str(i+1)] = bound_str

    return feature_box




def features_weights(data, K, f, opt_list):

    data = data.copy()
    features = list(data.columns[:f])

    feature_weight = pd.DataFrame(features,columns = ['Features'])

    for i in range(K):
        feature_weight['Cluster '+str(i+1)] = np.reshape(np.around(opt_list[i].coef_,decimals=3),(f,1))

    return feature_weight


def initialize(data, K, f, KM_intialize = False, randstate = 123):
    data = data.copy()

    if KM_intialize == False:
        np.random.RandomState(randstate)
        data['model'] = np.random.randint(1, high = K+1, size=data.shape[0])
    
    else:
        kmeans = KMeans(n_clusters=K, init= 'k-means++' , random_state=randstate).fit(
            data.iloc[:, 0:f])
        kmeans_model = kmeans.labels_
        kmeans_model = list(map(lambda x: x + 1, kmeans_model))
        data = data.assign(model=kmeans_model)


    return data


def cluster_assign_KNN(data, K, f, test, k_neighbors = 5):

    data = data.copy()
    knn = KNeighborsClassifier(n_neighbors=k_neighbors)
    knn.fit(data.iloc[:,0:f], data['model'])
    
    test_model = knn.predict(test.iloc[:,0:f])

    return test_model



def predict_knn(val_data,  K, f, opt_list , train= None, k_neighbors = 0 ):
    
    data =  val_data.copy()
    if k_neighbors != 0:
        data['model'] = cluster_assign_KNN(train, K, f, data, k_neighbors)

    for i in range(K):
        if (len(data[data['model'] == i+1]) > 0) & (opt_list[i] is not None):
            data.loc[data['model'] == i+1 , 'pred'] = opt_list[i].predict(data[data['model'] == i+1].iloc[:, 0:f])
        else:
            data.loc[data['model'] == i+1 , 'pred'] = data[data['model'] == i+1].iloc[:, f]


    return data


def validate_knn(val_data, K, f, opt_list,train= None, k_neighbors = 0):
    
    data =  val_data.copy()
    data = predict_knn(data,K,f, opt_list, train, k_neighbors)
    total_MSEerror = 0
    r2_score = np.zeros((K,1))

    for i in range(K):
        # print(opt_list[i] is not None)
        if (len(data[data['model'] == i+1]) > 0) & (opt_list[i] is not None):
            r2_score[i,0] = opt_list[i].score(data[data['model'] == i+1].iloc[:, 0:f],
                            data[data['model'] == i+1].iloc[:, f:f+1])
        
            
            total_MSEerror = total_MSEerror + sum( (data.loc[data['model'] == i+1 , 'pred'] -  data[data['model'] == i+1].iloc[:, f]) ** 2  )   
        else:
            r2_score[i,0] = 1

    total_MSEerror = total_MSEerror/data.shape[0]


    return data, total_MSEerror, r2_score 



def mse(pred, test):
    return (mean_squared_error(pred, test))

def mae(pred, test):
    return (mean_absolute_error(pred, test))

def r2_score(y_true, y_pred):


    numerator = ( (y_true - y_pred) ** 2).sum(axis=0)
    denominator = ( (y_true - np.average(y_true, axis=0)) ** 2).sum(axis=0)
    nonzero_denominator = denominator != 0
    nonzero_numerator = numerator != 0
    valid_score = nonzero_denominator & nonzero_numerator
    output_scores = np.ones(1)
    output_scores[valid_score] = 1 - (numerator[valid_score] / denominator[valid_score])
    # arbitrary set to zero to avoid -inf scores, having a constant
    # y_true is not interesting for scoring a regression anyway
    output_scores[nonzero_numerator & ~nonzero_denominator] = 0.0
    return np.average(output_scores)
