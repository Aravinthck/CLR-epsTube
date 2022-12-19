
from abc import ABC, abstractmethod
from SupClus_Greedy.utils import *
from sklearn import linear_model, svm



class SupervisedLoss(ABC):


    
    @abstractmethod
    def optimize_loss(self, data, K, f ):

        raise NotImplementedError



class LinearRegression(SupervisedLoss):


    def __init__(self, loss = 'squared_error',
                 regularize_param = 0.01,
                 fit_intercept=True):
        """
    
        :param regularize_param:    
        :param fit_intercept:       
        """
        self.loss = loss
        self.regularize_param = regularize_param
        self.fit_intercept = fit_intercept


    def optimize_loss(self, data, K, f):

        data = data.copy()
        # initialize k linear regression models

        opt_list = []
        for i in range(K):
            if self.loss == 'squared_error':
                temp_reg = linear_model.Ridge(
                    alpha=self.regularize_param, fit_intercept=self.fit_intercept, max_iter=None, normalize=False, random_state=None, solver='auto')
            else: 
                temp_reg = linear_model.QuantileRegressor(quantile=0.5, alpha=self.regularize_param, fit_intercept=self.fit_intercept)   
            opt_list.append(temp_reg)

        
        for i in range(K):
            if len(data[data['model'] == i+1]) > 0:
                opt_list[i].fit(data[data['model'] == i+1].iloc[:, 0:f],
                                data[data['model'] == i+1].iloc[:, f:f+1])

        data , _, score = validate_knn(data, K, f, opt_list)
        total_loss = 0

        for i in range(K):
            if len(data[data['model'] == i+1]) > 0:
                tmp = opt_list[i].predict(data.iloc[:, 0:f])
                if self.loss == 'squared_error':
                    data['loss'+str(i+1)] =  (tmp - data.iloc[:, f:f+1]) ** 2
                else:
                    data['loss'+str(i+1)] =  abs(tmp - data.iloc[:, f]) 

                total_loss = total_loss + sum(data.loc[data['model'] == i+1 , 'loss'+str(i+1)] )

        total_loss = total_loss/data.shape[0] 

        return data, total_loss, score, opt_list





class LinearClassification(SupervisedLoss):


    def __init__(self, 
                 C = 1e5,
                 kernel = 'linear'):
        """
        A SGD optimizer (optionally with nesterov/momentum).
        :param C:                   
        :param kernel:              
        """

        self.C = C
        self.kernel = kernel
        # self.fit_intercept = fit_intercept





    def optimize_loss(self, data, K, f):

        data = data.copy()
        # n_classes = int(max(data['y']))
        # initialize k linear classificatiion models

        opt_list = []
        for i in range(K):
            temp_opt = svm.LinearSVC(C= self.C, max_iter=3000)
            
            opt_list.append(temp_opt)   


        for i in range(K):
            if len(data[data['model'] == i+1]) > 0:
                cls = data.loc[data['model'] == i+1,'y'].unique()
                cls.sort()
                n_classes = cls.shape[0]
                if n_classes > 1:
                    opt_list[i].fit(data[data['model'] == i+1].iloc[:, 0:f],
                                    data[data['model'] == i+1].iloc[:, f:f+1])
                else: 
                    opt_list[i] = None
                # print(opt_list[i].coef_, opt_list[i].intercept_)

        data, _, score = validate_knn(data, K, f, opt_list)

        total_loss = 0
            
        for i in range(K):
            if len(data[data['model'] == i+1]) > 0:
                cls = data.loc[data['model'] == i+1,'y'].unique()
                cls.sort()
                # print(cls)
                n_classes = cls.shape[0]
                
                if n_classes > 2:
                    k=0
                    # print(n_classes)
                    for j in cls:
                        data.loc[ data['y'] == j,'loss'+str(i+1)] = np.maximum(np.zeros((data[data['y']==j].shape[0],1)) , 
                                        (1 - opt_list[i].decision_function(data[data['y']==j].iloc[:,0:f])[:,k]).reshape((data[data['y']==j].shape[0],1)))
                        k+=1

                elif n_classes == 2:
                    
                    data.loc[ data['y'] == cls[1],'loss'+str(i+1)] = np.maximum(np.zeros((data[data['y']==cls[1]].shape[0],1)) , 
                                        (1 - opt_list[i].decision_function(data[data['y']==cls[1]].iloc[:,0:f])).reshape((data[data['y']==cls[1]].shape[0],1)))
                    

                    data.loc[ data['y'] == cls[0],'loss'+str(i+1)] = np.maximum(np.zeros((data[data['y']==cls[0]].shape[0],1)) , 
                                        (1 + opt_list[i].decision_function(data[data['y']==cls[0]].iloc[:,0:f])).reshape((data[data['y']==cls[0]].shape[0],1)))

                else:
                    # this gives error - one class is not a classification problem so svm throws error 
                    data['loss'+str(i+1)] = 1e5
                    data.loc[data['model'] == i+1, 'loss'+str(i+1)] = 0


                total_loss = total_loss + sum(data.loc[data['model'] == i+1 , 'loss'+str(i+1)] )

        total_loss = total_loss/data.shape[0] 

        return data, total_loss, score, opt_list