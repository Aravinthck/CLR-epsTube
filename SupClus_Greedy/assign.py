
from abc import ABC, abstractmethod
from  SupClus_Greedy.utils import *
import numpy as np
from scipy.spatial import distance

class Assignment(ABC):


    @abstractmethod
    def assign_cluster(self, clus_data,K,f):
        """
        :param clus_data:    Data with previous cluster assignment
        :param K:            Number of clusters   

        """

        raise NotImplementedError


class ArbitraryAssign(Assignment):


    def assign_cluster(self, clus_data, K, f):

        new_model = []
        for _, row in clus_data.iterrows():
            new_model.append(int(row.iloc[-K:].idxmin()[-1]))
        loss_best_pre = 0

        for i in range(K):
            loss_best_pre = loss_best_pre + sum( clus_data[clus_data['model'] == i+1]['loss'+str(i+1)] )
        
        loss_best_pre = loss_best_pre/clus_data.shape[0]


        return new_model, loss_best_pre


class ClosestCentroid(ArbitraryAssign):


    def assign_cluster(self,clus_data,K,f):
        
        clus_data = clus_data.copy()
        new_model = []
        clus_data['model'], loss_best_pre = super().assign_cluster(clus_data,K,f)

        centroid = centroids(clus_data, K, f, with_y = False)
        
        centroid_dist = np.zeros((len(clus_data),K))

        for i in range(K):
            centroid_dist[:,i] = np.sum((clus_data.iloc[:,0:f] - centroid[i])**2,axis = 1)
        
        new_model = np.argmin(centroid_dist, axis=1)+1

        return new_model, loss_best_pre

class BoundingBox(ArbitraryAssign):


    def assign_cluster(self,clus_data,K,f):

        clus_data = clus_data.copy()
        new_model = []
        clus_data['model'], loss_best_pre = super().assign_cluster(clus_data,K,f)


        # centroid = medians(clus_data, K, f, with_y = False)
        centroid = centroids(clus_data, K, f, with_y = False)

        # print(centroid)
        centroid_dist = np.zeros((len(clus_data),K))

        for i in range(K):
            for j in range(len(clus_data)):
                centroid_dist[j,i] = distance.chebyshev(clus_data.iloc[j,0:f], centroid[i])
                # centroid_dist[j,i] = distance.cityblock(clus_data.iloc[j,0:f], centroid[i])

        
        
        new_model = np.argmin(centroid_dist, axis=1)+1
 
        # for i in range(len(new_model)):
        #     print(centroid_dist[i,:], new_model[i])

        return new_model, loss_best_pre
