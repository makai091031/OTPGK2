# -*- coding: utf-8 -*-
"""
Created on Mon Oct 31 09:31:01 2022

@author: makai
"""

import numpy as np
#import math
import ot

def Pyramid_embedding(Adjacent_mat,L,d):
    #input : Adjacent_mat is the  adjacent matrix of graphs
    #L: number of levels
    #d: dimensionality of node embeddings
    
    N=len(Adjacent_mat) # number of graphs
    
    #Compute embeddings
    print("Computing embeddings...")
    Us=[]
    for i in range(N):
        n=len(Adjacent_mat[i])
        e_vals,e_vecs = np.linalg.eig(Adjacent_mat[i])
        U=e_vecs[:,:min(n,d)]
        U=abs(U)
        Us.append(U) 
        
    #Histogram creation
    print("Creating histograms...")
    Hs=[[] for i in range(N) ]
    
    for i in range(N):
        print("Graph: ", i)
        H=[]
        for j in range(L):
        #PM works by partitioning the feature space into regions of
        #increasingly larger size and taking a weighted sum of tha matches
        #that ouc at each level
            l=2**j
            D=np.zeros((d,l))
            # T=(Us[i]*l).astype(int)
            T=np.ceil((Us[i]*l)).astype(int)
            T[T==0]=1
            
            for p in range(Us[i].shape[0]):
                for q in range(Us[i].shape[1]):
                    D[q,T[p,q]-1]=D[q,T[p,q]-1]+1
                    
            H.append(D)
            
        Hs[i]=H
        
    return Us,Hs

def Compute_wasserstein_distance(Hs,ground_distance='euclidean',sinkhorn=False, sinkhorn_lambda=1e-2):
    #Input： Hs is the Histogram of graphs
    print("Compute wasserstein distance for Histogram")
    W_distance=np.zeros((len(Hs),len(Hs)))
    L=len(Hs[0])
    WDistance=[]
    for k in range(L):
        for i in range(len(Hs)):
            for j in range(len(Hs)):
                 costs = ot.dist(Hs[i][k], Hs[j][k], metric=ground_distance)
                 if sinkhorn:
                     mat = ot.sinkhorn(np.ones(len(Hs[i][L-1]))/len(Hs[i][L-1]), 
                                  np.ones(len(Hs[j][L-1]))/len(Hs[j][L-1]), costs, sinkhorn_lambda, 
                                  numItermax=50)
                     W_distance[i,j] = np.sum(np.multiply(mat, costs))
                 else:
                     W_distance[i,j] =ot.emd2([], [], costs)
        WDistance.append(W_distance)
                
                
    return WDistance #return wasserstein distances between pairwise graphs


def Compute_OTPGK(Adjacent_mat,L,d,g):
    #choice: True-Compute_wasserstein_distance_eigenvector, False-Compute_wasserstein_distance_Histogram
    
    #label,Adjacent_mat=read_file2(filename)
    #label,Adjacent_mat,NL=read_file2(filename)
    
    Us,Hs=Pyramid_embedding(Adjacent_mat,L,d) 
    
    #Hslabel=Pyramid_match_label(Adjacent_mat,NL,4,2)
    
   
    WDistance=Compute_wasserstein_distance(Hs,ground_distance='euclidean',sinkhorn=False, sinkhorn_lambda=1e-2)

    # if choice==2:
    #         print("Compute wasserstein distance for Histogram label")
    #         W_distance=Compute_wasserstein_distance_Histogram(Hslabel,ground_distance='euclidean',sinkhorn=False, sinkhorn_lambda=1e-2)

    #g=1
    Kernel=np.zeros((len(Adjacent_mat),len(Adjacent_mat)))
    for i in range(len(WDistance)):
        K=np.exp(-g*WDistance[i])
        print("Compute Wasserstein Pyramid match Kernel,i=",i)
        # Kernel.append(K)
        Kernel=Kernel+K
    
    
    return Kernel