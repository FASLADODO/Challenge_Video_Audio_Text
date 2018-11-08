from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
import pandas as pd

"""
Cette fonction permet de comparer les différents clusters

"""

def best_k(X,range_min = 20,verbose=True): 

    if range_min< 2:
        raise ValueError('range_min is less than 2')
    score = []
    for i,k in enumerate(range(2,range_min)):

        model = KMeans(n_clusters=k,random_state=42,n_init=30)
        score.append(silhouette_score(X,model.fit_predict(X)))
        if verbose: 
            print(f'Le score pour k={k} est : {score[i]:.2f}')

    return range(2,range_min)[score.index(max(score))]