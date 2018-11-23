from plot_cluster import plot_cluster
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import pandas as pd
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize

# def best_k(X, range_min=20):

#     if range_min < 1:
#         raise ValueError('range_min is less than 1')
#     score = []
#     for i, k in enumerate(range(2, range_min)):

#         model = KMeans(n_clusters=k, random_state=42, n_init=30)
#         score.append(silhouette_score(X, model.fit_predict(X)))
        
#         print(f'Le score pour k={k} est : {score[i]:.2f}')

#     return range(2, range_min)[score.index(max(score))]

def clustering(df, file_out, reduce_pca=None, nb_cluster=2):

    if reduce_pca is not None:
        pca = PCA(n_components=reduce_pca)
        df = pd.DataFrame(pca.fit_transform(df), index=df.index)
        print(sum(pca.explained_variance_ratio_))

    kmeans = KMeans(n_clusters=nb_cluster, random_state=42, n_init=30)
    labels = kmeans.fit_predict(df)

    tsne = TSNE(n_components=2)
    transf = pd.DataFrame(tsne.fit_transform(df), index=df.index)

    plot_cluster(transf.values, transf.index, labels, file_out)

    return labels

if __name__ == '__main__':
    path_csv = 'features/audio/csv/'

    lium = pd.read_csv(path_csv + 'locutors.csv', sep='§', index_col='Sequence', engine='python')

    df_nrj_f0 = pd.read_csv(path_csv + 'audio_nrj_f0.csv', sep='§', index_col='Sequence', engine='python')

    nrj = df_nrj_f0[[col for col in df_nrj_f0.columns if 'nrj' in col]]
    f0 = df_nrj_f0[[col for col in df_nrj_f0.columns if 'f0' in col]]

    df = pd.merge(lium, df_nrj_f0, left_index=True, right_index=True)
    df = df_nrj_f0

    clustering(df, 'result/audio/cluster_all.html', nb_cluster=4)
    print(nrj.head(2))
    print(f0.head(2))