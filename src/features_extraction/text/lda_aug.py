
import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation,PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk import word_tokenize
from nltk.corpus import stopwords
from sklearn.cluster import KMeans
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, plot, iplot
from sklearn.metrics import silhouette_score


def best_k(X, range_min=20, verbose=True):

    if range_min < 2:
        raise ValueError('range_min is less than 2')
    score = []
    for i, k in enumerate(range(2, range_min)):

        model = KMeans(n_clusters=k, random_state=42, n_init=30)
        score.append(silhouette_score(X, model.fit_predict(X)))
        if verbose:
            print(f'Le score pour k={k} est : {score[i]:.2f}')

    return range(2, range_min)[score.index(max(score))]

def plot_cluster(coords, names, labels, name_plot):
    ''' 
    Create a scatter plot
    
    Arguments:
        coords {numpy array} -- A numpy array with N lines and 2 columns 
                                (N=number of individuals) each column 
                                correspond to a dimension
        names {list} -- list corresponding to the names of the individuals
        labels {type} -- Label of the cluster (an integer like 0 for the first cluster, 1 for the second...)
        name_plot {str} -- name of the html file of the plot
    '''

    # Create a trace
    trace = go.Scatter(
        x = coords[:, 0],
        y = coords[:, 1],
        mode = 'markers',
        text = names,
        marker = dict(
            size = 10,
            color = labels,
            line = dict(
                width = 2,
                color = 'rgb(0, 0, 0)'
            )
        )
    )

    data = [trace]

    layout = dict(title = 'Styled Scatter',
                    yaxis = dict(zeroline = False),
                    xaxis = dict(zeroline = False)
                    )

    fig = dict(data=data, layout=layout)
    plot(fig, filename=name_plot)

if __name__ == '__main__':
    stop = set(stopwords.words('french'))
    stop.update(['.', ',', '"', "'", '?', '!', ':',
                    ';', '(', ')', '[', ']', '{', '}','-'])


    data = pd.read_csv('features/text/sequence_text.csv', sep='§')

    data = data.groupby(['Sequence'])['Text'].sum()
    data = data.reset_index()

    data['Text'] = data['Text'].apply(
        lambda x: ' '.join([i for i in word_tokenize(x) if i not in stop]))

    # data = pd.concat([data,pd.read_csv('data/external/corpus_wiki_300.csv', sep='§')],axis = 0)
    external = pd.read_csv('data/external/corpus_wiki_300.csv', sep='§')
    external['Sequence'] = 'Augmentation'
    data = pd.concat([data,external], axis =0)

    del external

    tfidf = TfidfVectorizer(max_df=0.95, min_df=2)

    tfidf = pd.DataFrame(tfidf.fit_transform(data['Text']).todense())

    lda = LatentDirichletAllocation(n_components=22, max_iter=30,random_state=42,verbose=1)
    lda = pd.DataFrame(lda.fit_transform(tfidf))
    lda = lda.add_prefix(f'LDA_')
    del tfidf

    pca = PCA(n_components=2)
    lda = pd.DataFrame(pca.fit_transform(lda))
    lda = lda.add_prefix(f'LDA_')

    print(lda.head())
    assert len(lda) == len(data)
    data= data.reset_index(drop = True)
    lda = lda.reset_index(drop= True)
    lda = pd.concat([lda,data['Sequence']],axis='columns',ignore_index=True)

    lda = lda.rename(columns={0: 'LDA_0', 1:'LDA_1',2:'Sequence'})
    #print(lda)
    lda = lda.loc[lda['Sequence'] != 'Augmentation']

    assert lda.isnull().sum().sum() == 0

    cluster = KMeans(n_clusters=5, random_state=42,n_init=30)
    cluster = cluster.fit_predict(lda[['LDA_0','LDA_1']])

    best_k(lda[['LDA_0','LDA_1']])
    import matplotlib as mpl
    mpl.use('TkAgg')  # or whatever other backend that you want
    import matplotlib.pyplot as plt
    import seaborn as sns; sns.set()  # for plot styling
    import warnings
    warnings.filterwarnings('ignore')

    plt.scatter(lda['LDA_0'], lda['LDA_1'].values,c=cluster, s=50, cmap='viridis')
    plt.show()

    plot_cluster(lda[['LDA_0', 'LDA_1']].values, data['Sequence'],
                cluster,
f'result/plot_cluster__{5}_lda22.html')