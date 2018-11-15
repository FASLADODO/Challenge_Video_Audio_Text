from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_is_fitted
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import f1_score, accuracy_score
import pandas as pd
import numpy as np
from tqdm import tqdm
from scipy import sparse
import warnings
warnings.filterwarnings("ignore")

if __name__ == '__main__':

    data = pd.read_csv('/home/mickael/Documents/Challenge_Video_Audio_Text/features/text/tfidf_doc_punct.csv',sep = '§')

    data = pd.merge(
        left=data,
        right=pd.read_csv('/home/mickael/Documents/Challenge_Video_Audio_Text/features/text/emotion_doc.csv',sep = '§'),
        how='left',
        on='Sequence'
    )
    data = pd.merge(
        left=data,
        right=pd.read_csv('/home/mickael/Documents/Challenge_Video_Audio_Text/features/text/nmf_tfidf_5_punct.csv',sep = '§'),
        how='left',
        on='Sequence'
    )
    data = pd.merge(
        left=data,
        right=pd.read_csv('/home/mickael/Documents/Challenge_Video_Audio_Text/features/text/svd_tfidf_5_punct.csv',sep = '§'),
        how='left',
        on='Sequence'
    )
    
    labels = pd.read_csv('/home/mickael/Documents/Challenge_Video_Audio_Text/data/external/annotation.csv',usecols=['Sequence','Violent'])

    assert labels.isnull().sum().sum() == 0 
    assert len(labels) == len(data)    
    data.sort_values(by=['Sequence'],inplace =True)
    labels.sort_values(by=['Sequence'],inplace =True)

    X_train = data.drop(['Sequence'],axis = 1)
    Y_train = labels['Violent']

    model = LogisticRegression()
    # Permet de découper le jeux de données en n_splits avec un aléatoire fixé
    # http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html
    CV = LeaveOneOut()


    # Liste qui sauvegardera les valeurs de notre modèle à chaque fold
    fit_score = []
    val_score = []

    verbose = False # Si vous voulez des information lors de la periode d'entrainement mettez la valeur à True

    # enumerate un mot clef python : https://docs.python.org/3/library/functions.html#enumerate
    for i, (fit_idx,val_idx) in tqdm(enumerate(CV.split(X_train))):
        
        X_fit = X_train.iloc[fit_idx]
        Y_fit = Y_train.iloc[fit_idx]
        X_val = X_train.iloc[val_idx]
        Y_val = Y_train.iloc[val_idx]
        
        '''Mise en log des variables à predire car dans la partie evaluation du challenge sur kaggle, ils parlent
        de mettre en log la variable cible'''
        
        model.fit(X_fit,Y_fit)
        
        pred_fit = model.predict(X_fit)
        pred_val = model.predict(X_val)
        
        if verbose :
            print(f'accuracy_score fit pour le fold {i+1} : {accuracy_score(pred_fit,Y_fit):.3f}')
            print(f'accuracy_score val pour le fold fold {i+1} : {accuracy_score(pred_val,Y_val):.3f}')
        
        fit_score.append(accuracy_score(pred_fit,Y_fit))
        val_score.append(accuracy_score(pred_val,Y_val))

    fit_score = np.array(fit_score)
    val_score = np.array(val_score)

    print(f'accuracy_score score pour le fit :{np.mean(fit_score):.3f} ± {np.std(fit_score):.3f}')
print(f'accuracy_score score pour le val :{np.mean(val_score):.3f} ± {np.std(val_score):.3f}')