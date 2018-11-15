from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import train_test_split
import pandas as pd

path_annot = 'annotations/'

y_train = pd.read_csv(path_annot + 'y_train_exterieur.csv', sep='§', index_col='Sequence', usecols=['Exterieur', 'Sequence'])
y_test = pd.read_csv(path_annot + 'y_test_exterieur.csv', sep='§', index_col='Sequence', usecols=['Exterieur', 'Sequence'])



print(y_train)
# y_test = 