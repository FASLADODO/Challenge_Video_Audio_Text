import os
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt

def create_dict_LIUM(files):
    '''
    Parsing des fichiers .seg créés par l'outil de segmentation de LIUM 
    
    Arguments:
        files {list} -- Liste contenant les noms (chemins) des fichiers 
                        générés par l'executable 
    
    Returns:
        [dict] -- Dictionnaire contenant les données similaires à celles 
                  contenues dans les fichiers .seg 
    '''

    dict_caract = {}

    for file in files:
        name_seq = file.split('/')[-1].split('.')[0][:7]
        dict_caract[name_seq] = {}

        with open(file, 'r') as f:
            for line in f:
                if ';; ' in line:
                    speaker = line.split(' ')[2]
                    dict_caract[name_seq][speaker] = {}

                    for metric in ['FS', 'FT', 'MS', 'MT']:
                        search_value = re.search(f"{metric} = (?P<score>[-\.\d]+) ] ", line)
                        value = search_value.group('score')

                        dict_caract[name_seq][speaker][metric] = float(value)
                else:
                    dict_caract[name_seq][speaker]['gender'] = line.split()[4]
    return dict_caract

def features_LIUM(path_segmentation):
    '''
    Fonction permettant d'extriare le nombre de locuteurs différents
    ainsi que le ratio hommes / femmes à partir de fichier .seg générés
    par l'outil de segmentation de LIUM
    
    Arguments:
        path_segmentation {str} -- dossier où sont stockés les fichiers .seg
    
    Returns:
        [pandas DataFrame] -- Un DataFrame avec en index le nom de la séquence
         et une colonne avec le nombre de locuteurs détectés et une autre avec
         le ratio hommes / femmes
    '''

    files = sorted([path_segmentation + file for file in os.listdir(path_segmentation) if file.split('.')[-1] == 'seg'])
    dict_lium = create_dict_LIUM(files)
    # print(dict_lium)
    dict_features = {}

    #print(dict_lium[files[0]]['S0']['gender'])
    for file in files:
        name_seq = file.split('/')[-1].split('.')[0][:7]
        nb_locutors = len(dict_lium[name_seq])
        nb_men = sum([1 if loc['gender'] == 'M' else 0 for loc in dict_lium[name_seq].values()])
        # nb_men = ([print(loc) for loc in dict_lium[name_seq].values()])
        # print(nb_men)
        dict_features[name_seq] = {'nb_locutors':nb_locutors,
                                   'ratio_HF':nb_men/nb_locutors}
    
    df = pd.DataFrame.from_dict(dict_features, orient='index')
    return df


if __name__ == '__main__':
    path_segmentation = 'features/audio/LIUM_segmentation/'
    
    df = features_LIUM(path_segmentation)
    df.to_csv('features/audio/csv/locutors.csv', sep='§', index_label='Sequence')
    plt.plot(df['nb_locutors'].values, df['ratio_HF'].values, 'ro')
    plt.show()