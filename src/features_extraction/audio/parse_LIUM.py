import os
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt

def create_dict_LIUM(files):

    dict_caract = {}

    for filename in files:
        file = path_segmentation + filename
        name_seq = filename.split('.')[0]
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

    files = sorted([file for file in os.listdir(path_segmentation) if file.split('.')[-1] == 'seg'])
    dict_lium = create_dict_LIUM(files)
    # print(dict_lium)
    dict_features = {}

    #print(dict_lium[files[0]]['S0']['gender'])
    for filename in files:
        name_seq = filename.split('.')[0]
        nb_locutors = len(dict_lium[name_seq])
        nb_men = sum([1 if loc['gender'] == 'M' else 0 for loc in dict_lium[name_seq].values()])
        # nb_men = ([print(loc) for loc in dict_lium[name_seq].values()])
        # print(nb_men)
        dict_features[name_seq] = {'nb_locutors':nb_locutors, 'ratio_HF':nb_men/nb_locutors}
    
    return pd.DataFrame.from_dict(dict_features, orient='index')


if __name__ == '__main__':
    path_segmentation = 'features/audio/LIUM_segmentation/'
    
    df = features_LIUM(path_segmentation)
    plt.plot(df['nb_locutors'].values, df['ratio_HF'].values, 'ro')
    plt.show()