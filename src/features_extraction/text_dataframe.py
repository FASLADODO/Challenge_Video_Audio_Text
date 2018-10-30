import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET
import os


class GetText:
    def __init__(self, list_path_in, path_out: str):
        """
        Arguments:
            list_path_in {List[str]} -- List of file you want to read
            path_out {str} -- path of csv file 
        """

        self.list_path_in = list_path_in
        self.path_out = path_out

    def get_text(self):
        for path in self.list_path_in:
            tree = ET.parse(path)
            root = tree.getroot()
            dict_replica = {}
            for s in root.iter('s'):
                dict_replica.update(
                    {s.attrib['id']: ''.join([w.text for w in s.iter('w')])})
            yield path, {root.attrib['id']: dict_replica}

    def generate_dataframe(self):
        dataframe = pd.DataFrame(columns=['Sequence', 's_id', 'Text'])
        for document,dict_sequence in self.get_text():
                data = pd.DataFrame.from_dict(dict_sequence)
                name_col = data.columns[0]
                data['Sequence'] = name_col
                data = data.reset_index()
                data = data.rename(columns={name_col: 'Text', 'index': 's_id'})
                dataframe = pd.concat([dataframe, data], axis='rows')
        
        dataframe.to_csv(self.path_out, index=False,sep='§')


if __name__ == '__main__':
    list_path = [f'/home/mickael/Documents/challenge_son_vidéo_texte/data/text/{f}'for f in os.listdir('/home/mickael/Documents/challenge_son_vidéo_texte/data/text/')]
    list_path = [x for x in list_path if x[-4:]=='.xml']
    test = GetText(list_path, '/home/mickael/Documents/challenge_son_vidéo_texte/features/text/sequence_text.csv')
test.generate_dataframe()