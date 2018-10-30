import xml.etree.ElementTree as ET
import pandas as pd
import os
from tqdm import tqdm

def create_time_csv(path, path_csv_out):

    files = list(sorted([file for file in os.listdir(path) if file.split('.')[-1] == 'xml']))
    # print(files)
    time_csv = pd.DataFrame()

    for filename in tqdm(files):
        file = path + filename

        tree = ET.parse(file)
        root = tree.getroot()

        dict_time = {}

        for s in root.iter('s'):
            time = s.iter('time')
            dict_time[s.attrib['id']] = {time.attrib['id'][-1]:time.attrib['value'] for time in s.iter('time')}
            df = pd.DataFrame.from_dict(dict_time, orient='index')
            df = df.reset_index()
            df.index = [root.attrib['id']] * df.shape[0]
        
        time_csv = pd.concat([time_csv, df], axis=0)

    time_csv = time_csv.rename(columns={'index':'s_index', 'S':'start', 'E':'end'})
    time_csv.to_csv(path_csv_out, sep='§')
    return time_csv


if __name__ == '__main__':
    path = './data/text/'
    create_time_csv(path, 'features/text/time_csv.csv')











# list_id = set([id_.attrib['id'][:-1] for id_ in root.iter('time')])
# #print(list_id)
# columns = ['start', 'end', 'text']
# time_csv = pd.DataFrame(index=list_id, columns=columns)

# for time, text in zip(root.iter('time'), root.iter('w')):
#     id_ = time.attrib['id'][:-1]

#     print(text.text)
#     print()
#     if time.attrib['id'][-1] == 'S':
#         time_csv.loc[id_, ['start', 'text']] = [time.attrib['value'], text.text]
#     else:
#         time_csv.loc[id_, 'end'] = time.attrib['value']

# print(time_csv.head(6))
