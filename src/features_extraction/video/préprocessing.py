# -*- coding: utf-8 -*-
"""Préprocessing.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1juE01m2PxsZzccBfY4fTZ4-1yyTfwS31

####Imports
"""

# %matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.image import imread, imsave
import cv2
from tqdm import tqdm
import os
import glob
import shutil
from sklearn import svm, grid_search, datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from random import shuffle

! pip install imageio
! pip install opencv-python

"""data  
        text  
        audio  
        video

##Préprocessing

- fonctions propres et commentées :
    - video_to_frames_onepersec : extraction des frames d'une vidéo (une par seconde)
    - video_to_frames_onepercut : extraction des frames d'une vidéo (une par plan)
    
    
###Calcul Descripteurs

- formattage en vecteur pour préparer la classification

- couleur : ...
- contour : ...


- librairie YOLO ?
- etc

### Mémo

- `cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)` : convertire image en niveaux de gris
- `imread` pour lire une image avec `matplotlib`
- https://bcastell.com/posts/scene-detection-tutorial-part-1/
"""

from google.colab import files, drive, auth
import os

drive.mount("/content/gdrive", force_remount=False)

PATH = "/content/gdrive/My Drive/AED/"

if os.path.isfile(f"{PATH}data.zip"): #and not os.path.isdir("data/"):
    print("\nUnziping the data...")
    !unzip -q gdrive/My\ Drive/AED/data.zip
    print("Done.")
else:
    print("\nData directory already ready.")


PATH = "/content/gdrive/My Drive/AED/"
if os.path.isfile(f"{PATH}image_3.zip") and not os.path.isdir("data/image_3"):
    print("\nUnziping the data...")
    !mkdir -p data/image_3/
    !unzip -q gdrive/My\ Drive/AED/image_3 -d data/image_3/
    print("Done.")
else:
    print("\nData directory already ready.")
    
if os.path.isfile(f"{PATH}image_sec.zip") and not os.path.isdir("data/image_sec"):
    print("\nUnziping the data...")
    !mkdir -p data/image_sec/
    !unzip -q gdrive/My\ Drive/AED/image_sec -d data/image_sec/
    print("Done.")
else:
    print("\nData directory already ready.")
    
if os.path.isfile(f"{PATH}image_200.zip"): #and not os.path.isdir("data/image_sec"):
    print("\nUnziping the data...")
    !mkdir -p data/image_200/
    !unzip -q gdrive/My\ Drive/AED/image_200 -d data/image_200/
    print("Done.")
else:
    print("\nData directory already ready.")

"""###Plot images"""

def show_images(*args, col=3):
    """
        Plot image(s)
        
        Take as param: list, str (for a folder's path) or np.ndarray.
    """
    for arg in args:
        if isinstance(arg, list):
            images = arg
            rows = len(images) // col + 1
            fig = plt.figure(figsize=(col*8, rows*6))
            for i, image in enumerate(images):
                try:
                    fig.add_subplot(rows, col, i+1)
                    plt.imshow(image)
                    plt.grid(False)
                    plt.axis('off')
                    plt.title(i)
                except:
                    pass
        elif isinstance(arg, str):
            folder = arg
            paths = sorted(glob.glob(f"{folder}/*.jpg"))
            if not paths:
                print(f"The folder '{folder}' does not contain any JPG image.")
            else:
                rows = len(paths) // col + 1
                fig = plt.figure(figsize=(col*8, rows*6))
                for i, path in enumerate(paths):
                    try:
                        fig.add_subplot(rows, col, i+1)
                        plt.imshow(imread(path))
                        plt.grid(False)
                        plt.axis('off')
                        plt.title(i)
                    except:
                        pass
        elif isinstance(arg, np.ndarray):
            image = arg
            plt.figure(figsize=(8, 6))
            plt.imshow(image)
            plt.grid(False)
            plt.axis('off')
        else:
            print("Invalid type of argument (must be 'list', 'str' or 'np.ndarray')")
    plt.show()

"""###Transform a video into frames (one per second,  one per cut)"""

def video_to_frames(videopath):
    frames = []
    vidcap = cv2.VideoCapture(videopath)
    framerate = int(vidcap.get(5))
    name = os.path.splitext(os.path.basename(videopath))[0]
    success, frame = vidcap.read()
    frame_number = 0
    while success:
        if frame_number % framerate == 0:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # convertion to RGB
            frames.append(frame)
        success, frame = vidcap.read() 
        frame_number += 1
    return frames, framerate, name


def seq_to_3_frames(images):
    duration = len(images)
    tiers_values = [duration//4, duration//2, 3*duration//4]
    frames = [images[tiers] for tiers in tiers_values]
    return frames, duration


# f, framerate, n = video_to_frames("data/video/SEQ_003_VIDEO.mp4")
# f, d = seq_to_3_frames(f)

# show_images(f, col=3)

"""###Parse all videos to extract frames"""

def build_image_folder(start=None, end=None):
    for videopath in ProgressBar((sorted(glob.glob("data/video/*.mp4")))[start:end]):
        frames_per_sec, framerate, name = video_to_frames(videopath)
        folder = f"data/image_sec/{name}"
        os.makedirs(folder, exist_ok=True)
        for i, frame in enumerate(frames_per_sec):
            imsave(f"{folder}/frame_{i:03}.jpg", frame)
            
        frames_3, duration = seq_to_3_frames(frames_per_sec)
        folder = f"data/image_3/{name}"
        os.makedirs(folder, exist_ok=True)
        for i, frame in enumerate(frames_3):
            imsave(f"{folder}/frame_{i}.jpg", frame)

    shutil.make_archive("image_3", 'zip', "data/image_3")
    shutil.make_archive("image_sec", 'zip', "data/image_sec")
    ! mv image_sec.zip "/content/gdrive/My Drive/AED/"
    ! mv image_3.zip "/content/gdrive/My Drive/AED/"
    
# build_image_folder()
# !ls /content/gdrive/My Drive/AED/

def folder_to_list(folder):
    paths = sorted(glob.glob(f"{folder}/*.jpg"))
    if paths:
        frames = []
        for path in paths:
            frames.append(imread(path))        
    else:
        print(f"The folder '{folder}' does not contain any JPG image.")
    return frames

def folder_to_list_grey(folder):
    paths = sorted(glob.glob(f"{folder}/*.jpg"))
    if paths:
        frames = []
        for path in paths:
            frames.append(imread(path, 0))     
    else:
        print(f"The folder '{folder}' does not contain any JPG image.")
    return frames

"""##Calcul descripteurs

###Transform Images to Colours histograms
"""

# # def quantification(img, nbits = 2):
# #     num = 0
# #     for i in range(nbits):
# #         num += 128 / (2**i)  # on determine la valeur correspondant à la quantification
# #     Rouge = np.bitwise_and(img[:,:,0], int(num))  # en fonction du nombre de bits choisits
# #     Vert = np.bitwise_and(img[:,:,1], int(num))
# #     Bleu = np.bitwise_and(img[:,:,2], int(num))
# #     Rouge = np.floor(Rouge / (2**(8-3*nbits)))
# #     Vert = np.floor(Vert / (2**(8-2*nbits)))
# #     Bleu = np.floor(Bleu / (2**(8-nbits)))
# #     return Rouge + Vert + Bleu

# def histogramme(img):
#     M = img.shape[0]
#     N = img.shape[1]
#     list_histo = []
#     val =1/(M*N)
#     for color in range(3):
#         histo = np.zeros(256)    
#         for i in range(M):
#             for j in range(N):
#                 histo[int(img[i,j, color])] += val
#         list_histo.append(histo)
#     return list_histo
        
# def dist_Manhattan(hist1, hist2):
#     return sum(np.abs(np.array(hist1) - np.array(hist2)))

"""### Récupérer 200 images par séquence"""

def seq_to_200_frames(images):
    duration = len(images)
    tiers_values = []
    for i in range(1,201):
        tiers_values.append(i*duration//201)
#         tiers_values = [duration//201, 2*duration//11, 3*duration//11, 4*duration//11, 5*duration//11, 6*duration//11,
#                     7*duration//11, 8*duration//11, 9*duration//11, 10*duration//11]
    frames = [images[tiers] for tiers in tiers_values]
    return frames, duration

def build_image_folder_200(start=None, end=None):
    for videopath in tqdm((sorted(glob.glob("data/video/*.mp4")))[start:end]):
        frames_per_sec, framerate, name = video_to_frames(videopath)
        #folder = f"data/image_sec/{name}"
        #os.makedirs(folder, exist_ok=True)
        #for i, frame in enumerate(frames_per_sec):
        #    imsave(f"{folder}/frame_{i:03}.jpg", frame)
            
        frames_200, duration = seq_to_200_frames(frames_per_sec)
        folder = f"data/image_200/{name}"
        os.makedirs(folder, exist_ok=True)
        for i, frame in enumerate(frames_200):
            imageio.imwrite(f"{folder}/frame_{i:03}.jpg", frame)

    shutil.make_archive("image_200", 'zip', "data/image_200")
    ! mv image_200.zip "/content/gdrive/My Drive/AED/"

build_image_folder_200()

import cv2
import os
#from imageio import imread
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.decomposition import PCA
REP = 'data/images_200'

"""####Videos to frame per cut"""

# def cut(distance):
#     seuil = 60000  # A affiner
#     cuts = [0]
#     for i in range(1,len(distance)):
#         if distance[i] > seuil:
#             cuts.append(i)
#     return len(cuts), cuts

#     for name in tqdm(glob.glob("data/image_200/*")[start:end]):
#         images_200 = folder_to_list(name)
#         features = []
#         for frame in images_200:
#             folder_to_list_grey(folder)                for h in histo:
#                     features.append(int(h))


# def cuts(seq):
#     distance = []
#     cuts = [0]
#     for i in range(1, len(seq) - 1):
#         r, v, b = cv2.calcHist(seq[i], [color], None, [256], [0, 256]) for color in [0, 1, 2]
#         h1 = (r+v+b)/3
#         r, v, b = histogramme(seq[i+1])
#         h2 = (r+v+b)/3
#         distance.append(dist_Manhattan(h1, h2))
#         if dist_Manhattan(h1, h2) > 0.3:  # seuil à affiner
#             cuts.append(i)    
#     return cuts




# folder_to_list_grey(folder)



def process_cuts(start=None, end=None):
    dic = {}
    for name in tqdm(sorted(glob.glob("data/image_sec/*"))[start:end]):
        seq = folder_to_list_grey(name)
        dist = []
        for i in range(len(seq) -1):
            hist1 = cv2.calcHist(seq[i], [0], None, [256], [0, 256])
            hist2 = cv2.calcHist(seq[i+1], [0], None, [256], [0, 256])
            dist.append(sum(np.abs(np.array(hist1) - np.array(hist2))))
#     distance = []
#     cuts = [0]
#     for i in range(1, len(seq) - 1):
#         r, v, b = cv2.calcHist(seq[i], [color], None, [256], [0, 256]) for color in [0, 1, 2]
#         h1 = (r+v+b)/3
#         r, v, b = histogramme(seq[i+1])
#         h2 = (r+v+b)/3
#         distance.append(dist_Manhattan(h1, h2))
#         if dist_Manhattan(h1, h2) > 0.3:  # seuil à affiner
#             cuts.append(i)    

    return pd.DataFrame.from_dict(dic, orient="index")

process_cuts(end=2)

"""216 : plan séquence

###Calculate the momentum (amount of movement)

####Naïve version
"""

import numpy as np
import cv2
import glob

# pas utilisé

# fichiers annotés violents
violence = {'SEQ_001_VIDEO','SEQ_011_VIDEO','SEQ_013_VIDEO','SEQ_016_VIDEO','SEQ_018_VIDEO','SEQ_025_VIDEO','SEQ_026_VIDEO','SEQ_034_VIDEO','SEQ_036_VIDEO','SEQ_037_VIDEO','SEQ_045_VIDEO','SEQ_047_VIDEO','SEQ_061_VIDEO','SEQ_062_VIDEO','SEQ_097_VIDEO','SEQ_103_VIDEO','SEQ_104_VIDEO','SEQ_106_VIDEO','SEQ_110_VIDEO','SEQ_112_VIDEO','SEQ_114_VIDEO','SEQ_115_VIDEO','SEQ_116_VIDEO','SEQ_130_VIDEO','SEQ_131_VIDEO','SEQ_141_VIDEO','SEQ_162_VIDEO','SEQ_169_VIDEO','SEQ_181_VIDEO','SEQ_193_VIDEO','SEQ_203_VIDEO','SEQ_219_VIDEO','SEQ_221_VIDEO','SEQ_223_VIDEO','SEQ_224_VIDEO','SEQ_225_VIDEO','SEQ_227_VIDEO','SEQ_233_VIDEO','SEQ_234_VIDEO','SEQ_235_VIDEO','SEQ_238_VIDEO','SEQ_241_VIDEO','SEQ_264_VIDEO','SEQ_265_VIDEO','SEQ_270_VIDEO','SEQ_271_VIDEO','SEQ_275_VIDEO','SEQ_277_VIDEO','SEQ_278_VIDEO','SEQ_279_VIDEO','SEQ_281_VIDEO','SEQ_289_VIDEO','SEQ_294_VIDEO','SEQ_295_VIDEO','SEQ_298_VIDEO','SEQ_301_VIDEO','SEQ_302_VIDEO','SEQ_307_VIDEO'}

for videopath in sorted(glob.glob("data/video/*.mp4")):

    name = os.path.splitext(os.path.basename(videopath))[0]
    
    if (name is in violence):
        # 002 => 55
        # 003 => 5
        # 009 => 16
        # 012 => 12
        # 047 => 23
        # 004 => 14
        # 006 => 6
        # 013 => 17
        # 016 => 1,6 à l'audio de repérer celle ci

        cap = cv2.VideoCapture(videopath)

        # params for ShiTomasi corner detection
        feature_params = dict( maxCorners = 100,
                               qualityLevel = 0.3,
                               minDistance = 7,
                               blockSize = 7 )

        # Parameters for lucas kanade optical flow
        lk_params = dict( winSize  = (15,15),
                          maxLevel = 2,
                          criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

        # Create some random colors
        color = np.random.randint(0,255,(100,3))

        # Take first frame and find corners in it
        ret, old_frame = cap.read()
        old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
        p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

        # Create a mask image for drawing purposes
        mask = np.zeros_like(old_frame)

        distance = 0

        while(ret):
            ret, frame = cap.read()
            if (frame is not None):
                frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                # calculate optical flow
                p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

                # Select good points

                if (p1 is None):
                    good_new = p0[st==1]
                else:
                    good_new = p1[st==1]
                good_old = p0[st==1]

                # draw the tracks
                for i,(new,old) in enumerate(zip(good_new,good_old)):
                    a,b = new.ravel()
                    c,d = old.ravel()
                    mask = cv2.line(mask, (a,b),(c,d), color[i].tolist(), 2)
                    distance += np.sqrt((a - c)**2 + (b - d)**2)
                frame = cv2.circle(frame,(a,b),5,color[i].tolist(),-1)
                img = cv2.add(frame,mask)
                #print(distance)

            #     cv2.imshow('frame',img)
            #     k = cv2.waitKey(30) & 0xff
            #     if k == 27:
            #         break

                # Now update the previous frame and previous points
                old_gray = frame_gray.copy()
                p0 = good_new.reshape(-1,1,2)


        number_of_frames = int(cap.get(7))
        mean_movement = distance/number_of_frames
#         if (mean_movement > 45):
        print(f"Name: {name}   Movement: {mean_movement}")


    cv2.destroyAllWindows()
    cap.release()

"""####Smart version"""

import cv2
import numpy as np
from tqdm import tqdm


# def optical_flow_smart(videopath):
#     cam = cv2.VideoCapture(videopath)
#     ret, img = cam.read()
#     prevgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     res = []
#     while ret:
#         gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#         flow = cv2.calcOpticalFlowFarneback(prevgray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
#         prevgray = gray
#         res.append(np.sum(flow))
#         ret, img = cam.read()
#     return np.sum(res)

def quant(img, flow, step=16):
    h, w = img.shape[:2]
    y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2,-1).astype(int)
    fx, fy = flow[y,x].T
    lines = np.vstack([x, y, x+fx, y+fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)
    tot = []
    for (x1, y1), (x2, y2) in lines:
        tot.append(np.sqrt((x2 - x1)**2 + (y2 - y1)**2))
    return np.sum(np.abs(np.abs(tot) - np.mean(np.abs(tot))))


def optical_flow_smart(videopath):
    frames = folder_to_list(videopath)
    res = []
    paths = sorted(glob.glob(f"{videopath}/*.jpg"))
    prevgray = cv2.imread(paths[0], 0)
    for path in paths[1:]:
        distance = 0
        gray = cv2.imread(path, 0)
        flow = cv2.calcOpticalFlowFarneback(prevgray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        q = quant(gray, flow)
        res.append(q)
    return res

def process_momentum(start=None, end=None):
    dic = {}
    for name in tqdm(sorted(glob.glob("data/image_200/*"))[start:end]):
#         images_200 = folder_to_list(name)
        features = []
        for path in glob.glob(name):
            res = optical_flow_smart(path)
            plt.plot(res)
            plt.title(path)
            plt.show()
            features = []
            for i in range(len(res)):
                features.append(res[i])
        dic[name[15:]] = features
    return pd.DataFrame.from_dict(dic, orient="index")



# prompt histograms
# for path in tqdm((sorted(glob.glob("data/image_sec/*")))):
#     plot data
#     plt.plot(optical_flow_smart(path))
#     plt.title(path)
#     plt.show()
    
df_momentum = process_momentum()
df_momentum.to_csv("df_momentum.csv", sep="§")
df_momentum

"""##Classification

###Intérieur/Extérieur avec les cuts

####KNN
"""

def KNN_plus_gridsearch(X, y, n_neighbors_knn, n_neighbors_grid):

    #KNN
    # split and shuffle X and y
    TEST_SIZE = np.int(np.floor(len(y)*0.7))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE)

    # train
    n_neighbors = n_neighbors_knn
    clf = KNeighborsClassifier(n_neighbors)
    #TODO: fit le modèle 1 seule fois
    clf.fit(X_train, y_train)

    # predict
    #TODO: prédire toujours sur les mêmes vidéos pour pouvoir comparer l'accuracy des prédictions avec différents paramètrages
    y_pred = clf.predict(X_test)
    
    # transform into dataframe
    y_pred = pd.DataFrame(y_pred)
    y_pred.index = y_test.index
    df = pd.concat([y_pred, y_test], axis=1)
    df.columns = ['pred', 'test']

    # confusion matrix
    confusion = confusion_matrix(y_test, y_pred)
    confusion = pd.DataFrame(data=confusion, index=["A", "B"], columns=["X", "Y"])

    #accuracy
    acc = 0
    for i in df.index:
        if (df['test'][i]==df['pred'][i]):
            acc +=1
    acc = acc/len(df)
    
    
    # GRID SEARCH
    # parameters
    myList = list(range(1,n_neighbors_grid))
    n = filter(lambda x: x % 2 != 0, myList)
    parameters = {'n_neighbors':n}

    # learn with grid search
    model = grid_search.GridSearchCV(clf, parameters)
    #TODO: fit le modèle 1 seule fois
    model.fit(X_train, y_train)
    
    # le best k trouvé est à priori dans les paramètres
    params = model.cv_results_.keys()

    
    return df, confusion, acc, params

# get in/out classification
y_inout = pd.read_csv(PATH+"Annotations.csv")[["Exterieur"]]

df, confusion, acc, params= KNN_plus_gridsearch(X_cuts, y_inout, 2, 50)
print (df)
print (confusion)
print(f"accuracy: {acc}")

"""####Grid search (to find the best number of nearest neighbors)

####Prédictions

Paramètres modifiable:
- seuil de détection des cuts (seuil)
- nombre de plus proches voisins (voisins)

Résultat :
- accuracy (acc)

### clasif intérieur/extérieur avec la luminosité ?
https://stackoverflow.com/questions/14243472/estimate-brightness-of-an-image-opencv
"""

def process_histo(start=None, end=None):
    dic = {}
    for name in tqdm(glob.glob("data/image_200/*")[start:end]):
        images_200 = folder_to_list(name)
        features = []
        for frame in images_200:
            for histo in [cv2.calcHist(frame, [color], None, [256], [0, 256]) for color in [0, 1, 2]]:
                for h in histo:
                    features.append(int(h))
        dic[name[15:]] = features
    return pd.DataFrame.from_dict(dic, orient="index")

df_histo = process_histo()
df_histo

df_histo.to_csv("df_histo.csv", sep="§")

"""###Violent/Non-violent avec le momentum"""

# get violent/non-violent classification
y_violent = pd.read_csv(PATH+"Annotations.csv")[["Violent"]]

df, confusion, acc, params = KNN_plus_gridsearch(X_momentum, y_violent, 2, 50)
print (df)
print (confusion)
print(f"accuracy: {acc}")

"""##Restitution"""

import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, plot, iplot
import numpy as np

# merci Adil
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

df_histo.to_csv("df_histo.csv", sep = "§")
df_momentum.to_csv("df_momentum.csv",  sep = "§")

