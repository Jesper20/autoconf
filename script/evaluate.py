import time
import numpy as np
import pandas as pd
import math
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.cluster import DBSCAN
from sklearn.cluster import MiniBatchKMeans
from sklearn.cluster import MeanShift
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import OPTICS
from sklearn.cluster import KMeans


def getData(df,h,overlap):
    hasLabel = False
    if 'label' in df.columns:
        hasLabel = True

    result = []
    step = h - overlap 
    if step == 0:
        step = 1
    data = { } 
    
    for stat in stats:
        for index, col in enumerate(cols):
            feature = stat + "_" + col
            # temp.append([])
            data[feature] = []
    if hasLabel:
        data['label'] = []

    for i in range(0, len(df)-h, step):
        label = 0
        # if i % 1000 == 0:
        #     print(i)

        j = i + h
        if j > len(df):
            j = len(df)
         
        w_df = df.iloc[i:j,:]
        stats_data = w_df.describe()
       
        if hasLabel and max(w_df['label']) == 1: # majority of labels in this time window == 1
            label = 1
            
        for stat in stats:
            for index, col in enumerate(cols):
                feature = stat + "_" + col
                if stat == 'variance':
                    val = stats_data[col]["std"]
                    val = val ** 2 # convert std to variance
                elif stat == 'freq':
                    f = abs(np.fft.fft(w_df[col]))
                    val = max(f)
                else:
                    val = stats_data[col][stat]
                
                data[feature].append(val)

        if hasLabel:
            data['label'].append(label)

    result = pd.DataFrame(data)
    return result


def buildmodel(space):
    if space['algo']['model'] == 'kmeans':
        model = KMeans(n_clusters= int(space['algo']['n_clusters']), n_init=10)
    elif space['algo']['model'] == 'minibatchkmeans':
        model = MiniBatchKMeans(n_clusters= int(space['algo']['minik_n_clusters']), max_iter=int(space['algo']['max_iter']), batch_size=int(space['algo']['batch_size']))
    elif space['algo']['model'] == 'meanshift':
        b = False if space['algo']['bin_seeding'] == 0 else True
        c = False if space['algo']['cluster_all'] == 0 else True
        model = MeanShift(bin_seeding=b, cluster_all=c)
    elif space['algo']['model'] == 'dbscan':
        model = DBSCAN(eps=round(space['eps'],2), min_samples=int(space['min_samples']), metric=space['metric'], algorithm=space['algorithm'])
    elif space['algo']['model'] == 'optics':
        model = OPTICS(max_eps=round(space['algo']['max_eps'],2), min_samples=int(space['algo']['optics_min_samples']))
    elif space['algo']['model'] == 'affinitypropagation':
        model = AffinityPropagation(damping=round(space['algo']['damping'],1), convergence_iter= int(space['algo']['convergence_iter']))
    
    return model


def predict(model_name, model, data, sample):
    X_copy = data.copy()
    X_copy['label'] = model.labels_
    threshold = 1
    label = 0

    if model_name == "kmeans" or model_name == "minibatchkmeans":
        components = model.cluster_centers_
        dists = np.sqrt(np.sum((components - sample)**2, axis=1))
        i = np.argmin(dists)
        cluster_center = components[i]
        cluster_members = X_copy[X_copy['label']==i].copy() 
        cluster_members = cluster_members.drop('label', axis=1)

        dists_cluster_members = np.sqrt(np.sum((cluster_members - cluster_center)**2))
        max_member = np.argmax(dists_cluster_members)
        if dists[i] < dists_cluster_members[max_member]:
            label = 0  
        elif dists[i] > dists_cluster_members[max_member] * threshold:
            label = 1
    else:
        components = model.components_
        dists = np.sqrt(np.sum((components - sample)**2, axis=1))
        i = np.argmin(dists)
        if dists[i] < model.eps:
            label = 0  # use binary (normal=0, anomaly=1)
        elif dists[i] > model.eps * threshold:
            label = 1
    
    return label

def helperEvaluateBest(conf):
    h = int(conf.loc[['h']]['value'])
    overlap = round(float(conf.loc[['overlap']]['value']),2)
    overlap = math.ceil(overlap * h)

    space = {
        'algo': {},
        'h' : h,
        'overlap' : overlap
    }
    algo_idx = int(conf.loc[['algo']]['value'])
    algo = models[algo_idx]

    if algo == 'kmeans':
        space['algo']['model'] = 'kmeans'
        space['algo']['n_clusters'] = int(conf.loc[['n_clusters']]['value'])
    elif algo == 'minibatchkmeans':
        space['algo']['model'] = 'minibatchkmeans'
        space['algo']['minik_n_clusters'] = int(conf.loc[['minik_n_clusters']]['value'])
        space['algo']['max_iter'] = int(conf.loc[['max_iter']]['value'])
        space['algo']['batch_size'] = int(conf.loc[['batch_size']]['value'])
    elif algo == 'meanshift':
        space['algo']['model'] = 'meanshift'
        space['algo']['bin_seeding'] = int(conf.loc[['bin_seeding']]['value'])
        space['algo']['cluster_all'] = int(conf.loc[['cluster_all']]['value'])
    elif algo == 'dbscan':
        space['algo']['model'] = 'dbscan'
        space['algo']['eps'] = float(conf.loc[['eps']]['value'])
        space['algo']['min_samples'] = int(conf.loc[['min_samples']]['value'])
    elif algo == 'optics':
        space['algo']['model'] = 'optics'
        space['algo']['max_eps'] = float(conf.loc[['max_eps']]['value'])
        space['algo']['optics_min_samples'] = int(conf.loc[['optics_min_samples']]['value'])
    elif algo == 'affinitypropagation':
        space['algo']['model'] = 'affinitypropagation'
        space['algo']['damping'] = float(conf.loc[['damping']]['value'])
        space['algo']['convergence_iter'] = int(conf.loc[['convergence_iter']]['value'])

    return space

if __name__ == '__main__':
    start_time = time.time()
    ################### Define Parameters ###################
    TC = 1
    showPlot = False
    threshold = 1
    conf_file = f"output/best_TC{TC}.csv"
    train_file = f'dataset/TC{TC}/train.csv'  
    test_file = f'dataset/TC{TC}/test.csv'  
    normal_file = f"output/TC{TC}_normal.csv"
    anomaly_file = f"output/TC{TC}_anomaly.csv"
    #########################################################
    # define certain params
    models = ['kmeans', 'minibatchkmeans','dbscan']
    metrics =  ['cityblock', 'euclidean', 'l1', 'l2', 'manhattan']
    algorithms = ['auto', 'ball_tree', 'kd_tree', 'brute'] 
    stats = ["mean", "variance", "min", "max"]
   
    # Create clusters with Best Model on train (reference) data
    df = pd.read_csv(train_file, header=0, index_col=None)
    cols = df.columns
    conf = pd.read_csv(conf_file, header=0, index_col=0)
    space = helperEvaluateBest(conf)
    # print(space)
    model_name = space['algo']['model']
    print(f"Evaluating Best Model {model_name}...")
    # data X from df
    df = pd.read_csv(train_file, header=0, index_col=None)
    df = df[cols]
    X = getData(df,space['h'],space['overlap'])
    
        
    # model
    print(f"Performing clustering...")
    best_scaler = preprocessing.StandardScaler() 
    best_model = buildmodel(space)
    X_scaled = best_scaler.fit_transform(X)
    best_model.fit(X_scaled)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
  
    # Test Data
    test_df = pd.read_csv(test_file, header=0, index_col=None)
    test_df = test_df[cols + ['label']] 
    test_X = getData(test_df,space['h'],space['overlap']) 

    # save the training and test files containing statistical features so that they can be used in compare.py
    X.to_csv(normal_file, index=False)
    test_X.to_csv(anomaly_file, index=False) 

    y_test = test_X['label']
    X_test = test_X.drop('label', axis=1)

    # evaluate recall, precision, f1
    y_pred = [] 
    n_test_samples = len(X_test)
    for i in range(n_test_samples):
        x = X_test.iloc[i:i+1,:].copy()
        x_scaled = best_scaler.transform(x) 
        
        label = predict(model_name, best_model, X_scaled, x_scaled)
        y_pred.append(label)
       
    y_true = np.array(y_test)
    y_pred = np.array(y_pred)
    precision = precision_score(y_true,y_pred,  zero_division=1)
    recall = recall_score(y_true,y_pred,  zero_division=1)
    fmeasure = f1_score(y_true, y_pred,  zero_division=1)

    timeTaken = f'{(time.time() - start_time)/60:.2f}'
    print(f"F1: {fmeasure}. Recall: {recall}. Precision: {precision}")
    
