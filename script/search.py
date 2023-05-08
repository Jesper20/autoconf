import time
import os as os
import pickle
import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.metrics import f1_score, precision_score, recall_score
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from hyperopt import rand
from sklearn.cluster import DBSCAN
from sklearn.cluster import MiniBatchKMeans
from sklearn.cluster import MeanShift
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import OPTICS
from sklearn.cluster import KMeans


def compute_cluster_means(X, model, n_clusters):
    X2 = X.copy()
    X2['cluster'] = model.labels_

    temp = []
    for label in range(n_clusters):
        #print(label)
        points = X2[X2['cluster'] == label]
        
        points = points.drop('cluster', axis=1)
        centroid = np.mean(points, axis=0) 
        temp.append(centroid)

    centroids = pd.DataFrame(temp)
    return centroids

def sample(X, n_samples=100):
    n = round(random.random() * len(X))
    #print(n)
    #samples = data.sample(100)
    samples = X.iloc[n:n+n_samples,:].copy()  # choose a random continuous sequence of samples
    return samples

def add_gaussian_noise(X):
    mu=0.1 # shift mean by 0.1
    std = 0.5 * np.std(X) # for 50% Gaussian noise
    noise = np.random.normal(mu, std, size = X.shape)
    x_noisy = X + noise
    return x_noisy 

def getData(df,h,overlap):
    hasLabel = False
    if 'label' in df.columns:
        hasLabel = True

    result = []
    step = h - overlap # if h=30, overlap=10 => step = 20
    if step == 0:
        step = 1
    data = { } 
    
    for stat in stats:
        for index, col in enumerate(cols):
            feature = stat + "_" + col
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

# MR3: Duplication of features
# For a given source input s, we denote the output as Os. 
# For the follow-up input, if new features are added by duplicating existing features, the output Of should remain unchanged
def MR3(X):
    X2 = X.copy()
    rand_col_idx = round(random.random() * len(X2.columns))
    #print(rand_col_idx)
    exist_col = X.iloc[:,rand_col_idx].copy()
    X2["new_feature"] = exist_col
    return X2

# MR5 - Addition of Uninformative Attribute: For the follow-up input, if a new uninformative feature (i.e., a feature having the same value for all the instances) is added, the output should remain unchanged.
def MR5(X):
    X2 = X.copy()
    X2['new_feature'] = 0
    return X2

# MRB1: modifying n existing instances from various clusters whose one of the features statiscally significantly deviate from all other instances (excluding outliers)
def MRB1(X, col_idx=2, n=100, n_samples=100):
    X2 = X.copy()
    cols = X.columns
    stats = X[cols[col_idx]].describe()
    mean = stats['mean']
    std = stats["std"]
    noise = mean + 2 * std
  
    if n+n_samples > len(X2):
        X2.loc[n:n+n_samples,cols[col_idx]] =  X2.loc[n:n+n_samples,cols[col_idx]] + noise 
    else:
         X2.loc[n-n_samples:n,cols[col_idx]] =  X2.loc[n-n_samples:n,cols[col_idx]] + noise 
    return X2

#MRB2: modifying n existing instances from various clusters whose random number of the features statiscally significantly deviate from all other instances (excluding outliers)
def MRB2(X, n=100, n_samples=100):
    
    X2 = MRB1(X,0,n, n_samples)
    X2 = MRB1(X2,3,n, n_samples)
    X2 = MRB1(X2,6,n, n_samples)
    # X2 = MRB1(X2,11,n, n_samples)
   
    return X2

# similar to MRB1 and MRB2. But instead of modifying existing attributes, add a new attribute where the bad instances are added with noises
def MRB3(X, col_idx=0, n=100, n_samples=100):
    X2 = X.copy()
    # col_idx = round(random.random() * len(X2.columns))
    exist_col = X.iloc[:,col_idx].copy()
    X2["new_feature"] = exist_col
  
    #cols = X.columns
    stats = X2["new_feature"].describe()
    mean = stats['mean']
    std = stats["std"]
    noise = mean + 2 * std
   
    # choose a random continuous sequence of samples
    if n+n_samples > len(X2):
        X2.loc[n:n+n_samples,"new_feature"] =  X2.loc[n:n+n_samples,"new_feature"] + noise 
    else:
         X2.loc[n-n_samples:n,"new_feature"] =  X2.loc[n-n_samples:n,"new_feature"] + noise 
   
    return X2

# MRB4: adding n new instances which statiscally significantly deviate from all core samples
def MRB4(X, model, model_name, n_clusters, n_samples=100):
    X2 = X.copy()
    if model_name == "kmeans" or model_name == "minibatchkmeans":
        centroids = pd.DataFrame(model.cluster_centers_, columns=X.columns)
        #centroids.columns =  centroids.columns.astype(str)
        # print(centroids)
    else:
        centroids = compute_cluster_means(X, model, n_clusters)
    n = 0
    #print(len(centroids))
    while n < n_samples:
        mu=0 # shift mean by 0.1
        std = 4 * np.std(centroids) # add noise 2*std
        noise = np.random.normal(mu, std, size = centroids.shape)
        noisy_samples = centroids + noise
        noisy_samples = noisy_samples.reset_index(drop=True)
        n+=len(noisy_samples)

        X2 = pd.concat([X2,noisy_samples], axis=0)
        
    X2 = X2.reset_index(drop=True)
    return X2

# MRG4: adding 1 new instance which statiscally significantly deviate from all core samples as a white noise
def MRG4(X, model, model_name, n_clusters, n_samples=1):
    X2 = X.copy()
    if model_name == "kmeans" or model_name == "minibatchkmeans":
        centroids = pd.DataFrame(model.cluster_centers_, columns=X.columns)
        #print(centroids)
        #centroids.columns =  centroids.columns.astype(str)
    else:
        centroids = compute_cluster_means(X, model, n_clusters)
   
    mu=0 # shift mean by 0.1
    std = 4 * np.std(centroids) # add noise 2*std
    noise = np.random.normal(mu, std, size = centroids.shape)
    noisy_samples = centroids + noise
    noisy_samples = noisy_samples.reset_index(drop=True)

    a_noisy_sample = noisy_samples.iloc[0:1,:].copy()

    X2 = pd.concat([X2,a_noisy_sample], axis=0)
        
    X2 = X2.reset_index(drop=True)
    # print(f"Original data: {X.shape}. Injected data: {X2.shape}")
    return X2

def evaluate_bad_MR(mr, space, n, n_samples, data, labels):
    scaler = preprocessing.StandardScaler()
    data_scaled = scaler.fit_transform(data)
    model2 = buildmodel(space)
    model2.fit(data_scaled)
    labels2 = model2.labels_
    # # idea: 
    # 1. Get the label of bad instances and put them into the bad list
    # 2. Check if those labels are also tagged to other normal instances. If yes, remove the label from the bad list (i.e., bad instances that have those labels are grouped together with normal instances)
    # 3. Count the num of instances that have the labels from the remaining bad list
    # 4. Error = total num of manipulated samples - count

    # X2.loc[n:n+n_samples,cols[col_idx]]  # these indices were manipulated
    err=0
    l2_size = len(labels2)
    bad_labels = []
    for i in range(n, n+n_samples):
        if i >= l2_size:
            break
        label = labels2[i] # get labels of bad instances
        bad_labels.append(label)
    
    for i,label in enumerate(labels2):
        if i < n or i > n+n_samples:
            if label in bad_labels:
                bad_labels.remove(label)
            
    for i in range(n, n+n_samples):
        if i >= l2_size:
            break
        label = labels2[i] # get labels of bad instances
        if label not in bad_labels:
            err += 1

    msr = err / n_samples # return ratio (0 - 1) instead of absolute value. 
    
    return msr  # return mean squared error

def evaluate_good_MR(mr, space, n, n_samples, data, labels):
    scaler = preprocessing.StandardScaler()
    data_scaled = scaler.fit_transform(data)
    model2 = buildmodel(space)
    model2.fit(data_scaled)
    labels2 = model2.labels_
    
    y_true = np.array(labels)
    y_pred = np.array(labels2)
  
    if mr == 'Good_b4':
        y_pred = np.delete(y_pred, n)
        # print(f"len of y_pred: {len(y_pred)}. len of y_true: {len(y_true)}.")

    count_equals = np.sum(y_true == y_pred)
    err = len(labels2) - count_equals
    
    msr = err / len(labels2)
   
    return msr

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
        model = DBSCAN(eps=round(space['algo']['eps'],2), min_samples=int(space['algo']['min_samples']), metric=space['algo']['metric'], algorithm=space['algo']['algorithm'])
    elif space['algo']['model'] == 'optics':
        model = OPTICS(max_eps=round(space['algo']['max_eps'],2), min_samples=int(space['algo']['optics_min_samples']))
    elif space['algo']['model'] == 'affinitypropagation':
        model = AffinityPropagation(damping=round(space['algo']['damping'],1), convergence_iter= int(space['algo']['convergence_iter']))
    
    return model

def objective(space):
    fig_counter[0] += 1
    errors = []
  
    h = int(space['h'])
    overlap = round(space['overlap'], 2)
    overlap = math.ceil(overlap * h)
    model_name = space['algo']['model']
    
    # data
    X = getData(df,h,overlap)
    # model
    print(f"Performing clustering with {model_name}. h: {h} and overlap: {overlap}...")
    scaler = preprocessing.StandardScaler() # can search for this as well
    model = buildmodel(space)
    X_scaled = scaler.fit_transform(X)
    #fit dbscan algorithm to data
    model.fit(X_scaled)
    
    # output
    #view cluster assignments for each observation
    labels = model.labels_
    # Number of clusters in labels, ignoring noise if present.
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    clusters = {}  # { 0 : [0,1,2], 1: [...], -1: [...]}
    for i,l in enumerate(labels):
        #print(i, l)
        if l in clusters:
            clusters[l].append(i)
        else:
            clusters[l] = [i]

    # we want as fewer noise as possible
    n_noise = list(labels).count(-1) # outliers from the origianl dataset
    noise_ratio = round(n_noise / len(X), 2)
    print(f"noise ratio: {noise_ratio}")
    errors.append(noise_ratio) 
  
    try:
        sil_score = silhouette_score(X_scaled, model.labels_) # this throws error when the model produces only ONE cluster
    except:
        sil_score = 1
    print(f"silhouette_score: {sil_score}")
    errors.append(1-sil_score) # when sil_score is 1 (perfect cluster), the error is zero.
   
    if OnlySil: # use this flag for comparison with usign Only Silhouette score
        mean_err = 1-sil_score
        print(f"Loss: {1-sil_score}")

        return {'loss': mean_err,'status': STATUS_OK,'eval_time': time.time()}
    
    # set some params
    n_samples = 100
    n = round(random.random() * len(X)) # generate a random index to slice from X 
   
    # Anomaly MRs that expect different outcomes for manipulated samples
    X2 = MRB1(X, 0, n, n_samples) # modify an attribute of bad samples
    errors.append(evaluate_bad_MR('b1', space, n, n_samples, X2, labels))  
    X2 = MRB2(X, n, n_samples) # modify multiple attributes of bad samples
    errors.append(evaluate_bad_MR('b2', space, n, n_samples, X2, labels)) 
    X2 = MRB3(X, 0, n, n_samples)  # add new attribute to all samples, with bad values for bad samples
    errors.append(evaluate_bad_MR('b3', space, n, n_samples, X2, labels)) 
    X2 = MRB4(X, model, model_name, n_clusters, n_samples)  # add bad instances
    errors.append(evaluate_bad_MR('b4', space, len(X), n_samples, X2, labels)) 

    # Benign MRs that expect exact same clustering as X
    # print(f"Good MR...")
    X2 = MRB1(X, 0, n, 1) # modify an attribute of ONE sample (white noise)
    errors.append(evaluate_good_MR('Good_b1', space, n, n_samples, X2, labels)) 
    X2 = MRB3(X, 0, n, 1)  # add new attribute to all samples, with bad values for bad samples
    errors.append(evaluate_good_MR('Good_b3', space, n, n_samples, X2, labels)) 
    X2 = MRG4(X, model, model_name, n_clusters, 1)  # add ONE bad instance
    errors.append(evaluate_good_MR('Good_b4', space, len(X), n_samples, X2, labels)) 

    # average the errors
    errors = np.array(errors)
    mean_err = np.mean(errors)
    print(f"Loss: {mean_err}")

    return {
        'loss': mean_err,
        'status': STATUS_OK,
        'eval_time': time.time()
        }

if __name__ == '__main__':
    start_time = time.time()
    ################### Define Parameters ###################
    randSearch = False
    OnlySil = False # use this switch to compare with Silouette Only approach
    showPlot = False
    TC = 1
    n_trials = 10000
    n_samples = 100 # number of samples to manipulate. It could be 10% of data => n_samples = round(0.1 * len(X))
   
    path = ".."
    save_trials = os.path.join(path,f"output/trial_TC{TC}.p")
    loss_file = os.path.join(path,f"output/losses_TC{TC}.csv")  # save losses at each trial here
    conf_file = os.path.join(path, f"output/best_TC{TC}.csv")
    train_file = os.path.join(path, f'dataset/TC{TC}/train.csv')
    
    #########################################################
    # define certain params for search space
    models = ['kmeans', 'minibatchkmeans','dbscan']
    models = ['dbscan']
    metrics =  ['cityblock', 'euclidean', 'l1', 'l2', 'manhattan']
    algorithms = ['auto', 'ball_tree', 'kd_tree', 'brute'] # nearest neighbours algo
    stats = ["mean", "variance", "min", "max"]
    #########################################################
    title = 'OurApproach'
    if OnlySil:
        title = 'Sil'
    fig_counter = [0]
   
    # Train Data 
    df = pd.read_csv(train_file, header=0, index_col=None)
    cols = df.columns
   
    # define search space and appropriate distribution below
    space = {
    'h' : hp.quniform('h', 30, 1500, 1),
    'overlap' : hp.uniform('overlap', 0, 1),
    "algo": hp.choice(
        "algo",
        [
            {"model": "kmeans", 'n_clusters' : hp.quniform('n_clusters', 1, 15, 1)},
            {"model": "minibatchkmeans", 'minik_n_clusters' : hp.quniform('minik_n_clusters', 1, 15, 1), 'max_iter': hp.choice('max_iter', [50, 100, 150]), 'batch_size': hp.choice('batch_size', [256, 512, 1024, 2048])},
            {"model": "meanshift", "bin_seeding": hp.choice('bin_seeding', [0, 1]), "cluster_all": hp.choice('cluster_all', [0, 1])},
            {"model": "dbscan", "eps": hp.uniform('eps', 0.1, 5), 'min_samples' : hp.quniform('min_samples', 5, 70, 1), 'metric': hp.choice('metric', metrics),'algorithm': hp.choice('algorithm', algorithms)},
            {"model": "optics", "max_eps": hp.uniform('max_eps', 0.1, 5), 'optics_min_samples' : hp.quniform('optics_min_samples', 5, 70, 1)},
            {"model": "affinitypropagation", 'damping' : hp.uniform('damping', 0.5, 1), "convergence_iter": hp.quniform('convergence_iter', 10, 30, 1)}
        ]
    )
    }
   
    if randSearch:
        search = rand.suggest # random search
        plot_title = 'Random Search'
    else:
        search = tpe.suggest # Tree-structured Parzen Estimator (TPE)
        plot_title = 'TPE Search'
    trials = Trials()
    search_start_time = time.time()
    best = fmin(
        fn=objective,  # "Loss" function to minimize
        space=space,  # Hyperparameter space
        algo=search,
        max_evals=n_trials,  # Perform n trials,
        trials=trials # by passing in a trials object directly, we can inspect all of the return values that were calculated during the experiment
    )
   
    # saving trials
    pickle.dump(trials, open(save_trials, "wb"))

    min_loss = 999
    for n,res in enumerate(trials):
        loss = round(res['result']['loss'], 4)
        if loss < min_loss:
            min_loss = loss

    best['n_trials'] = n_trials
    best['loss'] = min_loss
    pd.DataFrame.from_dict(best, orient='index', columns=['value']).to_csv(conf_file)

