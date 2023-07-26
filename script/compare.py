# Comparing various anomaly detection algorithms
import time, sys
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

from sklearn import svm
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score, recall_score, precision_score

matplotlib.rcParams["contour.negative_linestyle"] = "solid"

def plot_prediction(X_test, y_pred, y_test, result):
    pca_num_components = 2
    reduced_data = PCA(n_components=pca_num_components).fit_transform(X_test)
    results = pd.DataFrame(reduced_data,columns=['pca1','pca2'])
    # Put the testing dataset and predictions in the same dataframe
    results['actual'] = y_test
    results['prediction'] = y_pred
    fig, (ax1, ax2)=plt.subplots(1,2, sharey=True, figsize=(10,3))
    
    fig.suptitle(result, fontsize=10, fontstyle='italic')

    colors = ('blue', 'red')
    labels = ('normal', 'anomaly')
    labels2 = (0, 1)

    for color, label, l in zip(colors, labels, labels2):
        ax1.scatter(x=results[results['actual'] == l]['pca1'], y=results[results['actual'] == l]['pca2'], color=color,label=f"{label}",alpha=0.5
        )

        ax2.scatter(x=results[results['prediction'] == l]['pca1'], y=results[results['prediction'] == l]['pca2'], color=color,label=f"{label}",alpha=0.5
        )

    ax1.set_title('Actual')
    ax2.set_title('Prediction')
    for ax in (ax1, ax2):
        ax.set_xlabel('pca1')
        ax.set_ylabel('pca2')
        ax.legend(loc="upper right")
        ax.grid()
    
    #plt.savefig(f'output/{title}_TC{TC}.png')
    plt.show()

if __name__ == '__main__':
    start_time = time.time()

    test_cases = [1, 2, 3, 4]
    
    # define outlier/anomaly detection methods to be compared.
    outliers_fraction = 0.15  # this is like a threshold -- specifies % of outliers in ref dataset
    anomaly_algorithms = [
        ("One-Class SVM", svm.OneClassSVM(nu=outliers_fraction, kernel="rbf", gamma=0.1)),
        (
            "Isolation Forest",
            IsolationForest(contamination=outliers_fraction, random_state=42),
        ),
        (
            "Local Outlier Factor",
            LocalOutlierFactor(n_neighbors=35, contamination=outliers_fraction, novelty=True),
        ),
    ]

    for i, TC in enumerate(test_cases):
        normal_file = f"output/TC{TC}_normal.csv"
        anomaly_file = f"output/TC{TC}_anomaly.csv"

        X = pd.read_csv(normal_file, header=0, index_col=None)
        test_X = pd.read_csv(anomaly_file, header=0, index_col=None)
        print(X.columns, test_X.columns)
        y_test = test_X['label']
        X_test = test_X.drop('label', axis=1)
        y = [0] * len(X) # all labeled as 0
        scaler = preprocessing.StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_test_scaled = scaler.transform(X_test) 

        for name, clf in anomaly_algorithms:
            start_time = time.time()
            clf.fit(X_scaled)
        
            # Predict the anomalies
            y_pred = clf.predict(X_test_scaled)
            # Change the anomalies' values to make it consistent with the true values
            y_pred = [1 if i==-1 else 0 for i in y_pred]
            # Check the model performance
            # print(classification_report(y_test, y_pred))
            y_true = np.array(y_test)
            recall = round(recall_score(y_true,y_pred,  average='macro'),2)
            precision = round(precision_score(y_true,y_pred,  average='macro'),2)
            fmeasure = round(f1_score(y_true, y_pred,  average='macro'),2)
            outstr = f"TC{TC}-{name}: f1={fmeasure}, recall={recall}, precision={precision}"
            print(outstr)

            timeTaken = f'{(time.time() - start_time)/60:.2f}'
        
            # plot_prediction(X_test, y_pred, y_test, outstr)
