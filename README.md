# AutoConf

dataset folder:

    TC1 --- DJI Windy dataset
    TC2 --- DJI Velocity sensor fault dataset
    TC3 --- Ardupilot Gyro sensor fault dataset
    TC4 --- PX4 Vibrate dataset
    

script folder:

    - search.py : for finding the best configuration for a given reference/train dataset, e.g. dataset/TC1/train.csv

    - evaluate.py for evaluating the best configured model against the test dataset, e.g., dataset/TC1/test.csv. 
            This script is also used for comparing the anomaly detection performance against Silhouette-Only-Approach (RQ1) 
            and the search efficiency against Random Search (RQ3)

    - compare.py : for comparing against baseline anomaly detection algorithms --- 
                    One-Class SVM, Local Outlier Factor, Isolation Forest (RQ2)

Instruction:
1. place "dataset" folder and "script" folder in a workspace

2. create "output" folder in the same workspace

3. create a Python virtual environment
    - $ python -m venv .

4. Activate Python virtual environment
    - $ source .venv/bin/activate

5. Install required Python Libraries from requirements.txt file
    - $ pip install -r requirements.txt

6. Specify the dataset and perform the search 
    - $ python3 search.py dataset/TC1/train.csv

7. Evaluate the best configuration found by search.py against the test dataset 
    - $ python3 evaluate.py dataset/TC1/train.csv dataset/TC1/test.csv

8. Comparison with other anomaly detection algorithms
    - $ python3 compare.py
