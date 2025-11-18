import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, classification_report
from pyod.models.cblof import CBLOF
from pyod.models.hbos import HBOS
from pyod.models.pca import PCA
from sklearn.ensemble import IsolationForest
import itertools
import gc

def calculate_f1_score(df_train, df_test):
    benign_train_samples = df_train[df_train.Label == 0].drop(columns=["Label", "Attack"])
    normal_train_samples = df_train.drop(columns=["Label", "Attack"])
    train_labels = df_train["Label"]
    test_labels = df_test["Label"]
    test_samples = df_test.drop(columns=["Label", "Attack"])

    n_est = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    contamination_values = [0.01, 0.05, 0.1, 0.2]
    params = list(itertools.product(n_est, contamination_values))

    score = -1
    bs = None
    best_params = {}

    # CBLOF
    for n_est, con in params:
        try:
            clf_if = CBLOF(n_clusters=n_est, contamination=con)
            clf_if.fit(benign_train_samples)
            y_pred = clf_if.predict(test_samples)
            test_pred = y_pred
            f1 = f1_score(test_labels, test_pred, average='macro')
            if f1 > score:
                score = f1
                best_params = {'n_estimators': n_est, "con": con}
                bs = test_pred
        except ValueError:
            continue
        del clf_if
        gc.collect()



    return best_params, score, bs


def calculate_f1_score1(df_train, df_test):


    raw_benign_train_samples = df_train[df_train.Label == 0].drop(columns=["Label", "Attack"])
    raw_normal_train_samples = df_train.drop(columns=["Label", "Attack"])

    raw_train_labels = df_train["Label"]
    raw_test_labels = df_test["Label"]
    raw_test_samples = df_test.drop(columns=["Label", "Attack"])
    n_est = [2,3,5,7,9,10]
    contamination = [0.001, 0.01, 0.04, 0.05, 0.1, 0.2]
    params = list(itertools.product(n_est, contamination))
    score = -1
    best_params = {}
    bs = None
    for n_est, con in params:
        
        try:
            clf_b = CBLOF(n_clusters=n_est, contamination=con)
            clf_b.fit(raw_benign_train_samples)
        except ValueError as e:
            print(n_est)
            continue  
    
        y_pred = clf_b.predict(raw_test_samples)
        test_pred = y_pred

        f1 = f1_score(raw_test_labels, test_pred, average='macro')

        if f1 > score:
            score = f1
            best_params = {'n_estimators': n_est,
                            "con": con
                    }
            bs = test_pred
        del clf_b
        gc.collect()

    

    return best_params, score, bs


def float_to_ip(ip_str):
    try:
        ip_float = float(ip_str)
        ip_int = int(ip_float) & 0xFFFFFFFF
        return '.'.join([str((ip_int >> (i * 8)) & 0xFF) for i in range(3, -1, -1)])
    except (ValueError, TypeError):
        return ""

import os
def extract_and_save_npz_to_csv(data_dir, output_csv='extracted_data.csv'):

    all_data = []
    

    for filename in os.listdir(data_dir):
        if filename.endswith('.npz'):
            file_path = os.path.join(data_dir, filename)
            
            try:

                with np.load(file_path) as data:

                    x_features = data['X'][:,16,:112]
                    

                    y_label = data['Y']
                    
                    combined_features = np.hstack([x_features, y_label])
                    all_data.append(combined_features)
            
            except Exception as e:

                continue
    

    if not all_data:
        return
    final_data = np.vstack(all_data) 

    columns = [f'feature_{i+1}' for i in range(112)] + ['Attack', 'Label']
    columns[0] = 'IPV4_SRC_ADDR'
    columns[1] = 'IPV4_DST_ADDR'
    columns[2] = 'L4_SRC_PORT'
    columns[3] = 'L4_DST_PORT'

    df = pd.DataFrame(final_data, columns=columns)
    df['IPV4_SRC_ADDR'] = df['IPV4_SRC_ADDR'].apply(float_to_ip)
    df['IPV4_DST_ADDR'] = df['IPV4_DST_ADDR'].apply(float_to_ip)
    df.to_csv(output_csv, index=False)
    








import pandas as pd
import numpy as np

def add_timestamp_to_csv(input_file, output_file):

    df = pd.read_csv(input_file)
    

    start_time = pd.Timestamp.now()
    timestamps = pd.date_range(start=start_time, periods=len(df), freq='S')
    

    df['Timestamp'] = timestamps
    

    df.to_csv(output_file, index=False)


#input_file = 'NF-BoT-IoT-v2.csv'
#output_file = 'NF-BoT-IoT-v2_new.csv'
#add_timestamp_to_csv(input_file, output_file)



import pandas as pd

def swap_src_dst_for_specific_dstip(csv_path, output_path):


    data = pd.read_csv(csv_path).fillna(0)
    data.rename(columns=lambda x: x.strip(), inplace=True)
    

    mask = data['dstip'] == '172.16.0.1'
    specific_data = data[mask].copy()
    other_data = data[~mask].copy()

    specific_data[['srcip', 'dstip']] = specific_data[['dstip', 'srcip']]
    specific_data[['srcport', 'dstport']] = specific_data[['dstport', 'srcport']]
    

    combined_data = pd.concat([other_data, specific_data], ignore_index=True)
    
    

    combined_data.to_csv(output_path, index=False)



#swap_src_dst_for_specific_dstip("data/ids2017.csv", "data/ids2017_repair.csv")







