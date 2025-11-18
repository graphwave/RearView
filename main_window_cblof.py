
import pandas as pd
import numpy as np
import torch
from sklearn import preprocessing
import dgl
import time
import networkx as nx
from sklearn.model_selection import train_test_split
from model import *
from utils import *
import gc
from sklearn.metrics import confusion_matrix, classification_report
import os
from sklearn.preprocessing import StandardScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from dgl import batch
from VAE_model import *
from typing import Tuple, Dict, List, Any
from collections import deque
import hashlib
from collections import defaultdict
from scipy import stats
import pathlib
import seaborn as sns
import matplotlib.pyplot as plt
import time
from memory_profiler import profile

DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
TIME_COLUMN = "timestamp"
prefix = "data"
prefix_model = "model_apt2024_1019"
prefix_result = "result_apt2024_1019"
csvfile = "apt2024"




    

file_name = os.path.join(prefix, f"{csvfile}.csv")
vae_model_save_path = os.path.join(prefix_model, f"vae_models_{csvfile}.pt")
vae_global_model_save_path = os.path.join(prefix_model, f"vae_global_models_{csvfile}.pt")
dgi_model_save_path = os.path.join(prefix_model, f"{csvfile}2.pkl")
result_file = os.path.join(prefix_result, f"{csvfile}.txt")
tuple_file = os.path.join("tuple", f"{csvfile}_tuple.csv")

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
SESSION_THRESHOLD = 1000000
TRIPLET_MODEL_MAP = {}
GLOBAL_VAE_MODEL = {}

GLOBAL_SESSION_CACHE = deque(maxlen=SESSION_THRESHOLD)
ANOMALY_TRIGGER = False

ALL_PREDICTION_LABEL = []
ALL_TRUE_LABEL = []
ALL_GLOBAL_RECON = []
ALL_GLOBAL_LATENT = []
ALL_TEST_VAE_PROB = []
ALL_TEST_LOF_PROB = []

ALL_SRCIP_LIST = []
ALL_DSTIP_LIST = []
ALL_SRCPORT_LIST = []
ALL_DSTPORT_LIST = []

ALL_VAE_TRUE_LABEL = []
ALL_VAE_TEST_LABEL = []

TRIPLET_PREDICTIONS = defaultdict(lambda: {'true_labels': [], 'pred_labels': []})

def load_and_preprocess_data():

    data = pd.read_csv(file_name).fillna(0)
    data.rename(columns=lambda x: x.strip(), inplace=True)
    

    ip_cols = ["srcip", "dstip", "srcport", "dstport"]
    for col in ip_cols:
        data[col] = data[col].astype(str)
    if(csvfile == 'ids2017'):
        data = data[~((data['Label'] == 0) & 
                ((data['srcip'] == '172.16.0.1') | (data['dstip'] == '172.16.0.1')) )]
        data = data[~((data['Label'] == 1) & 
                ((data['srcip'] != '172.16.0.1') & (data['dstip'] != '172.16.0.1')))]
        data = data[~((data['Label'] == 0) & (data['timestamp']> 1499183999))]
    if(csvfile == 'unsw'):
        data = data[~(data['Attack'] == 'unknown_sessions')]

    data.sort_values(by=TIME_COLUMN, inplace=True)
    data.reset_index(drop=True, inplace=True)
    #data = data.sample(frac=1, random_state=42).reset_index(drop=True)
    return data

def filter_data_by_triple(data, csv_file):
    aciiot_df = pd.read_csv(csv_file, delimiter=',', names=['ip', 'port', 'protocol', 'degree'])
    csv_triples = set(zip(aciiot_df['ip'], aciiot_df['port'], aciiot_df['protocol']))
    
    #data_filter = data.copy()
    data['src_triple'] = list(zip(data['srcip'], data['srcport'], data['protocol'].astype(str)))
    data['dst_triple'] = list(zip(data['dstip'], data['dstport'], data['protocol'].astype(str)))
    
    data['match'] = data['src_triple'].isin(csv_triples) | data['dst_triple'].isin(csv_triples)
    
    filtered_data = data[data['match']].drop(columns=['src_triple', 'dst_triple', 'match'])
    return filtered_data
def filter_data_by_ip(data, csv_file):
    aciiot_df = pd.read_csv(csv_file, delimiter=',', names=['ip', 'port', 'protocol', 'degree'])

    ip_list = aciiot_df['ip'].tolist()

    filtered_data = data[(data['srcip'].isin(ip_list)) | (data['dstip'].isin(ip_list))]

    return filtered_data

def split_dataset(data: pd.DataFrame):
    X = data.drop(columns=["Attack", "Label"])
    y = data[["Attack", "Label"]]
    
    mask_label0 = y['Label'] == 0
    X_label0, y_label0 = X[mask_label0], y[mask_label0]
    X_non_label0, y_non_label0 = X[~mask_label0], y[~mask_label0]
    
    n_label0 = len(X_label0)
    split_idx = int(n_label0 * 0.8)
    X_train, y_train = X_label0.iloc[:split_idx], y_label0.iloc[:split_idx]
    X_val, y_val = X_label0.iloc[split_idx:], y_label0.iloc[split_idx:]
    
    X_test = pd.concat([X_val, X_non_label0])
    y_test = pd.concat([y_val, y_non_label0])
    
    #return X_train.join(y_train), X_test.join(y_test)
    return pd.concat([X_train, y_train], axis=1), pd.concat([X_test, y_test], axis=1)

def normalize_features(train_df: pd.DataFrame, test_df: pd.DataFrame):
    cols_to_norm = train_df.columns[7:-2]
    #scaler = preprocessing.Normalizer()
    scaler = MinMaxScaler()
    train_df['h'] = scaler.fit_transform(train_df[cols_to_norm]).tolist()
    test_df['h'] = scaler.transform(test_df[cols_to_norm]).tolist()
    
    #train_df['h'] = train_df[cols_to_norm].values.tolist()
    #test_df['h'] = test_df[cols_to_norm].values.tolist()
    return train_df, test_df
def build_triplet_mapping(aciiot_file: str):

    aciiot_df = pd.read_csv(aciiot_file, delimiter=',', names=['ip', 'port', 'protocol', 'degree'])
    

    aciiot_df = aciiot_df.drop(0).reset_index(drop=True)
    aciiot_df['triplet'] = aciiot_df.apply(
        lambda x: f"{x['ip']}_{x['port']}_{x['protocol']}", axis=1  # 假设port为srcport
    )
    return aciiot_df.set_index('triplet')

def match_triplet_data(train_df: pd.DataFrame, triplet_map: pd.DataFrame):


    train_df['src_triplet'] = train_df.apply(
        lambda x: f"{x['srcip']}_{x['srcport']}_{x['protocol']}", axis=1
    )
    train_df['dst_triplet'] = train_df.apply(
        lambda x: f"{x['dstip']}_{x['dstport']}_{x['protocol']}", axis=1
    )
    
    valid_triplets = set(triplet_map.index)
    train_df['matched_triplet'] = np.select(
        condlist=[
            train_df['src_triplet'].isin(valid_triplets),
            train_df['dst_triplet'].isin(valid_triplets)
        ],
        choicelist=[train_df['src_triplet'], train_df['dst_triplet']],
        default=None
    )
    

    grouped = train_df[train_df['matched_triplet'].notna()].groupby('matched_triplet')
    return {name: group for name, group in grouped}


#@profile
def train_vae_models(triplet_data: Dict[str, pd.DataFrame], params: Dict[str, Any]):

    models = {}
    for triplet, df in triplet_data.items():
        

        #if(triplet != '149.171.126.2_80_6'):
        #    continue

        if df.empty or len(df) < 50:
            continue
        if len(df) > 10000:
            df = df.iloc[:10000]
        print(f'now training for triplet {triplet} ============')

        X = np.array(df['h'].tolist())
        Y = np.zeros(len(df), dtype=int)
        

        scaler = MinMaxScaler()
        cols_to_norm = df.columns[7:-6]
        #scaler = preprocessing.Normalizer()
        X_scaled = scaler.fit_transform(df[cols_to_norm])
        #X_scaled = scaler.fit_transform(X)
        


        model_info = train_process(X_scaled, Y, params) 


        models[triplet] = {
            'model': model_info['model'].to(DEVICE),
            'stats': model_info['stats'], 
            'input_dim': X.shape[1],
            'scaler': scaler
        }
    
    return models

'''
def train_vae_global_models(df_train, parameters, sample_ratio=1):

    if 'Label' not in df_train.columns:
        raise ValueError("error")


    
    
    #n_samples = int(len(df_train) * sample_ratio)
    #sampled_indices = np.random.choice(len(df_train), size=n_samples, replace=False)
    #df_sampled = df_train.iloc[sampled_indices].reset_index(drop=True)
    n_samples = int(len(df_train) * sample_ratio)
    df_sampled = df_train.iloc[:n_samples].reset_index(drop=True) 

    feature_columns = [col for col in df_sampled.columns if col not in ['Label', 'Attack']]
    VAE_train_X = df_sampled[feature_columns].values
    VAE_train_Y = df_sampled['Label'].values 
    print(f"{len(df_train)}, {len(df_sampled)}")


    results = train_global_process(VAE_train_X, VAE_train_Y, parameters)
    return results

def train_vae_models3(triplet_data: Dict[str, pd.DataFrame], params: Dict[str, Any]):

    models = {}
    for triplet, df in triplet_data.items():



        if df.empty or len(df) < 100:
            print(f"warning")
            continue
        if len(df) > 10000:
            df = df.iloc[:10000]
            
        X = np.array(df['h'].tolist())
        Y = np.zeros(len(df), dtype=int) 


        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)


        model = IsolationForest(
            n_estimators=params.get('n_estimators', 100),
            contamination=params.get('contamination', 0.2),
            random_state=42
        )
        model.fit(X_scaled)
        

        train_scores = model.decision_function(X_scaled)
        threshold = np.quantile(train_scores, 0.2)
        
        models[triplet] = {
            'model': model,
            'threshold': threshold,
            'input_dim': X.shape[1],
            'scaler': scaler
        }

    
    return models
'''



def build_session_graphs(df: pd.DataFrame, group_size: int = 1000000, anomaly=False):

    graphs = []
    num_groups = len(df) // group_size + 1
    
    for i in range(num_groups):
        start_idx = i * group_size
        session_data = df.iloc[start_idx:start_idx+group_size].copy()
        if session_data.empty:
            continue
        session_data['original_index'] = session_data.index    

        if(anomaly):
            session_g = nx.from_pandas_edgelist(
                session_data, "srcip", "dstip", 
                edge_attr=["h", "Label", "Attack",  "timestamp", "vae_mu", "vae_recon_error", "vae_anomaly_prob", "original_index"],
                create_using=nx.MultiDiGraph()
            )
            dgl_g = dgl.from_networkx(session_g, edge_attrs=["h", "Label", "Attack",  "timestamp", "vae_mu", "vae_recon_error", "vae_anomaly_prob", "original_index"])
        else:#
            session_g = nx.from_pandas_edgelist(
                session_data, "srcip", "dstip", 
                edge_attr=["h", "Label", "Attack",  "timestamp", "vae_mu", "vae_recon_error", "vae_anomaly_prob"],
                create_using=nx.MultiDiGraph()
            )
            dgl_g = dgl.from_networkx(session_g, edge_attrs=["h", "Label", "Attack",  "timestamp", "vae_mu", "vae_recon_error", "vae_anomaly_prob"])

        dgl_g = dgl_g.to(DEVICE)

        #edim = len(session_data.iloc[0]['h'])
        edim = len(session_data.iloc[0]['vae_mu']) if len(session_data) > 0 else 0
        nfeat_weight = torch.ones([dgl_g.number_of_nodes(), edim], dtype=torch.float32, device=DEVICE) 
        dgl_g.ndata['h'] = torch.reshape(nfeat_weight, (dgl_g.num_nodes(), 1, edim))

        #dgl_g.ndata['h'] = torch.ones((dgl_g.num_nodes(), 1, edim), device=DEVICE)
        #dgl_g.edata['h'] = torch.tensor(np.array(session_data['h'].tolist()),  dtype=torch.float32, device=DEVICE).view(-1, 1, edim)
        edge_mu = torch.tensor(np.stack(dgl_g.edata['vae_mu'].cpu().numpy()), dtype=torch.float32, device=DEVICE)
        edge_recon_error = dgl_g.edata['vae_recon_error'].unsqueeze(1).float() 
        edge_anomaly_prob = dgl_g.edata['vae_anomaly_prob'].unsqueeze(1).float() 
        if(edge_mu.dim() > 2):
            edge_mu = edge_mu.squeeze(dim=1)
        #dgl_g.edata['h'] = (torch.cat([edge_mu, edge_recon_error, edge_anomaly_prob], dim=1)).view(-1, 1, edim) 
        dgl_g.edata['h'] = (edge_mu).view(-1, 1, edim)
        #h_features = dgl_g.edata['h']
        #dgl_g.edata['h'] = h_features.view(-1, 1, edim)
        del dgl_g.edata['vae_mu'], dgl_g.edata['vae_recon_error']
        graphs.append(dgl_g)
    return graphs
#@profile
def train_dgi_model(dgi, train_graphs: List[dgl.DGLGraph], val_graphs: List[dgl.DGLGraph], params: dict):

    ndim_in = params['ndim_in']
    edim = params['edim']
    #dgi = DGI(ndim_in, params['ndim_out'], edim, torch.relu).to(DEVICE).float()
    optimizer = torch.optim.Adam(dgi.parameters(), lr=params['lr'], weight_decay=5e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    best_val_loss = float('inf')
    cnt_wait = 0
    
    for epoch in range(params['epochs']):
        dgi.train()
        total_loss = 0.0
        
        for g in train_graphs:
            optimizer.zero_grad()
            loss = dgi(g, g.ndata['h'], g.edata['h'])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(dgi.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
            del g
            gc.collect()
        
        avg_train_loss = total_loss / len(train_graphs)
        

        dgi.eval()
        val_loss = 0.0
        with torch.no_grad():
            for g in val_graphs:
                loss = dgi(g, g.ndata['h'], g.edata['h'])
                val_loss += loss.item()
        avg_val_loss = val_loss / len(val_graphs)
        

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(dgi.state_dict(), dgi_model_save_path)
            cnt_wait = 0
        else:
            cnt_wait += 1
            if cnt_wait >= 50:
                print("Early stopping triggered")
                break
        
        scheduler.step(avg_val_loss)
        print(f"Epoch {epoch+1}/{params['epochs']} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
    
    #dgi.load_state_dict(torch.load(dgi_model_save_path))
    return dgi



#@profile
def update_session_cache(dgi, sample, is_anomaly, df_train, anomaly_prob):

    #global GLOBAL_SESSION_CACHE, ANOMALY_TRIGGER, ALL_PREDICTION_LABEL, ALL_TRUE_LABEL, ALL_TEST_VAE_PROB, ALL_TEST_LOF_PROB,ALL_GLOBAL_LATENT,ALL_GLOBAL_RECON
    global GLOBAL_SESSION_CACHE, ANOMALY_TRIGGER, ALL_PREDICTION_LABEL, ALL_TRUE_LABEL, ALL_TEST_VAE_PROB, ALL_TEST_LOF_PROB

    if is_anomaly:
        ANOMALY_TRIGGER = True
    

    session_data = {
        'srcip': sample['srcip'],
        'dstip': sample['dstip'],
        'srcport': sample['srcport'],
        'dstport': sample['dstport'],
        'h': sample['h'],
        'Label': sample['Label'],
        'Attack': sample['Attack'],
        'is_anomaly': is_anomaly,
        'timestamp':sample['timestamp'],
        'vae_mu':sample['vae_mu'],
        'vae_anomaly_prob':sample['vae_anomaly_prob'],
        'vae_recon_error': sample['vae_recon_error']
    }

    GLOBAL_SESSION_CACHE.append(session_data)
    

    if len(GLOBAL_SESSION_CACHE) >= SESSION_THRESHOLD and ANOMALY_TRIGGER:
        build_and_detect_graph(dgi, df_train)
        reset_trigger()
    elif len(GLOBAL_SESSION_CACHE) >= SESSION_THRESHOLD and not ANOMALY_TRIGGER:
        retired_sessions = []
        total = len(GLOBAL_SESSION_CACHE)
        remove_count = total * 4 // 5


        for _ in range(remove_count):
            retired_sessions.append(GLOBAL_SESSION_CACHE.popleft())

        for session in retired_sessions:
            ALL_TRUE_LABEL.append(session['Label'])
            ALL_PREDICTION_LABEL.append(session['is_anomaly'])
            #ALL_GLOBAL_RECON.append(session['anomaly_prob'])
            #ALL_GLOBAL_LATENT.append(session['anomaly_prob'])
            ALL_TEST_VAE_PROB.append(session['anomaly_prob'])
            ALL_TEST_LOF_PROB.append(session['anomaly_prob'])
    


    #GLOBAL_SESSION_CACHE = deque(active_samples, maxlen=SESSION_THRESHOLD)

def reset_trigger():

    global GLOBAL_SESSION_CACHE, ANOMALY_TRIGGER
    #GLOBAL_SESSION_CACHE.clear()
    ANOMALY_TRIGGER = False

from sklearn.mixture import GaussianMixture
def detect_by_gmm(df_train, df_test):

    raw_benign_train_samples = df_train[df_train.Label == 0].drop(columns=["Label", "Attack"])
    raw_normal_train_samples = df_train.drop(columns=["Label", "Attack"])

    raw_train_labels = df_train["Label"]
    raw_test_labels = df_test["Label"]
    vae_test_probs = df_test["anomaly_prob"]
    raw_test_samples = df_test.drop(columns=["Label", "timestamp", "anomaly_prob", "original_index"])
    gmm = GaussianMixture(n_components=3, covariance_type='full', random_state=42)
    gmm.fit(raw_normal_train_samples)
    test_scores = gmm.score_samples(raw_test_samples)
    train_scores = gmm.score_samples(raw_normal_train_samples)
    threshold = np.percentile(train_scores, 5) 


    test_predictions = (test_scores < threshold).astype(int)
    result_df = pd.DataFrame({
        'test_scores': test_scores,
        'vae_test_probs':vae_test_probs,
        'true_label':raw_test_labels,
        'test_label': test_predictions
    })
    return result_df

#@profile
def build_and_detect_graph(dgi, df_train, final_batch = False):

    global GLOBAL_SESSION_CACHE 
    global ALL_SRCIP_LIST, ALL_DSTIP_LIST, ALL_SRCPORT_LIST, ALL_DSTPORT_LIST

    session_data = list(GLOBAL_SESSION_CACHE)
    df = pd.DataFrame(session_data)
    

    starttime = time.time()
    graphs = build_session_graphs(df, group_size=len(df), anomaly=True)
    endtime = time.time()

    if not graphs:
        return
    

    dgi.eval()
    embeddings = []
    all_test_edge_scores = []
    with torch.no_grad():
        for g in graphs:
            node_feats = g.ndata['h'].to(DEVICE)
            edge_feats = g.edata['h'].to(DEVICE)
            #emb = dgi.encoder(g, node_feats, edge_feats)[1].cpu().numpy()
            starttime = time.time()
            emb = dgi.encoder(g, node_feats, edge_feats)[1]
            endtime = time.time()

            emb = emb.detach().cpu().numpy()
            embeddings.append(emb)
            #test_subgraph_emb = torch.sigmoid(emb.mean(dim=0))
            #discriminator_weight = dgi.discriminator.weight
            #scores = torch.matmul(emb, torch.matmul(discriminator_weight, test_subgraph_emb))
            
            #scores_np = scores.detach().cpu().numpy()
            #all_test_edge_scores.append(scores_np)
            



    embeddings = np.vstack(embeddings)

    embeddings_test = pd.DataFrame(embeddings)
    embeddings_test["Label"] = np.concatenate([
        g.edata['Label'].detach().cpu().numpy()
        for g in graphs
    ])
    embeddings_test["timestamp"] = np.concatenate([
        g.edata['timestamp'].detach().cpu().numpy()
        for g in graphs
    ])
    embeddings_test["anomaly_prob"] = np.concatenate([
        g.edata['vae_anomaly_prob'].detach().cpu().numpy()
        for g in graphs
    ])
    is_unique = embeddings_test["timestamp"].is_unique
    embeddings_test["original_index"] = np.concatenate([
        g.edata['original_index'].detach().cpu().numpy()
        for g in graphs
    ])

    #embeddings_test = embeddings_test.sort_values(by="timestamp", ascending=True).reset_index(drop=True)
    embeddings_test = embeddings_test.sort_values(by="original_index",  ascending=True).reset_index(drop=True)
    srcip_list = [item["srcip"] for item in session_data]
    dstip_list = [item["dstip"] for item in session_data]
    srcport_list = [item["srcport"] for item in session_data]
    dstport_list = [item["dstport"] for item in session_data]
    embeddings_test["srcip"] = srcip_list
    embeddings_test["dstip"] = dstip_list
    embeddings_test["srcport"] = srcport_list
    embeddings_test["dstport"] = dstport_list


    _,_,labels, result_df = cblof_detection(df_train, embeddings_test)
    #result_df = detect_by_global_vae_with_kde(embeddings_test)
    #result = detect_by_gmm(df_train, embeddings_test)


    retired_sessions = []
    total = len(GLOBAL_SESSION_CACHE)
    if final_batch:
        remove_count = total
    else:
        remove_count = total * 4 // 5
    for _ in range(remove_count):
        retired_sessions.append(GLOBAL_SESSION_CACHE.popleft())
    for i, session in enumerate(retired_sessions):
        ALL_TRUE_LABEL.append(session['Label'])
        #ALL_PREDICTION_LABEL.append(result_df['is_anomaly'][i])
        ALL_PREDICTION_LABEL.append(result_df['test_pred'][i])
        #ALL_GLOBAL_RECON.append(result_df['recon_anomaly_prob'][i])
        #ALL_GLOBAL_LATENT.append(result_df['latent_anomaly_prob'][i])
        ALL_TEST_VAE_PROB.append(result_df['vae_test_probs'][i])
        ALL_TEST_LOF_PROB.append(result_df['lof_test_probs'][i])
        ALL_SRCIP_LIST.append(result_df['srcip'][i])
        ALL_DSTIP_LIST.append(result_df['dstip'][i])
        ALL_SRCPORT_LIST.append(result_df['srcport'][i])
        ALL_DSTPORT_LIST.append(result_df['dstport'][i])

#@profile
def cblof_detection(df_train, df_test):
    raw_benign_train_samples = df_train[df_train.Label == 0].drop(columns=["Label", "Attack"])
    raw_normal_train_samples = df_train.drop(columns=["Label", "Attack"])

    raw_train_labels = df_train["Label"]
    raw_test_labels = df_test["Label"]
    vae_test_probs = df_test["anomaly_prob"]
    srcip = df_test["srcip"]
    dstip = df_test["dstip"]
    srcport = df_test["srcport"]
    dstport = df_test["dstport"]
    raw_test_samples = df_test.drop(columns=["Label", "timestamp", "anomaly_prob", "original_index", "srcip", "dstip", "srcport", "dstport"])
    n_est = [1, 2, 3, 5, 7, 10, 15]
    contamination = [0.001, 0.01, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5]
    params = list(itertools.product(n_est, contamination))
    score = -1
    best_params = {}
    bs = None
    results = [] 
    for n_est, con in params:
        try:
            #clf_b = CBLOF(n_clusters=n_est, contamination=con)
            clf_b = CBLOF(
                n_clusters=n_est,
                contamination=con,
                check_estimator=False
            )
            clf_b.fit(raw_benign_train_samples)
        except ValueError as e:
            continue  
        starttime = time.time()
        y_pred = clf_b.predict(raw_test_samples)
        endtime = time.time()
        lof_test_probs = clf_b.predict_proba(raw_test_samples)[:, 1]
        test_pred = y_pred.copy()
        test_pred[vae_test_probs > 0.95] = 1  

        f1 = f1_score(raw_test_labels, test_pred, average='macro')
        #precision = precision_score(raw_test_labels, test_pred)
        #recall = recall_score(raw_test_labels, test_pred)
        #specificity = recall_score(raw_test_labels, test_pred, pos_label=0)
            
        if f1 > score:
            score = f1
            best_params = {'n_estimators': n_est, "con": con}
            best_vae_probs = vae_test_probs.copy()
            best_lof_probs = lof_test_probs.copy()
            best_test_pred = test_pred.copy()
            best_raw_labels = raw_test_labels.copy()

        del clf_b
        gc.collect()

    if best_lof_probs is not None:
        result_df = pd.DataFrame({
            'srcip':srcip,
            'dstip':dstip,
            'srcport':srcport,
            'dstport':dstport,
            'vae_test_probs': best_vae_probs,
            'lof_test_probs': best_lof_probs,
            'test_pred': best_test_pred,
            'raw_test_labels': best_raw_labels
        })
        
    return best_params, score, best_test_pred, result_df



def load_vae_models(model_path):

    global TRIPLET_MODEL_MAP
    TRIPLET_MODEL_MAP = torch.load(model_path, map_location=DEVICE)
    #TRIPLET_MODEL_MAP = torch.load(model_path, map_location=DEVICE, weights_only=False)


#@profile
def detect_by_vae_with_kde(sample, triplet_src, triplet_dst, cols_to_norm, weight_recon=0.9, threshold=0.6):
    """

    """
    global ALL_VAE_TRUE_LABEL, ALL_VAE_TEST_LABEL, TRIPLET_PREDICTIONS
    

    triplet = triplet_src if triplet_src in TRIPLET_MODEL_MAP else triplet_dst
    model_info = TRIPLET_MODEL_MAP[triplet]
    model = model_info['model'].to(DEVICE)
    kde_recon = model_info['stats']['kde_recon']
    kde_latent = model_info['stats']['kde_latent']
    recon_stats = model_info['stats']['recon_stats']
    latent_stats = model_info['stats']['latent_stats']
    scaler = model_info['scaler']


    #scaler = preprocessing.Normalizer()

    #sample['h'] = scaler.transform(np.array(sample['h']).reshape(1, -1)).flatten().tolist()
    #sample['h'] = scaler.transform(sample.iloc[7:-3].values.reshape(1, -1)).flatten().tolist()
    sample_features = sample[cols_to_norm].to_frame().T
    normalized_features = scaler.transform(sample_features).flatten().tolist()
    sample['h'] = normalized_features

    data = torch.tensor(np.array(sample['h']).reshape(1, -1), dtype=torch.float32).to(DEVICE)
    

    starttime = time.time()
    model.eval()
    with torch.no_grad():
        recon_batch, mu, logvar = model(data)
        recon_error = torch.mean((recon_batch - data) ** 2, dim=1).item()
        latent_norm = torch.norm(mu, dim=1).item()
    

    if np.isnan(recon_error) or np.isinf(recon_error):
        recon_density = 0
    else:
        recon_density  = np.exp(kde_recon.score_samples([[recon_error]])[0]) 
    if np.isnan(latent_norm) or np.isinf(latent_norm):
        latent_density = 0
    else:
        latent_density  = np.exp(kde_latent.score_samples([[latent_norm]])[0])
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
    recon_z = (recon_density - recon_stats['density_mean']) / (recon_stats['density_std'] + 1e-8)
    latent_z = (latent_density - latent_stats['density_mean']) / (latent_stats['density_std'] + 1e-8)

    recon_anomaly_prob = sigmoid(-recon_z)
    latent_anomaly_prob = sigmoid(-latent_z)

    

    anomaly_prob = weight_recon * recon_anomaly_prob + (1 - weight_recon) * latent_anomaly_prob
    

    pred_label = 1 if anomaly_prob > threshold else 0
    endtime = time.time()


    true_label = sample['Label']
    TRIPLET_PREDICTIONS[triplet]['true_labels'].append(true_label)
    TRIPLET_PREDICTIONS[triplet]['pred_labels'].append(pred_label)
    ALL_VAE_TRUE_LABEL.append(true_label)
    ALL_VAE_TEST_LABEL.append(pred_label)
    
    return {
        'is_anomaly': pred_label == 1,
        'anomaly_prob': anomaly_prob,
        'normal_prob': 1 - anomaly_prob,
        'recon_error': recon_error,
        'mu':mu,
        'latent_norm': latent_norm,
        'recon_density': recon_density,
        'latent_density': latent_density,
        'sample':sample
    }



def add_vae_features_to_train(df, triplet_model_map, cols_to_norm, vae_params):

    #vae_h_list = []
    vae_mu_list = []
    vae_recon_error_list = []
    vae_anomaly_prob_list = []
    weight_recon=0.9
    for idx, sample in df.iterrows():

        triplet_src = f"{sample['srcip']}_{sample['srcport']}_{sample['protocol']}"
        triplet_dst = f"{sample['dstip']}_{sample['dstport']}_{sample['protocol']}"
        if triplet_src in triplet_model_map:
            triplet = triplet_src
        elif triplet_dst in triplet_model_map:
            triplet = triplet_dst
        else:

            #vae_h_list.append(np.zeros(256)) 
            vae_mu_list.append(np.zeros(vae_params['latent_dim']))#
            vae_recon_error_list.append(0.0)#
            vae_anomaly_prob_list.append(1.0)
            continue
        

        model_info = triplet_model_map[triplet]
        model = model_info['model'].to(DEVICE)
        scaler = model_info['scaler']
        kde_recon = model_info['stats']['kde_recon']
        kde_latent = model_info['stats']['kde_latent']
        recon_stats = model_info['stats']['recon_stats']
        latent_stats = model_info['stats']['latent_stats']
        

        sample_features = sample[cols_to_norm].to_frame().T
        normalized_features = scaler.transform(sample_features).flatten()
        x = torch.tensor(normalized_features, dtype=torch.float32).to(DEVICE).unsqueeze(0)
        

        model.eval()
        with torch.no_grad():
            recon_x, mu, logvar = model(x)
            recon_error = torch.mean((recon_x - x) ** 2, dim=1).item()
            latent_norm = torch.norm(mu, dim=1).item()
        



        if np.isnan(recon_error) or np.isinf(recon_error):
            recon_density = 0
        else:
            recon_density  = np.exp(kde_recon.score_samples([[recon_error]])[0]) 
        if np.isnan(latent_norm) or np.isinf(latent_norm):
            latent_density = 0
        else:
            latent_density  = np.exp(kde_latent.score_samples([[latent_norm]])[0]) 
        def sigmoid(x):
            return 1 / (1 + np.exp(-x))
        recon_z = (recon_density - recon_stats['density_mean']) / (recon_stats['density_std'] + 1e-8)
        latent_z = (latent_density - latent_stats['density_mean']) / (latent_stats['density_std'] + 1e-8)

        recon_anomaly_prob = sigmoid(-recon_z)
        latent_anomaly_prob = sigmoid(-latent_z)

        
        

        anomaly_prob = weight_recon * recon_anomaly_prob + (1 - weight_recon) * latent_anomaly_prob

        vae_anomaly_prob_list.append(anomaly_prob)
        vae_mu_list.append(mu.cpu().numpy().squeeze())
        vae_recon_error_list.append(recon_error)


    df['vae_mu'] = vae_mu_list
    df['vae_recon_error'] = vae_recon_error_list
    df['vae_anomaly_prob'] = vae_anomaly_prob_list
    return df



def detect_by_global_vae_with_kde(df_test, weight_recon=0.2, threshold=0.6):

    global GLOBAL_VAE_MODEL
    if not GLOBAL_VAE_MODEL:
        raise ValueError("GLOBAL_VAE_MODEL")
    
    model = GLOBAL_VAE_MODEL['model'].to(DEVICE)
    kde_recon = GLOBAL_VAE_MODEL['stats']['kde_recon']
    kde_latent = GLOBAL_VAE_MODEL['stats']['kde_latent']
    recon_stats = GLOBAL_VAE_MODEL['stats']['recon_stats']
    latent_stats = GLOBAL_VAE_MODEL['stats']['latent_stats']



    raw_test_samples = df_test.drop(columns=["Label", "timestamp", "anomaly_prob", "original_index"], errors='ignore')

    raw_test_labels = df_test["Label"]
    vae_test_probs = df_test["anomaly_prob"]
    X_test = torch.tensor(raw_test_samples.values, dtype=torch.float32, device=DEVICE)
    batch_size = X_test.size(0)


    model.eval()
    with torch.no_grad():
        recon_batch, mu, logvar = model(X_test) 
        recon_error = torch.mean((recon_batch - X_test) ** 2, dim=1).cpu().numpy()  # [batch_size]
        latent_norm = torch.norm(mu, dim=1).cpu().numpy()  # [batch_size]


    try:
        recon_density = np.exp(kde_recon.score_samples(recon_error.reshape(-1, 1)))  # [batch_size]
        latent_density = np.exp(kde_latent.score_samples(latent_norm.reshape(-1, 1)))  # [batch_size]
    except Exception as e:
        raise RuntimeError(f"KDE: {str(e)}")


    recon_z = (recon_density - recon_stats['density_mean']) / (recon_stats['density_std'] + 1e-8)
    latent_z = (latent_density - latent_stats['density_mean']) / (latent_stats['density_std'] + 1e-8)


    recon_anomaly_prob = 1 / (1 + np.exp(recon_z)) 
    latent_anomaly_prob = 1 / (1 + np.exp(latent_z))
    anomaly_prob = weight_recon * recon_anomaly_prob + (1 - weight_recon) * latent_anomaly_prob
    anomaly_prob[vae_test_probs > 0.95] = 1

    result_df = pd.DataFrame({
        'original_index': df_test.get('original_index', range(batch_size)), 
        'recon_error': recon_error,
        'latent_norm': latent_norm,
        'recon_density': recon_density,
        'latent_density': latent_density,
        'recon_anomaly_prob': recon_anomaly_prob,
        'latent_anomaly_prob': latent_anomaly_prob,
        'lof_test_probs': anomaly_prob,
        'vae_test_probs':vae_test_probs,
        'true_label':raw_test_labels,
        'is_anomaly': (anomaly_prob > threshold).astype(int)
    })
    return result_df

def normalize_single_sample(sample: pd.Series, scaler: MinMaxScaler, cols_to_norm: list) -> pd.Series:

        sample_df = pd.DataFrame([sample]) 
        features_df = sample_df[cols_to_norm]
        scaled_array = scaler.transform(features_df)
        scaled_df = pd.DataFrame(scaled_array, columns=cols_to_norm, index=sample_df.index)
        sample_df[cols_to_norm] = scaled_df
        return sample_df  #






    



def main():


    raw_data = load_and_preprocess_data()
    #raw_data = filter_data_by_triple(raw_data, tuple_file)
    raw_data = filter_data_by_ip(raw_data, tuple_file)

    #raw_data = raw_data.groupby(by='Attack').sample(frac=0.2, random_state=42)
    train_df, test_df = split_dataset(raw_data)
    #train_df_copy = train_df.copy(deep=True)#
    train_df, test_df = normalize_features(train_df, test_df)#
    cols_to_norm = train_df.columns[7:-3]  # 
    #scaler = preprocessing.Normalizer()
    #global_scaler = MinMaxScaler()
    #train_df['h'] = global_scaler.fit_transform(train_df[cols_to_norm]).tolist()
    #train_df_copy[cols_to_norm] = global_scaler.fit_transform(train_df_copy[cols_to_norm])
    #train_df_copy['h'] = train_df_copy.iloc[:, 7:-2].values.tolist()
    #train_df['h'] = train_df[cols_to_norm].values.tolist()
    #test_df['h'] = test_df[cols_to_norm].values.tolist()
    #test_df['h'] = global_scaler.transform(test_df[cols_to_norm]).tolist()

    encoder = preprocessing.LabelEncoder()
    encoder.fit(raw_data["Attack"])
    train_df["Attack"] = encoder.transform(train_df["Attack"])
    test_df["Attack"] = encoder.transform(test_df["Attack"])
    #train_df_copy["Attack"] = encoder.transform(train_df_copy["Attack"])

    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    #train_df_copy = train_df_copy.reset_index(drop=True)
    #test_df = test_df.iloc[:1000]

    triplet_map = build_triplet_mapping(tuple_file)
    triplet_data = match_triplet_data(train_df, triplet_map)
    
    vae_params = {
        'batch_size': 32,
        'hidden_dim': 128,
        'latent_dim': 16,
        'learning_rate': 1e-4,
        'epochs': 50
    }
    vae_global_params = {
        'batch_size': 32,
        'hidden_dim': 128,
        'latent_dim': 16,
        'learning_rate': 1e-4,
        'epochs': 50
    }

    folder_result = pathlib.Path(prefix_result)
    folder_model = pathlib.Path(prefix_model)
    if not folder_result.exists():
        folder_result.mkdir(parents=True, exist_ok=False)
    if not folder_model.exists():
        folder_model.mkdir(parents=True, exist_ok=False)
    

    initial_memory = {
        "nvidia_smi_total_mib": get_nvidia_smi_memory(device_id=1),
        "pytorch_allocated_mib": torch.cuda.memory_allocated(device) / (1024**2),
        "pytorch_reserved_mib": torch.cuda.memory_reserved(device) / (1024**2) 
    }





    vae_models = train_vae_models(triplet_data, vae_params)


    initial_memory = {
        "nvidia_smi_total_mib": get_nvidia_smi_memory(device_id=1), 
        "pytorch_allocated_mib": torch.cuda.memory_allocated(device) / (1024**2), 
        "pytorch_reserved_mib": torch.cuda.memory_reserved(device) / (1024**2)     
    }



    torch.save(vae_models, vae_model_save_path)

    load_vae_models(vae_model_save_path)


    train_df = add_vae_features_to_train(train_df, TRIPLET_MODEL_MAP, cols_to_norm, vae_params)



    latent_dim = vae_params['latent_dim']
    dgi_params = {
        #'ndim_in': len(train_df.iloc[0]['h']),
        'ndim_in': latent_dim,#16
        'hidden_dim':64,
        'ndim_out': 32,
        #'edim': len(train_df.iloc[0]['h']),
        'edim': latent_dim,#16
        'lr': 1e-3,
        'epochs':100
    }

    initial_memory = {
        "nvidia_smi_total_mib": get_nvidia_smi_memory(device_id=1), 
        "pytorch_allocated_mib": torch.cuda.memory_allocated(device) / (1024**2),  
        "pytorch_reserved_mib": torch.cuda.memory_reserved(device) / (1024**2) 
    }



    dgi_model = DGI(dgi_params['ndim_in'], dgi_params['ndim_out'], dgi_params['edim'], torch.relu).to(DEVICE).float()
    total_samples = len(train_df)
    val_ratio = 0.2
    val_samples = int(total_samples * val_ratio)
    train_sessions = build_session_graphs(train_df.iloc[:-val_samples], anomaly=False)
    val_sessions = build_session_graphs(train_df.iloc[-val_samples:], anomaly=False) 

    dgi_model = train_dgi_model(dgi_model, train_sessions, val_sessions, dgi_params)
    initial_memory = {
        "nvidia_smi_total_mib": get_nvidia_smi_memory(device_id=1),  
        "pytorch_allocated_mib": torch.cuda.memory_allocated(device) / (1024**2),  
        "pytorch_reserved_mib": torch.cuda.memory_reserved(device) / (1024**2)    
    }

    dgi_model.load_state_dict(torch.load(dgi_model_save_path))




    all_training_embs = []
    all_training_vae_embs = []
    for session_g in train_sessions:
        training_emb = dgi_model.encoder(session_g, session_g.ndata['h'], session_g.edata['h'])[1]
        #training_emb = training_emb.detach().cpu().numpy()
        #subgraph_emb = torch.sigmoid(training_emb.mean(dim=0))#
        #discriminator_weight = dgi_model.discriminator.weight
        #scores = torch.matmul(training_emb, torch.matmul(discriminator_weight, subgraph_emb))
        training_emb = training_emb.detach().cpu().numpy()
        #scores_np = scores.detach().cpu().numpy()
        all_training_embs.append(training_emb)
        


        all_training_vae_embs.append(session_g.edata['h'])
    
    training_emb = np.vstack(all_training_embs)
    df_train = pd.DataFrame(training_emb)

    df_train["Attack"] = np.concatenate([
        encoder.inverse_transform(g.edata['Attack'].detach().cpu().numpy())
        for g in train_sessions
    ])
    df_train["Label"] = np.concatenate([
        g.edata['Label'].detach().cpu().numpy()
        for g in train_sessions
    ])

    ##############################################################################################

    #global_vae_model = train_vae_global_models(df_train, vae_global_params, sample_ratio=0.5)
    #torch.save(global_vae_model, vae_global_model_save_path)
    #global GLOBAL_VAE_MODEL
    #GLOBAL_VAE_MODEL = torch.load(vae_global_model_save_path, map_location=DEVICE)

    ##############################################################################################





    global dgi 
    dgi = dgi_model
    #global ALL_PREDICTION_LABEL, ALL_TRUE_LABEL, ALL_TEST_VAE_PROB, ALL_TEST_LOF_PROB, ALL_GLOBAL_LATENT, ALL_GLOBAL_RECON#
    global ALL_PREDICTION_LABEL, ALL_TRUE_LABEL, ALL_TEST_VAE_PROB, ALL_TEST_LOF_PROB

    vae_result = []
    vae_recon_error = []
    vae_latent_norm = []
    vae_recon_threshold = []
    vae_latent_threshold = []
    vae_anomaly_prob = []
    i = 0
    for idx, sample in test_df.iterrows():

        
        #vae_true_label.append(sample['Label'])#
        triplet_src = f"{sample['srcip']}_{sample['srcport']}_{sample['protocol']}"
        triplet_dst = f"{sample['dstip']}_{sample['dstport']}_{sample['protocol']}"
        #sample_copy = sample.copy(deep=True)
        #sample_copy['h'] = normalize_single_sample(sample_copy, global_scaler, cols_to_norm)#

        if csvfile == 'aciiot':
            if triplet_src == '192.168.1.1_53_17' or triplet_dst == '192.168.1.1_53_17' \
                or triplet_src == '239.255.255.250_1900_17' or triplet_dst == '239.255.255.250_1900_17':
                continue
        if triplet_src not in TRIPLET_MODEL_MAP and triplet_dst not in TRIPLET_MODEL_MAP:
            #is_anomaly = True
            recon_error = 0
            continue
            sample['vae_mu'] = np.zeros(vae_params['latent_dim'], dtype=np.float32)  
            sample['vae_recon_error'] = 0.0  
            sample['vae_anomaly_prob'] = 1.0

            
            update_session_cache(dgi_model, sample, is_anomaly=True, df_train=df_train, \
                                    anomaly_prob=1.0)
        else:
            #i += 1
            #is_anomaly, anomaly_score = detect_anomaly_with_iforest(sample, triplet_src, triplet_dst)
            #starttime = time.time()
            vae_result  = detect_by_vae_with_kde(sample, triplet_src, triplet_dst, cols_to_norm)
            #endtime = time.time()
            #print(f"：{endtime - starttime}")
            vae_result['sample']['vae_mu'] = vae_result['mu'].cpu().numpy().squeeze()
            vae_result['sample']['vae_recon_error'] = vae_result['recon_error']
            vae_result['sample']['vae_anomaly_prob'] = vae_result['anomaly_prob']


            if isinstance(vae_result['recon_error'], torch.Tensor):
                vae_result['recon_error'] = vae_result['recon_error'].cpu().numpy()
            if isinstance(vae_result['latent_norm'], torch.Tensor):
                vae_result['latent_norm'] = vae_result['latent_norm'].cpu().numpy()
            if isinstance(vae_result['recon_density'], torch.Tensor):
                vae_result['recon_density'] = vae_result['recon_density'].cpu().numpy()
            if isinstance(vae_result['latent_density'], torch.Tensor):
                vae_result['latent_density'] = vae_result['latent_density'].cpu().numpy()
            if isinstance(vae_result['anomaly_prob'], torch.Tensor):
                vae_result['anomaly_prob'] = vae_result['anomaly_prob'].cpu().numpy()

            vae_recon_error.append(vae_result['recon_error'])
            vae_latent_norm.append(vae_result['latent_norm'])
            vae_recon_threshold.append(vae_result['recon_density'])
            vae_latent_threshold.append(vae_result['latent_density'])
            vae_anomaly_prob.append(vae_result['anomaly_prob'])


        #vae_result.append(1 if is_anomaly else 0)

            update_session_cache(dgi_model, vae_result['sample'], vae_result['is_anomaly'], df_train, vae_result['anomaly_prob'])#
        #update_session_cache(dgi_model, sample_copy, result['is_anomaly'], df_train, result['anomaly_prob'])
        

        if idx % 10000 == 0:
            gc.collect()
            print(f" {idx}/{len(test_df)}")


    if len(GLOBAL_SESSION_CACHE) > 0:
        build_and_detect_graph(dgi_model, df_train, final_batch = True)


    data = {
        'recon_error': vae_recon_error,
        'latent_norm': vae_latent_norm,
        'recon_density': vae_recon_threshold,
        'latent_density': vae_latent_threshold,
        'vae_anomaly_prob': vae_anomaly_prob,
        'test_label': ALL_VAE_TEST_LABEL,
        'true_label':ALL_VAE_TRUE_LABEL
    }
    df = pd.DataFrame(data)
    df.to_csv(prefix_result + '/' + csvfile + '_recon.csv', index=False)#

    data2 = {
        'srcip':ALL_SRCIP_LIST,
        'dstip':ALL_DSTIP_LIST,
        'srcport':ALL_SRCPORT_LIST,
        'dstport':ALL_DSTPORT_LIST,
        'test_vae_prob': ALL_TEST_VAE_PROB,
        'test_lof_prob': ALL_TEST_LOF_PROB,
        'test_label': ALL_PREDICTION_LABEL,
        'true_label': ALL_TRUE_LABEL
    }
    df1 = pd.DataFrame(data2)
    df1.to_csv(prefix_result + '/' + csvfile + '_prob.csv', index=False)#



    report = classification_report(ALL_TRUE_LABEL, ALL_PREDICTION_LABEL)
    print("\Classification Report for dgi:")
    print(report)
    print("\nConfusion Matrix for dgi:")
    cm = confusion_matrix(ALL_TRUE_LABEL, ALL_PREDICTION_LABEL)
    print(cm)
    with open(prefix_result + '/' + csvfile + '_all_metrics.txt', 'w') as f:
        f.write("Classification Report for dgi:\n")
        f.write(report)
        f.write("\n\nConfusion Matrix for dgi:\n")
        f.write(str(cm))


    save_reports_to_file(prefix_result + '/' + csvfile + '_vae_metrics.txt')

if __name__ == "__main__":
    main()



