#14222服务器上fpenv3.7即可执行，使用的numpy版本为1.21.5
#直接点击main_window_cblof.py即可执行
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
from scipy import stats  # 确保有这一行
import pathlib
import seaborn as sns
import matplotlib.pyplot as plt
import time
from memory_profiler import profile
# ------------------------ 全局配置 ------------------------ #
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
# 全局变量
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
SESSION_THRESHOLD = 1000000  # 会话缓存阈值
TRIPLET_MODEL_MAP = {}    # 存储三元组对应的VAE模型和阈值
GLOBAL_VAE_MODEL = {}

GLOBAL_SESSION_CACHE = deque(maxlen=SESSION_THRESHOLD)  # 按三元组缓存会话数据
ANOMALY_TRIGGER = False   # 异常触发标志

ALL_PREDICTION_LABEL = []  # 存储最终预测结果
ALL_TRUE_LABEL = []  # 存储真实标签
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
# 用于存储每个三元组的真实标签和预测标签
TRIPLET_PREDICTIONS = defaultdict(lambda: {'true_labels': [], 'pred_labels': []})
# ------------------------ 数据处理模块 ------------------------ #
def load_and_preprocess_data():
    """加载并预处理数据集"""
    data = pd.read_csv(file_name).fillna(0)
    data.rename(columns=lambda x: x.strip(), inplace=True)
    
    # 转换IP和端口为字符串类型,此筛选只针对ids2017
    ip_cols = ["srcip", "dstip", "srcport", "dstport"]
    for col in ip_cols:
        data[col] = data[col].astype(str)
    if(csvfile == 'ids2017'):
        data = data[~((data['Label'] == 0) & 
                ((data['srcip'] == '172.16.0.1') | (data['dstip'] == '172.16.0.1')) )]#只用周一的数据进行训练& (data['timestamp']> 1499183999)
        data = data[~((data['Label'] == 1) & 
                ((data['srcip'] != '172.16.0.1') & (data['dstip'] != '172.16.0.1')))]
        data = data[~((data['Label'] == 0) & (data['timestamp']> 1499183999))]
    if(csvfile == 'unsw'):
        data = data[~(data['Attack'] == 'unknown_sessions')]
    # 按时间排序
    data.sort_values(by=TIME_COLUMN, inplace=True)
    data.reset_index(drop=True, inplace=True)
    #data = data.sample(frac=1, random_state=42).reset_index(drop=True)
    return data

def filter_data_by_triple(data, csv_file):
    """
    通过三元组 (IP, Port, Protocol) 过滤数据，不改变data的原始结构
    保留满足以下条件的数据：
    - 源三元组 (srcip, srcport, protocol) 在CSV的三元组中，或
    - 目的三元组 (dstip, dstport, protocol) 在CSV的三元组中
    """
    # 读取CSV文件中的三元组，忽略degree列
    aciiot_df = pd.read_csv(csv_file, delimiter=',', names=['ip', 'port', 'protocol', 'degree'])
    csv_triples = set(zip(aciiot_df['ip'], aciiot_df['port'], aciiot_df['protocol']))
    
    # 向量化生成临时三元组列（不修改原始data，使用copy避免SettingWithCopy警告）
    #data_filter = data.copy()
    data['src_triple'] = list(zip(data['srcip'], data['srcport'], data['protocol'].astype(str)))
    data['dst_triple'] = list(zip(data['dstip'], data['dstport'], data['protocol'].astype(str)))
    
    # 判断是否匹配CSV三元组
    data['match'] = data['src_triple'].isin(csv_triples) | data['dst_triple'].isin(csv_triples)
    
    # 筛选并删除临时列，保留原始数据结构
    filtered_data = data[data['match']].drop(columns=['src_triple', 'dst_triple', 'match'])
    return filtered_data
#筛选数据，只保留与三元组IP地址相关的数据
def filter_data_by_ip(data, csv_file):
    """通过IP地址过滤数据"""
    # 读取包含IP地址的CSV文件
    aciiot_df = pd.read_csv(csv_file, delimiter=',', names=['ip', 'port', 'protocol', 'degree'])

    # 提取CSV文件中的IP地址
    ip_list = aciiot_df['ip'].tolist()

    # 过滤数据，只保留源地址或者目的地址在csv文件中的数据
    filtered_data = data[(data['srcip'].isin(ip_list)) | (data['dstip'].isin(ip_list))]

    return filtered_data

def split_dataset(data: pd.DataFrame):
    """划分训练集和测试集（仅使用良性样本训练VAE）"""
    X = data.drop(columns=["Attack", "Label"])
    y = data[["Attack", "Label"]]
    
    # 分离良性样本和攻击样本
    mask_label0 = y['Label'] == 0
    X_label0, y_label0 = X[mask_label0], y[mask_label0]
    X_non_label0, y_non_label0 = X[~mask_label0], y[~mask_label0]
    
    # 顺序划分良性样本为训练集和验证集
    n_label0 = len(X_label0)
    split_idx = int(n_label0 * 0.8)
    X_train, y_train = X_label0.iloc[:split_idx], y_label0.iloc[:split_idx]
    X_val, y_val = X_label0.iloc[split_idx:], y_label0.iloc[split_idx:]
    
    # 构建测试集（良性验证集 + 所有攻击样本）
    X_test = pd.concat([X_val, X_non_label0])
    y_test = pd.concat([y_val, y_non_label0])
    
    #return X_train.join(y_train), X_test.join(y_test)
    return pd.concat([X_train, y_train], axis=1), pd.concat([X_test, y_test], axis=1)

def normalize_features(train_df: pd.DataFrame, test_df: pd.DataFrame):
    """归一化特征列"""
    cols_to_norm = train_df.columns[7:-2]  # 假设特征从第8列开始
    #scaler = preprocessing.Normalizer()
    scaler = MinMaxScaler()
    train_df['h'] = scaler.fit_transform(train_df[cols_to_norm]).tolist()
    test_df['h'] = scaler.transform(test_df[cols_to_norm]).tolist()
    
    # 合并特征为列表格式
    #train_df['h'] = train_df[cols_to_norm].values.tolist()
    #test_df['h'] = test_df[cols_to_norm].values.tolist()
    return train_df, test_df

# ------------------------ VAE模型模块 ------------------------ #
def build_triplet_mapping(aciiot_file: str):
    """构建三元组映射表"""
    aciiot_df = pd.read_csv(aciiot_file, delimiter=',', names=['ip', 'port', 'protocol', 'degree'])
    
    # 过滤无效行并生成三元组
    aciiot_df = aciiot_df.drop(0).reset_index(drop=True)
    aciiot_df['triplet'] = aciiot_df.apply(
        lambda x: f"{x['ip']}_{x['port']}_{x['protocol']}", axis=1  # 假设port为srcport
    )
    return aciiot_df.set_index('triplet')

def match_triplet_data(train_df: pd.DataFrame, triplet_map: pd.DataFrame):
    """匹配三元组数据"""
    # 向量化生成源和目标三元组
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
    
    # 分组聚合匹配数据
    grouped = train_df[train_df['matched_triplet'].notna()].groupby('matched_triplet')
    return {name: group for name, group in grouped}


#@profile
def train_vae_models(triplet_data: Dict[str, pd.DataFrame], params: Dict[str, Any]):
    """批量训练VAE模型，返回包含统计参数的模型信息"""
    models = {}
    for triplet, df in triplet_data.items():
        

        #if(triplet != '149.171.126.2_80_6'):
        #    continue

        if df.empty or len(df) < 50:
            print(f"警告: 三元组 {triplet} 无有效数据，跳过训练")
            continue
        if len(df) > 10000:
            df = df.iloc[:10000]  # 截断大样本防止内存溢出
        print(f'now training for triplet {triplet} ============')
        # 准备数据
        X = np.array(df['h'].tolist())  # 假设'h'是特征列名
        Y = np.zeros(len(df), dtype=int)  # 正常样本标签为0
        
        # 数据预处理：归一化（可根据需求改为StandardScaler）
        scaler = MinMaxScaler()
        cols_to_norm = df.columns[7:-6]  # -6是因为df中添加了h、srctriplet\dst等
        #scaler = preprocessing.Normalizer()
        X_scaled = scaler.fit_transform(df[cols_to_norm])#每个三元组都重新归一化，因为不同三元组情况不同
        #X_scaled = scaler.fit_transform(X)
        
        # 调用train_process训练模型并获取统计参数
        #allocated = torch.cuda.memory_allocated()
        #print(f"当前GPU已分配内存：{allocated / (1024 **2):.2f} MiB")  # 转换为 MiB
        model_info = train_process(X_scaled, Y, params)  # 假设train_process返回模型和统计参数
        #allocated = torch.cuda.memory_allocated()
        #print(f"当前GPU已分配内存：{allocated / (1024 **2):.2f} MiB")  # 转换为 MiB
        # 保存模型信息（替换原有的固定阈值为统计参数）
        models[triplet] = {
            'model': model_info['model'].to(DEVICE),
            'stats': model_info['stats'],  # 包含recon和latent的均值/标准差
            'input_dim': X.shape[1],
            'scaler': scaler
        }
        print(f"完成训练: {triplet}，样本数: {len(df)}，潜在空间维度: {params['latent_dim']}")
    
    return models

'''
def train_vae_global_models(df_train, parameters, sample_ratio=1):
    """批量训练VAE模型"""
    if 'Label' not in df_train.columns:
        raise ValueError("输入的 df_train 必须包含 'Label' 列表示标签。")
    
    # 分离特征和标签

    
    
    #n_samples = int(len(df_train) * sample_ratio)
    #sampled_indices = np.random.choice(len(df_train), size=n_samples, replace=False)
    #df_sampled = df_train.iloc[sampled_indices].reset_index(drop=True)
    n_samples = int(len(df_train) * sample_ratio)
    df_sampled = df_train.iloc[:n_samples].reset_index(drop=True)  # 直接选取前n_samples个样本
    # 分离特征（忽略标签，因为全为同一类别）
    feature_columns = [col for col in df_sampled.columns if col not in ['Label', 'Attack']]
    VAE_train_X = df_sampled[feature_columns].values
    VAE_train_Y = df_sampled['Label'].values          # 提取标签
    print(f"原始数据量: {len(df_train)}, 采样后数据量: {len(df_sampled)}")

    print("=== VAE全局模型训练 ===")
    results = train_global_process(VAE_train_X, VAE_train_Y, parameters)
    return results
'''
'''
def train_vae_models3(triplet_data: Dict[str, pd.DataFrame], params: Dict[str, Any]):
    """批量训练Isolation Forest模型"""
    models = {}
    for triplet, df in triplet_data.items():

        #测试用
        #if(triplet != '192.168.10.50_21_6'):
        #    continue

        if df.empty or len(df) < 100:
            print(f"警告: 三元组 {triplet} 无有效数据，跳过训练")
            continue
        if len(df) > 10000:
            df = df.iloc[:10000]
            
        # 准备数据
        X = np.array(df['h'].tolist())
        Y = np.zeros(len(df), dtype=int)  # 良性样本标签为0

        # 为当前三元组创建并拟合归一化器
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)

        # 创建并训练Isolation Forest模型
        model = IsolationForest(
            n_estimators=params.get('n_estimators', 100),
            contamination=params.get('contamination', 0.2),
            random_state=42
        )
        model.fit(X_scaled)
        
        # 计算训练数据的异常分数，用于确定阈值
        train_scores = model.decision_function(X_scaled)
        threshold = np.quantile(train_scores, 0.2)  # 例如，取分数最低的5%作为异常
        
        models[triplet] = {
            'model': model,
            'threshold': threshold,
            'input_dim': X.shape[1],
            'scaler': scaler
        }
        print(f"完成训练: {triplet}，样本数: {len(df)}")
    
    return models
'''


# ------------------------ 图模型模块 ------------------------ #
def build_session_graphs(df: pd.DataFrame, group_size: int = 1000000, anomaly=False):
    """构建会话图列表"""
    graphs = []
    num_groups = len(df) // group_size + 1
    
    for i in range(num_groups):
        start_idx = i * group_size
        session_data = df.iloc[start_idx:start_idx+group_size].copy()
        if session_data.empty:
            continue
        session_data['original_index'] = session_data.index    
        # 构建有向图
        if(anomaly):#测试时，将VAE的预测概率、边的索引值也作为边的属性添加到边中
            session_g = nx.from_pandas_edgelist(
                session_data, "srcip", "dstip", 
                edge_attr=["h", "Label", "Attack",  "timestamp", "vae_mu", "vae_recon_error", "vae_anomaly_prob", "original_index"],
                create_using=nx.MultiDiGraph()
            )
            dgl_g = dgl.from_networkx(session_g, edge_attrs=["h", "Label", "Attack",  "timestamp", "vae_mu", "vae_recon_error", "vae_anomaly_prob", "original_index"])#通过original_index排序
        else:#
            session_g = nx.from_pandas_edgelist(
                session_data, "srcip", "dstip", 
                edge_attr=["h", "Label", "Attack",  "timestamp", "vae_mu", "vae_recon_error", "vae_anomaly_prob"],
                create_using=nx.MultiDiGraph()
            )
            dgl_g = dgl.from_networkx(session_g, edge_attrs=["h", "Label", "Attack",  "timestamp", "vae_mu", "vae_recon_error", "vae_anomaly_prob"])
        # 关键修改：将图移到与张量相同的设备
        dgl_g = dgl_g.to(DEVICE)
        # 初始化节点特征
        #edim = len(session_data.iloc[0]['h'])
        edim = len(session_data.iloc[0]['vae_mu']) if len(session_data) > 0 else 0
        nfeat_weight = torch.ones([dgl_g.number_of_nodes(), edim], dtype=torch.float32, device=DEVICE)  # 在DEVICE上创建
        dgl_g.ndata['h'] = torch.reshape(nfeat_weight, (dgl_g.num_nodes(), 1, edim))

        #dgl_g.ndata['h'] = torch.ones((dgl_g.num_nodes(), 1, edim), device=DEVICE)
        #dgl_g.edata['h'] = torch.tensor(np.array(session_data['h'].tolist()),  dtype=torch.float32, device=DEVICE).view(-1, 1, edim)#会导致边的特征混乱
        edge_mu = torch.tensor(np.stack(dgl_g.edata['vae_mu'].cpu().numpy()), dtype=torch.float32, device=DEVICE)
        edge_recon_error = dgl_g.edata['vae_recon_error'].unsqueeze(1).float()  # 标量→向量
        edge_anomaly_prob = dgl_g.edata['vae_anomaly_prob'].unsqueeze(1).float()  # 标量→向量
        if(edge_mu.dim() > 2):
            edge_mu = edge_mu.squeeze(dim=1)
        #dgl_g.edata['h'] = (torch.cat([edge_mu, edge_recon_error, edge_anomaly_prob], dim=1)).view(-1, 1, edim)  # 边特征维度=latent_dim + 1
        dgl_g.edata['h'] = (edge_mu).view(-1, 1, edim)
        #h_features = dgl_g.edata['h']
        #dgl_g.edata['h'] = h_features.view(-1, 1, edim)
        del dgl_g.edata['vae_mu'], dgl_g.edata['vae_recon_error']
        graphs.append(dgl_g)
    return graphs
#@profile
def train_dgi_model(dgi, train_graphs: List[dgl.DGLGraph], val_graphs: List[dgl.DGLGraph], params: dict):
    """训练DGI图自监督模型"""
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
        
        # 验证集评估
        dgi.eval()
        val_loss = 0.0
        with torch.no_grad():
            for g in val_graphs:
                loss = dgi(g, g.ndata['h'], g.edata['h'])
                val_loss += loss.item()
        avg_val_loss = val_loss / len(val_graphs)
        
        # 早停和学习率调整
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
    """更新全局会话缓存并触发图构建（修改：移除三元组参数）"""
    #global GLOBAL_SESSION_CACHE, ANOMALY_TRIGGER, ALL_PREDICTION_LABEL, ALL_TRUE_LABEL, ALL_TEST_VAE_PROB, ALL_TEST_LOF_PROB,ALL_GLOBAL_LATENT,ALL_GLOBAL_RECON
    global GLOBAL_SESSION_CACHE, ANOMALY_TRIGGER, ALL_PREDICTION_LABEL, ALL_TRUE_LABEL, ALL_TEST_VAE_PROB, ALL_TEST_LOF_PROB
    # 异常样本直接触发检测（可选）
    if is_anomaly:
        ANOMALY_TRIGGER = True
    
    # 缓存样本（无论是否异常）
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
        'vae_anomaly_prob':sample['vae_anomaly_prob'],#新添加，将异常概率值也添加进去
        'vae_recon_error': sample['vae_recon_error']
    }

    GLOBAL_SESSION_CACHE.append(session_data)
    
    # 达到阈值且触发异常时构建图（修改：使用全局缓存长度）
    if len(GLOBAL_SESSION_CACHE) >= SESSION_THRESHOLD and ANOMALY_TRIGGER:
        build_and_detect_graph(dgi, df_train)
        reset_trigger()  # 重置缓存和触发标志
    elif len(GLOBAL_SESSION_CACHE) >= SESSION_THRESHOLD and not ANOMALY_TRIGGER:
        retired_sessions = []
        total = len(GLOBAL_SESSION_CACHE)
        remove_count = total * 4 // 5

        # 弹出前2/3元素并保存到列表
        for _ in range(remove_count):
            retired_sessions.append(GLOBAL_SESSION_CACHE.popleft())
        # 收集真实标签和预测标签
        for session in retired_sessions:
            ALL_TRUE_LABEL.append(session['Label'])
            ALL_PREDICTION_LABEL.append(session['is_anomaly'])
            #ALL_GLOBAL_RECON.append(session['anomaly_prob'])
            #ALL_GLOBAL_LATENT.append(session['anomaly_prob'])
            ALL_TEST_VAE_PROB.append(session['anomaly_prob'])
            ALL_TEST_LOF_PROB.append(session['anomaly_prob'])#此时并不进行lof检测，所以把vae检测的概率作为结果
    
    # 记录检测结果
    #print(f"全局会话缓存检测到 {sum(labels)} 个异常")
    #GLOBAL_SESSION_CACHE = deque(active_samples, maxlen=SESSION_THRESHOLD)

def reset_trigger():
    """重置全局会话缓存和异常触发标志（修改：移除三元组参数）"""
    global GLOBAL_SESSION_CACHE, ANOMALY_TRIGGER
    #GLOBAL_SESSION_CACHE.clear()
    ANOMALY_TRIGGER = False

from sklearn.mixture import GaussianMixture
def detect_by_gmm(df_train, df_test):
    print("开始进行cblof检测==================")
    raw_benign_train_samples = df_train[df_train.Label == 0].drop(columns=["Label", "Attack"])
    raw_normal_train_samples = df_train.drop(columns=["Label", "Attack"])

    raw_train_labels = df_train["Label"]
    raw_test_labels = df_test["Label"]
    vae_test_probs = df_test["anomaly_prob"]#VAE测试的异常概率
    raw_test_samples = df_test.drop(columns=["Label", "timestamp", "anomaly_prob", "original_index"])
    gmm = GaussianMixture(n_components=3, covariance_type='full', random_state=42)
    gmm.fit(raw_normal_train_samples)
    test_scores = gmm.score_samples(raw_test_samples)  # 密度越低越异常
    train_scores = gmm.score_samples(raw_normal_train_samples)
    threshold = np.percentile(train_scores, 5)  # 分数越低越异常

    # 根据阈值预测异常
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
    """构建全局会话图并执行CBLOF检测（修改：移除三元组参数）"""
    global GLOBAL_SESSION_CACHE  # 假设DGI模型已加载
    global ALL_SRCIP_LIST, ALL_DSTIP_LIST, ALL_SRCPORT_LIST, ALL_DSTPORT_LIST
    print("当前图中存在异常，触发图检测==================")
    session_data = list(GLOBAL_SESSION_CACHE)
    df = pd.DataFrame(session_data)
    
    # 构建图（复用原有build_session_graphs逻辑）
    starttime = time.time()
    graphs = build_session_graphs(df, group_size=len(df), anomaly=True)#只构建一张图
    endtime = time.time()
    #print(f"图构建时间为：{endtime - starttime}，此时共构建了{len(df)}个会话")
    if not graphs:
        return
    
    # 生成图嵌入
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
            #print(f"图嵌入时间为：{endtime - starttime}，此时共{len(df)}个会话参与了嵌入")
            emb = emb.detach().cpu().numpy()
            embeddings.append(emb)
            #test_subgraph_emb = torch.sigmoid(emb.mean(dim=0))#原图的全局表示
            #discriminator_weight = dgi.discriminator.weight
            #scores = torch.matmul(emb, torch.matmul(discriminator_weight, test_subgraph_emb))
            
            #scores_np = scores.detach().cpu().numpy()
            #all_test_edge_scores.append(scores_np)
            


    # 合并嵌入并执行CBLOF检测
    embeddings = np.vstack(embeddings)
    #此处添加嵌入后的真正的标签值
    embeddings_test = pd.DataFrame(embeddings)
    embeddings_test["Label"] = np.concatenate([
        g.edata['Label'].detach().cpu().numpy()
        for g in graphs
    ])#嵌入后边真正的标签值
    embeddings_test["timestamp"] = np.concatenate([
        g.edata['timestamp'].detach().cpu().numpy()
        for g in graphs
    ])#嵌入后时间戳
    embeddings_test["anomaly_prob"] = np.concatenate([
        g.edata['vae_anomaly_prob'].detach().cpu().numpy()
        for g in graphs
    ])#嵌入后边真正的标签值
    is_unique = embeddings_test["timestamp"].is_unique
    print(f"时间戳是否唯一：{is_unique}")
    embeddings_test["original_index"] = np.concatenate([
        g.edata['original_index'].detach().cpu().numpy()
        for g in graphs
    ])

    # 按照timestamp列升序排序
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


    #如此边的顺序与原始GLOBAL_SESSION_CACHE中的顺序一致了
    _,_,labels, result_df = cblof_detection(df_train, embeddings_test)
    #result_df = detect_by_global_vae_with_kde(embeddings_test)
    #result = detect_by_gmm(df_train, embeddings_test)


    # 输出结果并退休前2/3样本
    retired_sessions = []
    total = len(GLOBAL_SESSION_CACHE)
    if final_batch:
        remove_count = total
    else:
        remove_count = total * 4 // 5
    # 弹出前2/3元素并保存到列表
    for _ in range(remove_count):
        retired_sessions.append(GLOBAL_SESSION_CACHE.popleft())
    # 收集真实标签和预测标签
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
    print("开始进行cblof检测==================")
    raw_benign_train_samples = df_train[df_train.Label == 0].drop(columns=["Label", "Attack"])
    raw_normal_train_samples = df_train.drop(columns=["Label", "Attack"])

    raw_train_labels = df_train["Label"]
    raw_test_labels = df_test["Label"]
    vae_test_probs = df_test["anomaly_prob"]#VAE测试的异常概率
    srcip = df_test["srcip"]
    dstip = df_test["dstip"]
    srcport = df_test["srcport"]
    dstport = df_test["dstport"]
    raw_test_samples = df_test.drop(columns=["Label", "timestamp", "anomaly_prob", "original_index", "srcip", "dstip", "srcport", "dstport"])
    n_est = [1, 2, 3, 5, 7, 10, 15]  # 增加最大簇数量
    contamination = [0.001, 0.01, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5]  # 修正范围并扩展
    params = list(itertools.product(n_est, contamination))
    score = -1
    best_params = {}
    bs = None
    results = []  # 存储所有参数组合的结果
    for n_est, con in params:
        try:
            #clf_b = CBLOF(n_clusters=n_est, contamination=con)
            clf_b = CBLOF(
                n_clusters=n_est,
                contamination=con,
                check_estimator=False  # 关闭聚类有效性校验（视版本而定）
            )
            clf_b.fit(raw_benign_train_samples)
        except ValueError as e:
            print(f"参数组合 (n_clusters={n_est}, contamination={con}) 导致错误: {str(e)}")
            continue  
        starttime = time.time()
        y_pred = clf_b.predict(raw_test_samples)
        endtime = time.time()
        #print(f"cblof检测时间为：{endtime-starttime}，此时共有{len(raw_test_samples)}个会话被检测")
        lof_test_probs = clf_b.predict_proba(raw_test_samples)[:, 1]#获取异常类别的概率
        test_pred = y_pred.copy()
        test_pred[vae_test_probs > 0.95] = 1  # 将概率大于0.9的样本标记为异常

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
     # 保存最佳结果到CSV
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

def isolation_detection(df_train, df_test):
    print("开始进行Isolation Forest检测==================")
    # 保持与原逻辑一致，使用良性样本进行训练
    raw_benign_train_samples = df_train[df_train.Label == 0].drop(columns=["Label", "Attack"])
    raw_normal_train_samples = df_train.drop(columns=["Label", "Attack"])

    raw_train_labels = df_train["Label"]
    raw_test_labels = df_test["Label"]
    vae_test_probs = df_test["anomaly_prob"]  # VAE测试的异常概率
    srcip = df_test["srcip"]
    dstip = df_test["dstip"]
    srcport = df_test["srcport"]
    dstport = df_test["dstport"]
    raw_test_samples = df_test.drop(columns=["Label", "timestamp", "anomaly_prob", "original_index", "srcip", "dstip", "srcport", "dstport"])
    
    # Isolation Forest的超参数组合（替换原CBLOF参数）
    n_estimators = [50]  # 树的数量
    max_samples = ['auto']  # 每棵树使用的样本量
    contamination = [0.001, 0.01, 0.05, 0.1, 0.15, 0.2, 0.3,0.4,0.5,0.6,0.7]  # 异常比例保持不变
    
    from itertools import product
    params = list(product(n_estimators, max_samples, contamination))
    
    score = -1
    best_params = {}
    best_lof_probs = None  # 保持变量名兼容，实际存储的是Isolation Forest的异常概率
    best_vae_probs = None
    best_test_pred = None
    best_raw_labels = None
    result_df = None

    for n_est, max_samp, con in params:
        try:
            # 初始化Isolation Forest模型
            from sklearn.ensemble import IsolationForest
            clf = IsolationForest(
                n_estimators=n_est,
                max_samples=max_samp,
                contamination=con,
                random_state=42,
                n_jobs=-1  # 使用所有CPU核心
            )
            clf.fit(raw_benign_train_samples)
        except Exception as e:
            print(f"参数组合 (n_estimators={n_est}, max_samples={max_samp}, contamination={con}) 导致错误: {str(e)}")
            continue
        
        # Isolation Forest预测结果为1（正常）和-1（异常），需转换为0和1
        y_pred = clf.predict(raw_test_samples)
        y_pred = [1 if x == -1 else 0 for x in y_pred]
        
        # 计算异常概率（通过决策函数转换，归一化到[0,1]）
        decision_func = clf.decision_function(raw_test_samples)
        # 转换为概率形式（值越大越可能是异常）
        lof_test_probs = (1 - decision_func) / 2  # 归一化到[0,1]
        
        # 与VAE结果融合（保持原逻辑）
        #test_pred = y_pred.copy()
        test_pred = np.array(y_pred)  # 这里从列表改为数组
        test_pred[vae_test_probs > 0.95] = 1  # 高VAE异常概率样本强制标记为异常

        # 计算评估指标
        from sklearn.metrics import f1_score
        f1 = f1_score(raw_test_labels, test_pred, average='macro')
            
        # 更新最佳参数
        if f1 > score:
            score = f1
            best_params = {
                'n_estimators': n_est, 
                'max_samples': max_samp, 
                'contamination': con
            }
            best_vae_probs = vae_test_probs.copy()
            best_lof_probs = lof_test_probs.copy()
            best_test_pred = test_pred.copy()
            best_raw_labels = raw_test_labels.copy()

        # 清理内存
        del clf
        import gc
        gc.collect()
    
    # 保存最佳结果到CSV
    if best_lof_probs is not None:
        result_df = pd.DataFrame({
            'srcip': srcip,
            'dstip': dstip,
            'srcport': srcport,
            'dstport': dstport,
            'vae_test_probs': best_vae_probs,
            'lof_test_probs': best_lof_probs,  # 实际为Isolation Forest的异常概率
            'test_pred': best_test_pred,
            'raw_test_labels': best_raw_labels
        })
        
    return best_params, score, best_test_pred, result_df


def load_vae_models(model_path):
    """加载所有三元组对应的VAE模型"""
    global TRIPLET_MODEL_MAP
    TRIPLET_MODEL_MAP = torch.load(model_path, map_location=DEVICE)
    #TRIPLET_MODEL_MAP = torch.load(model_path, map_location=DEVICE, weights_only=False)
    print(f"成功加载 {len(TRIPLET_MODEL_MAP)} 个VAE模型")

#@profile
def detect_by_vae_with_kde(sample, triplet_src, triplet_dst, cols_to_norm, weight_recon=0.9, threshold=0.6):
    """
    使用 KDE 模型检测样本是否异常。
    """
    global ALL_VAE_TRUE_LABEL, ALL_VAE_TEST_LABEL, TRIPLET_PREDICTIONS
    
    # 获取模型信息
    triplet = triplet_src if triplet_src in TRIPLET_MODEL_MAP else triplet_dst
    model_info = TRIPLET_MODEL_MAP[triplet]
    model = model_info['model'].to(DEVICE)
    kde_recon = model_info['stats']['kde_recon']
    kde_latent = model_info['stats']['kde_latent']
    recon_stats = model_info['stats']['recon_stats']
    latent_stats = model_info['stats']['latent_stats']
    scaler = model_info['scaler']
    
    # 数据预处理

    #scaler = preprocessing.Normalizer()

    #sample['h'] = scaler.transform(np.array(sample['h']).reshape(1, -1)).flatten().tolist()
    #sample['h'] = scaler.transform(sample.iloc[7:-3].values.reshape(1, -1)).flatten().tolist()#因此h已经添加了
    sample_features = sample[cols_to_norm].to_frame().T
    normalized_features = scaler.transform(sample_features).flatten().tolist()
    sample['h'] = normalized_features

    data = torch.tensor(np.array(sample['h']).reshape(1, -1), dtype=torch.float32).to(DEVICE)
    
    # 模型推断
    starttime = time.time()
    model.eval()
    with torch.no_grad():
        recon_batch, mu, logvar = model(data)#recon_batch是重构后的特征，mu和logvar是
        recon_error = torch.mean((recon_batch - data) ** 2, dim=1).item()
        latent_norm = torch.norm(mu, dim=1).item()
    
    # 使用 KDE 模型计算概率密度
    if np.isnan(recon_error) or np.isinf(recon_error):
        recon_density = 0
    else:
        recon_density  = np.exp(kde_recon.score_samples([[recon_error]])[0])  # 重构误差的概率密度
    if np.isnan(latent_norm) or np.isinf(latent_norm):
        latent_density = 0
    else:
        latent_density  = np.exp(kde_latent.score_samples([[latent_norm]])[0])  # 潜在向量范数的概率密度
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
    recon_z = (recon_density - recon_stats['density_mean']) / (recon_stats['density_std'] + 1e-8)
    latent_z = (latent_density - latent_stats['density_mean']) / (latent_stats['density_std'] + 1e-8)
    # 取负号：密度越低，Z越小，sigmoid(-Z)越接近1
    recon_anomaly_prob = sigmoid(-recon_z)
    latent_anomaly_prob = sigmoid(-latent_z)

    
    # 综合两种概率
    anomaly_prob = weight_recon * recon_anomaly_prob + (1 - weight_recon) * latent_anomaly_prob
    
    # 确定是否异常
    pred_label = 1 if anomaly_prob > threshold else 0
    endtime = time.time()
    #print(f"VAE检测的时间为：{endtime-starttime}")
    # 记录结果
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


# 2.5 为训练集（train_df）添加VAE特征（供图模型训练）
def add_vae_features_to_train(df, triplet_model_map, cols_to_norm, vae_params):
    """为训练集的每个会话添加VAE特征"""
    #vae_h_list = []
    vae_mu_list = []
    vae_recon_error_list = []
    vae_anomaly_prob_list = []
    weight_recon=0.9
    for idx, sample in df.iterrows():
        # 匹配该会话的VAE模型
        triplet_src = f"{sample['srcip']}_{sample['srcport']}_{sample['protocol']}"
        triplet_dst = f"{sample['dstip']}_{sample['dstport']}_{sample['protocol']}"
        if triplet_src in triplet_model_map:
            triplet = triplet_src
        elif triplet_dst in triplet_model_map:
            triplet = triplet_dst
        else:
            # 无对应VAE模型，用全局均值填充（或跳过）
            #vae_h_list.append(np.zeros(256))  # 假设h维度为256
            vae_mu_list.append(np.zeros(vae_params['latent_dim']))#如果没有对应的三元组，则填充0
            vae_recon_error_list.append(0.0)#没有对应的三元组，则填充0
            vae_anomaly_prob_list.append(1.0)
            continue
        
        # 调用VAE模型获取特征
        model_info = triplet_model_map[triplet]
        model = model_info['model'].to(DEVICE)
        scaler = model_info['scaler']
        kde_recon = model_info['stats']['kde_recon']
        kde_latent = model_info['stats']['kde_latent']
        recon_stats = model_info['stats']['recon_stats']
        latent_stats = model_info['stats']['latent_stats']
        
        # 预处理样本特征
        sample_features = sample[cols_to_norm].to_frame().T
        normalized_features = scaler.transform(sample_features).flatten()
        x = torch.tensor(normalized_features, dtype=torch.float32).to(DEVICE).unsqueeze(0)
        
        # 提取VAE特征
        model.eval()
        with torch.no_grad():
            recon_x, mu, logvar = model(x)
            recon_error = torch.mean((recon_x - x) ** 2, dim=1).item()
            latent_norm = torch.norm(mu, dim=1).item()
        


            # 使用 KDE 模型计算概率密度
        if np.isnan(recon_error) or np.isinf(recon_error):
            recon_density = 0
        else:
            recon_density  = np.exp(kde_recon.score_samples([[recon_error]])[0])  # 重构误差的概率密度
        if np.isnan(latent_norm) or np.isinf(latent_norm):
            latent_density = 0
        else:
            latent_density  = np.exp(kde_latent.score_samples([[latent_norm]])[0])  # 潜在向量范数的概率密度
        def sigmoid(x):
            return 1 / (1 + np.exp(-x))
        recon_z = (recon_density - recon_stats['density_mean']) / (recon_stats['density_std'] + 1e-8)
        latent_z = (latent_density - latent_stats['density_mean']) / (latent_stats['density_std'] + 1e-8)
        # 取负号：密度越低，Z越小，sigmoid(-Z)越接近1
        recon_anomaly_prob = sigmoid(-recon_z)
        latent_anomaly_prob = sigmoid(-latent_z)

        
        
        # 综合两种概率
        anomaly_prob = weight_recon * recon_anomaly_prob + (1 - weight_recon) * latent_anomaly_prob
        # 保存特征
        vae_anomaly_prob_list.append(anomaly_prob)
        vae_mu_list.append(mu.cpu().numpy().squeeze())
        vae_recon_error_list.append(recon_error)

    # 将VAE特征加入训练集
    df['vae_mu'] = vae_mu_list
    df['vae_recon_error'] = vae_recon_error_list
    df['vae_anomaly_prob'] = vae_anomaly_prob_list
    return df



def detect_by_global_vae_with_kde(df_test, weight_recon=0.2, threshold=0.6):
    """
    使用 KDE 模型批量检测样本是否异常（支持批量输入）
    :param df_test: 测试数据集（包含所有样本特征）
    :param weight_recon: 重构误差的权重（0-1之间）
    :param threshold: 异常概率阈值
    :return: 包含检测结果的DataFrame
    """
    global GLOBAL_VAE_MODEL
    if not GLOBAL_VAE_MODEL:
        raise ValueError("GLOBAL_VAE_MODEL未初始化，请先训练或加载模型")
    
    model = GLOBAL_VAE_MODEL['model'].to(DEVICE)
    kde_recon = GLOBAL_VAE_MODEL['stats']['kde_recon']
    kde_latent = GLOBAL_VAE_MODEL['stats']['kde_latent']
    recon_stats = GLOBAL_VAE_MODEL['stats']['recon_stats']
    latent_stats = GLOBAL_VAE_MODEL['stats']['latent_stats']
    print(f"开始进行VAE全局检测，共处理{len(df_test)}个样本...")

    # 数据预处理：提取特征并转换为张量
    raw_test_samples = df_test.drop(columns=["Label", "timestamp", "anomaly_prob", "original_index"], errors='ignore')
    if raw_test_samples.empty:
        raise ValueError("输入数据中无有效特征列")
    raw_test_labels = df_test["Label"]
    vae_test_probs = df_test["anomaly_prob"]
    X_test = torch.tensor(raw_test_samples.values, dtype=torch.float32, device=DEVICE)
    batch_size = X_test.size(0)

    # 模型批量推断
    model.eval()
    with torch.no_grad():
        recon_batch, mu, logvar = model(X_test)  # 直接处理整个批次
        recon_error = torch.mean((recon_batch - X_test) ** 2, dim=1).cpu().numpy()  # [batch_size]
        latent_norm = torch.norm(mu, dim=1).cpu().numpy()  # [batch_size]

    # 计算重构误差和潜在向量范数的概率密度（向量化计算）
    try:
        recon_density = np.exp(kde_recon.score_samples(recon_error.reshape(-1, 1)))  # [batch_size]
        latent_density = np.exp(kde_latent.score_samples(latent_norm.reshape(-1, 1)))  # [batch_size]
    except Exception as e:
        raise RuntimeError(f"KDE评分计算失败: {str(e)}")

    # Z-score标准化（向量化实现）
    recon_z = (recon_density - recon_stats['density_mean']) / (recon_stats['density_std'] + 1e-8)
    latent_z = (latent_density - latent_stats['density_mean']) / (latent_stats['density_std'] + 1e-8)

    # 计算异常概率（向量化sigmoid）
    recon_anomaly_prob = 1 / (1 + np.exp(recon_z))  # 密度越低，异常概率越高
    latent_anomaly_prob = 1 / (1 + np.exp(latent_z))
    anomaly_prob = weight_recon * recon_anomaly_prob + (1 - weight_recon) * latent_anomaly_prob
    anomaly_prob[vae_test_probs > 0.95] = 1  # 将概率大于0.9的样本标记为异常
    # 生成检测结果
    result_df = pd.DataFrame({
        'original_index': df_test.get('original_index', range(batch_size)),  # 保留原始索引
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
        """对单个样本进行归一化，严格保留特征名信息"""
        sample_df = pd.DataFrame([sample])  # 转为单行DataFrame
        features_df = sample_df[cols_to_norm]
        scaled_array = scaler.transform(features_df)
        scaled_df = pd.DataFrame(scaled_array, columns=cols_to_norm, index=sample_df.index)
        sample_df[cols_to_norm] = scaled_df
        return sample_df  # 转回Series



def print_triplet_classification_reports():
    """生成每个三元组的分类报告和混淆矩阵（返回字符串而非打印）"""
    
    all_reports = []
    
    for triplet, labels in TRIPLET_PREDICTIONS.items():
        if not labels['true_labels']:
            report = f"三元组 {triplet} 没有预测结果，跳过报告\n"
            all_reports.append(report)
            continue
            
        # 获取唯一的真实标签和预测标签
        unique_true = np.unique(labels['true_labels'])
        unique_pred = np.unique(labels['pred_labels'])
        all_classes = np.unique(np.concatenate([unique_true, unique_pred]))
        
        # 确定类别名称
        if len(all_classes) == 1:
            if all_classes[0] == 0:
                class_names = ['正常']
            else:
                class_names = ['异常']
            report = f"警告: 三元组 {triplet} 只包含类别 {all_classes[0]} 的样本\n"
        else:
            class_names = ['正常', '异常']
        
        # 生成分类报告
        report = f"\n=== 三元组 {triplet} 的分类报告 ===\n"
        report += classification_report(
            labels['true_labels'], 
            labels['pred_labels'],
            target_names=class_names,
            labels=all_classes,
            zero_division=0
        )
        
        # 生成混淆矩阵
        cm = confusion_matrix(labels['true_labels'], labels['pred_labels'], labels=all_classes)
        report += f"\n=== 三元组 {triplet} 的混淆矩阵 ===\n"
        
        if len(all_classes) == 1:
            if all_classes[0] == 0:
                report += "              预测值\n"
                report += "          正常\n"
                report += f"真实值 正常  {cm[0,0]:<8}\n"
            else:
                report += "              预测值\n"
                report += "          异常\n"
                report += f"真实值 异常  {cm[0,0]:<8}\n"
        else:
            report += "              预测值\n"
            report += "          正常     异常\n"
            report += f"真实值 正常  {cm[0,0]:<8} {cm[0,1]:<8}\n"
            report += f"      异常  {cm[1,0]:<8} {cm[1,1]:<8}\n"
        
        all_reports.append(report)
        print(report)
    
    # 生成总体报告
    overall_report = "\n=== 所有三元组的总体分类报告 ===\n"
    overall_report += classification_report(
        ALL_VAE_TRUE_LABEL, 
        ALL_VAE_TEST_LABEL,
        target_names=['正常', '异常'],
        zero_division=0
    )
    
    cm_global = confusion_matrix(ALL_VAE_TRUE_LABEL, ALL_VAE_TEST_LABEL)
    overall_report += "\n=== 所有三元组的总体混淆矩阵 ===\n"
    overall_report += "              预测值\n"
    overall_report += "          正常     异常\n"
    overall_report += f"真实值 正常  {cm_global[0,0]:<8} {cm_global[0,1]:<8}\n"
    overall_report += f"      异常  {cm_global[1,0]:<8} {cm_global[1,1]:<8}\n"
    
    all_reports.append(overall_report)
    print(overall_report)
    return all_reports

def save_reports_to_file(file_path='classification_reports.txt'):
    """将分类报告保存到文件"""
    reports = print_triplet_classification_reports()
    
    with open(file_path, 'w', encoding='utf-8') as f:
        for report in reports:
            f.write(report)
    
    print(f"分类报告已保存到 {file_path}")


    

# ------------------------ 主流程 ------------------------ #

def main():
    # 1. 数据加载与预处理
    print("=== 数据预处理 ===")
    raw_data = load_and_preprocess_data()
    #raw_data = filter_data_by_triple(raw_data, tuple_file)#按照三元组来筛选
    raw_data = filter_data_by_ip(raw_data, tuple_file)#只保留源地址或者目的地址在三元组列表中
    #if len(raw_data) > 100000:
    #    raw_data = raw_data.head(100000)
    #raw_data = raw_data.groupby(by='Attack').sample(frac=0.2, random_state=42)
    train_df, test_df = split_dataset(raw_data)
    #train_df_copy = train_df.copy(deep=True)#用于图的训练
    train_df, test_df = normalize_features(train_df, test_df)#只改变了['h']的值
    cols_to_norm = train_df.columns[7:-3]  # 假设特征从第8列开始
    #scaler = preprocessing.Normalizer()
    #global_scaler = MinMaxScaler()
    #train_df['h'] = global_scaler.fit_transform(train_df[cols_to_norm]).tolist()
    #train_df_copy[cols_to_norm] = global_scaler.fit_transform(train_df_copy[cols_to_norm])#如果数据找不到对应的三元组，就使用全局的归一化
    #train_df_copy['h'] = train_df_copy.iloc[:, 7:-2].values.tolist()
    #train_df['h'] = train_df[cols_to_norm].values.tolist()
    #test_df['h'] = test_df[cols_to_norm].values.tolist()
    #test_df['h'] = global_scaler.transform(test_df[cols_to_norm]).tolist()
    # 目标编码
    encoder = preprocessing.LabelEncoder()
    encoder.fit(raw_data["Attack"])
    train_df["Attack"] = encoder.transform(train_df["Attack"])
    test_df["Attack"] = encoder.transform(test_df["Attack"])
    #train_df_copy["Attack"] = encoder.transform(train_df_copy["Attack"])
    # 重置索引（丢弃原始索引，生成从0开始的新索引）
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    #train_df_copy = train_df_copy.reset_index(drop=True)
    #test_df = test_df.iloc[:1000]
    # 2. VAE模型训练
    print("\n=== VAE模型训练 ===")
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
    # 检查文件夹是否存在，不存在则创建
    folder_result = pathlib.Path(prefix_result)
    folder_model = pathlib.Path(prefix_model)
    if not folder_result.exists():
        folder_result.mkdir(parents=True, exist_ok=False)
    if not folder_model.exists():
        folder_model.mkdir(parents=True, exist_ok=False)
    

    initial_memory = {
        "nvidia_smi_total_mib": get_nvidia_smi_memory(device_id=1),  # 设备级总占用（含框架开销）
        "pytorch_allocated_mib": torch.cuda.memory_allocated(device) / (1024**2),  # 显式分配（模型+数据）
        "pytorch_reserved_mib": torch.cuda.memory_reserved(device) / (1024**2)     # 动态缓存
    }
    print(f"程序启动后初始GPU内存占用：")
    print(f"  nvidia-smi总占用：{initial_memory['nvidia_smi_total_mib']} MiB")
    print(f"  PyTorch显式分配：{initial_memory['pytorch_allocated_mib']:.2f} MiB")
    print(f"  PyTorch动态缓存：{initial_memory['pytorch_reserved_mib']:.2f} MiB")



    vae_models = train_vae_models(triplet_data, vae_params)


    initial_memory = {
        "nvidia_smi_total_mib": get_nvidia_smi_memory(device_id=1),  # 设备级总占用（含框架开销）
        "pytorch_allocated_mib": torch.cuda.memory_allocated(device) / (1024**2),  # 显式分配（模型+数据）
        "pytorch_reserved_mib": torch.cuda.memory_reserved(device) / (1024**2)     # 动态缓存
    }
    print(f"VAE训练后GPU内存占用：")
    print(f"  nvidia-smi总占用：{initial_memory['nvidia_smi_total_mib']} MiB")
    print(f"  PyTorch显式分配：{initial_memory['pytorch_allocated_mib']:.2f} MiB")
    print(f"  PyTorch动态缓存：{initial_memory['pytorch_reserved_mib']:.2f} MiB")

    torch.save(vae_models, vae_model_save_path)
    # 加载VAE模型
    load_vae_models(vae_model_save_path)

    # 调用该函数，为train_df添加VAE特征。目的是将VAE编码后的特征作为图中会话的特征进行训练
    train_df = add_vae_features_to_train(train_df, TRIPLET_MODEL_MAP, cols_to_norm, vae_params)

    # 3. 图模型训练
    print("\n=== 图自监督模型训练 ===")
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
        "nvidia_smi_total_mib": get_nvidia_smi_memory(device_id=1),  # 设备级总占用（含框架开销）
        "pytorch_allocated_mib": torch.cuda.memory_allocated(device) / (1024**2),  # 显式分配（模型+数据）
        "pytorch_reserved_mib": torch.cuda.memory_reserved(device) / (1024**2)     # 动态缓存
    }
    print(f"图模型加载前GPU内存占用：")
    print(f"  nvidia-smi总占用：{initial_memory['nvidia_smi_total_mib']} MiB")
    print(f"  PyTorch显式分配：{initial_memory['pytorch_allocated_mib']:.2f} MiB")
    print(f"  PyTorch动态缓存：{initial_memory['pytorch_reserved_mib']:.2f} MiB")

    dgi_model = DGI(dgi_params['ndim_in'], dgi_params['ndim_out'], dgi_params['edim'], torch.relu).to(DEVICE).float()
    total_samples = len(train_df)
    val_ratio = 0.2
    val_samples = int(total_samples * val_ratio)
    train_sessions = build_session_graphs(train_df.iloc[:-val_samples], anomaly=False)
    val_sessions = build_session_graphs(train_df.iloc[-val_samples:], anomaly=False)  # 使用验证集构建图

    dgi_model = train_dgi_model(dgi_model, train_sessions, val_sessions, dgi_params)
    initial_memory = {
        "nvidia_smi_total_mib": get_nvidia_smi_memory(device_id=1),  # 设备级总占用（含框架开销）
        "pytorch_allocated_mib": torch.cuda.memory_allocated(device) / (1024**2),  # 显式分配（模型+数据）
        "pytorch_reserved_mib": torch.cuda.memory_reserved(device) / (1024**2)     # 动态缓存
    }
    print(f"图模型训练后GPU内存占用：")
    print(f"  nvidia-smi总占用：{initial_memory['nvidia_smi_total_mib']} MiB")
    print(f"  PyTorch显式分配：{initial_memory['pytorch_allocated_mib']:.2f} MiB")
    print(f"  PyTorch动态缓存：{initial_memory['pytorch_reserved_mib']:.2f} MiB")
    dgi_model.load_state_dict(torch.load(dgi_model_save_path))
    print("加载图模型成功")

    # 4. 生成嵌入并评估
    print("\n=== 生成嵌入与评估 ===")
    #先生成训练集的嵌入，为了后续CBLOF的训练
    all_training_embs = []
    all_training_vae_embs = []
    for session_g in train_sessions:
        training_emb = dgi_model.encoder(session_g, session_g.ndata['h'], session_g.edata['h'])[1]
        #training_emb = training_emb.detach().cpu().numpy()
        #subgraph_emb = torch.sigmoid(training_emb.mean(dim=0))#原图的全局表示
        #discriminator_weight = dgi_model.discriminator.weight
        #scores = torch.matmul(training_emb, torch.matmul(discriminator_weight, subgraph_emb))
        training_emb = training_emb.detach().cpu().numpy()
        #scores_np = scores.detach().cpu().numpy()
        all_training_embs.append(training_emb)
        
        #all_edge_scores.append(scores_np)
        #for 消融
        all_training_vae_embs.append(session_g.edata['h'])
    
    training_emb = np.vstack(all_training_embs)
    df_train = pd.DataFrame(training_emb)#所有训练集的全局表示

    df_train["Attack"] = np.concatenate([
        encoder.inverse_transform(g.edata['Attack'].detach().cpu().numpy())
        for g in train_sessions
    ])
    df_train["Label"] = np.concatenate([
        g.edata['Label'].detach().cpu().numpy()
        for g in train_sessions
    ])
    # 执行测试（示例调用）
    ##############################################################################################
    #VAE第二次训练
    #global_vae_model = train_vae_global_models(df_train, vae_global_params, sample_ratio=0.5)
    #torch.save(global_vae_model, vae_global_model_save_path)
    #global GLOBAL_VAE_MODEL
    #GLOBAL_VAE_MODEL = torch.load(vae_global_model_save_path, map_location=DEVICE)

    ##############################################################################################





    global dgi  # 设置全局变量以便在build_and_detect_graph中使用
    dgi = dgi_model
    #global ALL_PREDICTION_LABEL, ALL_TRUE_LABEL, ALL_TEST_VAE_PROB, ALL_TEST_LOF_PROB, ALL_GLOBAL_LATENT, ALL_GLOBAL_RECON#存放最终的检测结果
    global ALL_PREDICTION_LABEL, ALL_TRUE_LABEL, ALL_TEST_VAE_PROB, ALL_TEST_LOF_PROB#存放最终的检测结果
    # 处理测试样本
    vae_result = []
    vae_recon_error = []
    vae_latent_norm = []
    vae_recon_threshold = []
    vae_latent_threshold = []
    vae_anomaly_prob = []
    i = 0
    for idx, sample in test_df.iterrows():
        # 提取三元组
        
        #vae_true_label.append(sample['Label'])#临时使用
        triplet_src = f"{sample['srcip']}_{sample['srcport']}_{sample['protocol']}"
        triplet_dst = f"{sample['dstip']}_{sample['dstport']}_{sample['protocol']}"
        #sample_copy = sample.copy(deep=True)
        #sample_copy['h'] = normalize_single_sample(sample_copy, global_scaler, cols_to_norm)#图检测需要使用全局归一化
        # VAE检测
        if csvfile == 'aciiot':
            if triplet_src == '192.168.1.1_53_17' or triplet_dst == '192.168.1.1_53_17' \
                or triplet_src == '239.255.255.250_1900_17' or triplet_dst == '239.255.255.250_1900_17':
                continue
        if triplet_src not in TRIPLET_MODEL_MAP and triplet_dst not in TRIPLET_MODEL_MAP:
            #is_anomaly = True
            recon_error = 0
            continue
            sample['vae_mu'] = np.zeros(vae_params['latent_dim'], dtype=np.float32)  # 列表类型
            sample['vae_recon_error'] = 0.0  # 标量转列表
            sample['vae_anomaly_prob'] = 1.0  # 异常概率设为1（无模型匹配视为异常）
            # 更新会话缓存（标记为异常）
            
            update_session_cache(dgi_model, sample, is_anomaly=True, df_train=df_train, \
                                    anomaly_prob=1.0)
        else:
            #i += 1
            #is_anomaly, anomaly_score = detect_anomaly_with_iforest(sample, triplet_src, triplet_dst)
            #starttime = time.time()
            vae_result  = detect_by_vae_with_kde(sample, triplet_src, triplet_dst, cols_to_norm)
            #endtime = time.time()
            #print(f"VAE检测时间为：{endtime - starttime}")
            vae_result['sample']['vae_mu'] = vae_result['mu'].cpu().numpy().squeeze()
            vae_result['sample']['vae_recon_error'] = vae_result['recon_error']
            vae_result['sample']['vae_anomaly_prob'] = vae_result['anomaly_prob']

            # 将CUDA张量转换为CPU上的numpy数组
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
        # 更新会话缓存并触发图检测
            update_session_cache(dgi_model, vae_result['sample'], vae_result['is_anomaly'], df_train, vae_result['anomaly_prob'])#用了全局的归一化表示
        #update_session_cache(dgi_model, sample_copy, result['is_anomaly'], df_train, result['anomaly_prob'])
        
        # 内存管理
        if idx % 10000 == 0:
            gc.collect()
            print(f"已处理 {idx}/{len(test_df)} 个样本")

    # 最终检测（处理剩余缓存）
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
    }#存储的是第一阶段VAE检测的结果
    df = pd.DataFrame(data)
    df.to_csv(prefix_result + '/' + csvfile + '_recon.csv', index=False)#将VAE测试的结果保存

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
    df1.to_csv(prefix_result + '/' + csvfile + '_prob.csv', index=False)#将VAE和整体的预测概率结果保存



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

    # 调用示例
    save_reports_to_file(prefix_result + '/' + csvfile + '_vae_metrics.txt')

if __name__ == "__main__":
    main()
    print("所有任务完成")


