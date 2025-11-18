import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import os
import pandas as pd
import ipaddress
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, f1_score, recall_score, confusion_matrix
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")


import torch.nn.functional as F
import subprocess
import re
def get_nvidia_smi_memory(device_id=0):

    try:

        result = subprocess.run(
            ["nvidia-smi", f"--id={device_id}", "--query-gpu=memory.used", "--format=csv,nounits,noheader"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        output = result.stdout.strip()
        if output:
            return int(re.findall(r"\d+", output)[0])  
        return 0
    except Exception as e:
        return 0
class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(VAE, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16)
        )
        

        self.attention = nn.Linear(16, 16)
        

        self.fc_mu = nn.Linear(16, latent_dim)
        self.fc_logvar = nn.Linear(16, latent_dim)
        

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )

    def encode(self, x):
        h = self.encoder(x)

        attn_weights = torch.sigmoid(self.attention(h))
        h = h * attn_weights

        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar
'''
class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(VAE, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.attention = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, latent_dim * 2)
        self.fc3 = nn.Linear(latent_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, input_dim)
        self.latent_dim = latent_dim
        

        self.min_logvar = -10 
        self.max_logvar = 10  

    def encode(self, x):
        h1 = torch.relu(self.fc1(x))
        attn_weights = torch.sigmoid(self.attention(h1))
        h1 = h1 * attn_weights
        return self.fc2(h1)

    def reparameterize(self, mu, logvar):

        logvar = torch.clamp(logvar, self.min_logvar, self.max_logvar)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h3 = torch.relu(self.fc3(z))

        return torch.sigmoid(self.fc4(h3)) 

    def forward(self, x):
        mu_logvar = self.encode(x).chunk(2, dim=1)
        mu, logvar = mu_logvar
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar
'''
def vae_loss(recon_x, x, mu, logvar, beta=1.0, sparsity_weight=0.001):

    MSE = nn.functional.mse_loss(recon_x, x, reduction='sum')
    

    logvar = torch.clamp(logvar, -10, 10)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    

    sparsity_penalty = torch.sum(torch.abs(mu))
    

    return MSE + beta * KLD + sparsity_weight * sparsity_penalty

def train(model, train_loader, optimizer, device):
    model.train()
    train_loss = 0
    for data, _ in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = vae_loss(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
    return train_loss / len(train_loader.dataset)

def load_data(file_path):
    try:
        data = np.loadtxt(file_path, delimiter=',', skiprows=1)
        return torch.tensor(data, dtype=torch.float32)
    except Exception as e:
        print(f"Error loading data: {e}")

def normalize_data(train_x, test_x):

    scaler = MinMaxScaler()
    

    train_x[:, 16, 4:110] = np.nan_to_num(train_x[:, 16, 4:110], nan=0.0)
    train_data = scaler.fit_transform(train_x[:, 16, 4:110])
    #print(train_data[0].max())
    

    test_x[:, 16, 4:110] = np.nan_to_num(test_x[:, 16, 4:110], nan=0.0)
    test_data = scaler.transform(test_x[:, 16, 4:110])
    #print(test_data[0:-1].max())
    

    train_x[:, 16, 4:110] = train_data
    test_x[:, 16, 4:110] = test_data
    
    return train_x, test_x

def calculate_threshold(model, data_loader, device):

    model.eval()
    all_recon_errors = []
    all_latent_norms = []
    
    with torch.no_grad():
        for data, _ in data_loader:
            data = data.to(device)
            recon_batch, mu, logvar = model(data)
            

            recon_error = torch.mean((recon_batch - data) ** 2, dim=1)
            all_recon_errors.extend(recon_error.cpu().numpy())
            

            latent_norm = torch.norm(mu, dim=1)
            all_latent_norms.extend(latent_norm.cpu().numpy())
    

    return {
        'recon': {'mean': np.mean(all_recon_errors), 'std': np.std(all_recon_errors)},
        'latent': {'mean': np.mean(all_latent_norms), 'std': np.std(all_latent_norms)}
    }

from sklearn.neighbors import KernelDensity
def calculate_threshold_with_kde(model, data_loader, device, bandwidth_recon=0.1, bandwidth_latent=0.1):

    model.eval()
    all_recon_errors = []
    all_latent_norms = []
    
    with torch.no_grad():
        for data, _ in data_loader:
            data = data.to(device)
            recon_batch, mu, logvar = model(data)
            

            recon_error = torch.mean((recon_batch - data) ** 2, dim=1)
            all_recon_errors.extend(recon_error.cpu().numpy())
            

            latent_norm = torch.norm(mu, dim=1)
            all_latent_norms.extend(latent_norm.cpu().numpy())
    

    kde_recon = KernelDensity(kernel='gaussian', bandwidth=bandwidth_recon)
    kde_latent = KernelDensity(kernel='gaussian', bandwidth=bandwidth_latent)
    all_recon_errors = np.array(all_recon_errors)
    all_latent_norms = np.array(all_latent_norms)

    kde_recon.fit(np.array(all_recon_errors).reshape(-1, 1)) 
    kde_latent.fit(np.array(all_latent_norms).reshape(-1, 1))

    recon_densities = np.exp(kde_recon.score_samples(all_recon_errors.reshape(-1, 1)))
    latent_densities = np.exp(kde_latent.score_samples(all_latent_norms.reshape(-1, 1)))
    
    return {
        'kde_recon': kde_recon,
        'kde_latent': kde_latent,
        'recon_stats': {
            'mean': np.mean(all_recon_errors),
            'std': np.std(all_recon_errors),
            'min': np.min(all_recon_errors),
            'max': np.max(all_recon_errors),
            'density_mean': np.mean(recon_densities),
            'density_std': np.std(recon_densities)
        },
        'latent_stats': {
            'mean': np.mean(all_latent_norms),
            'std': np.std(all_latent_norms),
            'min': np.min(all_latent_norms),
            'max': np.max(all_latent_norms),
            'density_mean': np.mean(latent_densities),
            'density_std': np.std(latent_densities)
        }
    }

'''
def calculate_threshold(model, data_loader, device):
    model.eval()
    all_recon_errors = []
    all_latent_norms = []
    with torch.no_grad():
        for data, _ in data_loader:
            data = data.to(device)
            recon_batch, mu, logvar = model(data)
            #recon_batch = model(data)

            recon_error = torch.mean((recon_batch - data) ** 2, dim=1)
            all_recon_errors.extend(recon_error.cpu().numpy())
            

            latent_norm = torch.norm(mu, dim=1)
            all_latent_norms.extend(latent_norm.cpu().numpy())
    

    recon_threshold = np.quantile(all_recon_errors, 0.95)
    latent_threshold = np.quantile(all_latent_norms, 0.95)
    return recon_threshold,latent_threshold
'''

def detect_anomalies(model, data_loader, device, threshold):
    model.to(device) 
    model.eval()
    predictions = []
    labels = []
    with torch.no_grad():
        for data, target in data_loader:
            data = data.to(device)
            recon_batch, mu, logvar = model(data)
            recon_error = torch.mean((recon_batch - data) ** 2, dim=1)
            pred_labels = (recon_error > threshold).int()
            predictions.extend(pred_labels.cpu().numpy())
            #labels.extend(target.cpu().numpy())
            labels.extend([int(t.item()) for t in target])
    return labels, predictions

def find_csv_files(directory):
    csv_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.csv'):
                csv_files.append(os.path.join(root, file))
    return csv_files

def find_npz_files(directory):
    npz_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.npz'):
                npz_files.append(os.path.join(root, file))
    return npz_files

def read_and_append_columns(csv_files, label):
    all_data = pd.DataFrame() 
    if not isinstance(csv_files, list):
        csv_files = [csv_files]
    for file in csv_files:

        df = pd.read_csv(file)

        filename = os.path.splitext(os.path.basename(file))[0]

        df['attack_cat'] = filename
        df['label'] = label

        all_data = pd.concat([all_data, df], ignore_index=True)
    all_data.fillna(0, inplace=True)
    return all_data

def read_npz_files(npz_files):
    #all_data = pd.DataFrame() 
    all_X = []
    all_Y = []
    if not isinstance(npz_files, list):
        npz_files = [npz_files]
    for file in npz_files:

        df = np.load(file)
        print(file)
        
        X = df['X']
        Y = df['Y']
        X = X[:len(Y)]
        #print(len(X))
        #print(len(Y))
        
        all_X.append(X)
        all_Y.append(Y)
    return all_X, all_Y


def extract_ip_port_proto(category):

    parts = category.split('_')

    server_ip = int(ipaddress.ip_address(parts[0]))
    server_port = int(parts[1])
    proto = int(parts[2])

    #server_ip_int = np.dot([int(x) for x in server_ip.split('.')], [256**3, 256**2, 256**1, 256**0])
    server_ip_float = float(server_ip) 

    return server_ip_float, float(server_port), float(proto)


def match_and_select_data(category, all_X, all_Y):

    

    server_ip, server_port, server_proto = extract_ip_port_proto(category)
    

    ip_src = all_X[:, 16, 0] 
    ip_dst = all_X[:, 16, 1] 
    srcport = all_X[:, 16, 2]
    dstport = all_X[:, 16, 3]  
    protocol = all_X[:, 16, 4] 


    mask = ((ip_src == server_ip) & (srcport == server_port) & (protocol == server_proto)) | \
           ((ip_dst == server_ip) & (dstport == server_port) & (protocol == server_proto))
    

    matched_X = all_X[mask]
    matched_Y = all_Y[mask]
    
    return matched_X, matched_Y




def train_process(VAE_train_X, VAE_train_Y, parameters):

    selected_data = VAE_train_X
    train_data = selected_data.reshape(selected_data.shape[0], -1)
    train_y = VAE_train_Y.astype(int)
    

    train_x, val_x, train_y, val_y = train_test_split(
        train_data, train_y, test_size=0.2, random_state=13
    )
    

    train_x = torch.tensor(train_x, dtype=torch.float32)
    train_y = torch.tensor(train_y, dtype=torch.float32)
    val_x = torch.tensor(val_x, dtype=torch.float32)
    val_y = torch.tensor(val_y, dtype=torch.float32)
    

    train_dataset = TensorDataset(train_x, train_y)
    val_dataset = TensorDataset(val_x, val_y)
    train_loader = DataLoader(train_dataset, batch_size=parameters['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=parameters['batch_size'], shuffle=False)
    

    
    initial_memory = {
        "nvidia_smi_total_mib": get_nvidia_smi_memory(device_id=1), 
        "pytorch_allocated_mib": torch.cuda.memory_allocated(device) / (1024**2), 
        "pytorch_reserved_mib": torch.cuda.memory_reserved(device) / (1024**2)     
    }


    input_dim = train_x.shape[1]
    model = VAE(input_dim, parameters['hidden_dim'], parameters['latent_dim']).to(device)
    optimizer = optim.Adam(model.parameters(), lr=parameters['learning_rate'])
    
    initial_memory = {
        "nvidia_smi_total_mib": get_nvidia_smi_memory(device_id=1),  
        "pytorch_allocated_mib": torch.cuda.memory_allocated(device) / (1024**2),
        "pytorch_reserved_mib": torch.cuda.memory_reserved(device) / (1024**2)   
    }


    




    best_val_loss = float('inf')
    best_model_state = None     




    for epoch in range(1, parameters['epochs'] + 1):


        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(device)
            epoch_start_allocated = torch.cuda.memory_allocated(device)
        train_loss = train(model, train_loader, optimizer, device)

        val_loss = validate(model, val_loader, device)
        
        print(f'Epoch {epoch}, Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()
    

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    stats = calculate_threshold_with_kde(model, train_loader, device) 
    
    return {
        'model': model,
        'stats': stats  
    }

def validate(model, val_loader, device):

    model.eval() 
    total_loss = 0
    with torch.no_grad():
        for data, _ in val_loader:
            data = data.to(device)
            recon_data,mu,lovgr = model(data)
            loss = vae_loss(recon_data, data,mu, lovgr)  
            total_loss += loss.item()
    return total_loss / len(val_loader)


def test_process(model, threshold, test_loader, device):
    labels, predictions = detect_anomalies(model, test_loader, device, threshold)
    results_df = pd.DataFrame({
        'Labels': labels,
        'Predictions': predictions
    })
    
    precision = precision_score(labels, predictions, zero_division=0)
    f1 = f1_score(labels, predictions, zero_division=0)
    recall = recall_score(labels, predictions, zero_division=0)
    print(f'Precision: {precision:.4f}')
    print(f'F1 Score: {f1:.4f}')
    print(f'Recall Score: {recall:.4f}')
    all_classes = [0, 1]
    conf_matrix = confusion_matrix(labels, predictions)
    if conf_matrix.shape[0] > 1 and conf_matrix.shape[1] > 1:
        TP = conf_matrix[1, 1]
        FP = conf_matrix[0, 1]
        FN = conf_matrix[1, 0]
        TN = conf_matrix[0, 0] 
    else:
        TN = conf_matrix[0,0]
        TP = 0
        FP = 0
        FN = 0
    print(f'True Positives (TP): {TP}')
    print(f'False Positives (FP): {FP}')
    print(f'True Negative (TN): {TN}')
    print(f'False Negative (FN): {FN}')
    return results_df





