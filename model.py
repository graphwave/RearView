import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn
import dgl
import math


class SAGELayer2(nn.Module):
    def __init__(self, ndim_in, edims, ndim_out, activation):
      super(SAGELayer, self).__init__()
      #self.W_apply = nn.Linear(ndim_in + edims , ndim_out)
      #self.W_node = nn.Linear(ndim_in + edims * 2, ndim_out)
      self.W_node = nn.Linear(ndim_in*2, ndim_out)
      self.activation = F.relu
      #self.W_edge = nn.Linear(128 * 2, 256)
      #self.W_edge = nn.Linear(128 * 2+edims, 256)#modified by fp
      self.W_edge = nn.Linear(ndim_out+edims, 64)#modified by fp
      #self.W_edge = nn.Linear(ndim_out, 256)#20250527
      self.reset_parameters()

    def reset_parameters(self):
      gain = nn.init.calculate_gain('relu')
      nn.init.xavier_uniform_(self.W_node.weight, gain=gain)
      nn.init.xavier_uniform_(self.W_edge.weight, gain=gain)

    def message_func(self, edges):
      return {'m':  edges.data['h']}#消息传递函数，传递的实际上是边的特征

    def forward(self, g_dgl, nfeats, efeats):
      with g_dgl.local_scope():
        g = g_dgl
        g.ndata['h'] = nfeats
        g.edata['h'] = efeats
        #g.update_all(self.message_func, fn.mean('m', 'h_neigh'))
        # #DGL（Deep Graph Library）中核心的消息传递操作，用于实现图神经网络中的节点表示更新，保存结果至g.ndata['h_neigh']
        g.update_all(self.message_func, fn.mean('m', 'h_out_neigh'))
        
        #g.ndata['h'] = F.relu(self.W_apply(torch.cat([g.ndata['h'], g.ndata['h_neigh']], 2)))#将节点自身与传递的边的特征进行拼接，更新节点表示
        


        g_in = dgl.reverse(g, copy_edata=True)  # 创建反向图，但不需要copy_node_data
        g_in.update_all(
            message_func=lambda edges: {'m_in': edges.data['h']},
            reduce_func=fn.mean('m_in', 'h_in_neigh')
        )
        # 将入边聚合结果复制回原图
        g.ndata['h_in_neigh'] = g_in.ndata['h_in_neigh']
        g.ndata['h'] = self.activation(self.W_node(
                torch.cat([g.ndata['h_out_neigh'], g.ndata['h_in_neigh']], dim=2)
                #torch.cat([g.ndata['h'], g.ndata['h_in_neigh']], dim=2)
                #g.ndata['h_in_neigh']
            ))
        # Compute edge embeddings
        u, v = g.edges()#u中存储的是源节点的索引，v中存储的是目标节点的索引
        #edge = self.W_edge(torch.cat((g.srcdata['h'][u], g.dstdata['h'][v]), 2))#g.srcdata['h']存储的是所有源节点的特征，返回[63442, 1, 256]
        #edge = self.W_edge(torch.cat((g.srcdata['h'][u], g.dstdata['h'][v], efeats), 2))#modified by fp
        edge_features = torch.cat(( g.ndata['h'][u], efeats), dim=2)  # 源节点的聚合特征+边的特征
        #edge_features = g.ndata['h'][u]   #20250527
            # 应用线性变换得到最终边特征
        edge = self.W_edge(edge_features)
        return g.ndata['h'], edge



class SAGELayer(nn.Module):
    def __init__(self, ndim_in, edims, ndim_out, activation):
        super(SAGELayer, self).__init__()
        self.W_node = nn.Linear(ndim_in, ndim_out)  # 源聚合+目标聚合
        self.activation = activation
        self.W_edge = nn.Linear(ndim_out + edims, 64)  # 双向节点聚合+边特征
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_uniform_(self.W_node.weight, gain=gain)
        nn.init.xavier_uniform_(self.W_edge.weight, gain=gain)

    ###########################################################################
    # 步骤1：消息生成（边特征 × 异常得分权重）
    ###########################################################################
    def message_func(self, edges):
        # 获取边的异常得分（shape: [num_edges, 1]，已扩展维度）
        anomaly_weight = edges.data['vae_anomaly_prob']
        # 加权边特征 = 边特征 × 异常得分（异常得分越高，特征贡献越大）
        anomaly_weight = anomaly_weight.unsqueeze(1)  
        # 第二步：在最后补1维 → [104509, 1, 1]（3维，完全适配边特征维度）
        anomaly_weight = anomaly_weight.unsqueeze(-1)
        weighted_edge_feat = edges.data['h'] * anomaly_weight  # 维度匹配：[num_edges,1,edim]
        return {
            'weighted_m': weighted_edge_feat,  # 加权后的边特征
            'weight': anomaly_weight           # 单条边的权重（用于后续总权重计算）
        }

    ###########################################################################
    # 步骤2：加权求和 → 加权平均（核心修改：明确“求和后除以总权重”）
    ###########################################################################
    def reduce_func(self, nodes):
        # 1. 加权求和：对所有入边的“加权边特征”求和（突出异常边）
        weighted_sum = nodes.mailbox['weighted_m'].sum(dim=1)  # shape: [num_nodes,1,edim]
        # 2. 计算总权重：所有入边的异常得分之和（用于后续平均）
        total_weight = nodes.mailbox['weight'].sum(dim=1)      # shape: [num_nodes,1]
        # 3. 加权平均：求和后除以总权重（避免边数多的节点特征值过大）
        # 加1e-8防止总权重为0（所有边异常得分为0时，按0处理）
        weighted_avg = weighted_sum / (total_weight + 1e-8)  # 维度匹配：[num_nodes,1,edim]
        return {'agg_feat': weighted_avg}  # 最终的节点聚合特征（加权平均结果）

    ###########################################################################
    # 步骤3：双向聚合（源节点出边 + 目标节点入边）+ 生成新边特征
    ###########################################################################
    def forward(self, g_dgl, nfeats, efeats):
        with g_dgl.local_scope():
            g = g_dgl
            g.ndata['h'] = nfeats  # 节点初始特征（如IP统计特征）
            g.edata['h'] = efeats  # 原始边特征（VAE latent向量）
            # 确保异常得分已传入边属性（从main.txt的build_session_graphs获取）
            assert 'vae_anomaly_prob' in g.edata, "边属性中缺少VAE异常得分'vae_anomaly_prob'"

            # 1. 源节点的“出边加权平均聚合”（捕捉源IP的异常出会话）
            g.update_all(self.message_func, self.reduce_func)
            g.ndata['src_agg'] = g.ndata['agg_feat']  # 源节点聚合特征

            # 2. 目标节点的“入边加权平均聚合”（捕捉目标IP的异常入会话）
            g_in = dgl.reverse(g, copy_edata=True)  # 反向图：入边变给出边
            g_in.update_all(self.message_func, self.reduce_func)
            g.ndata['dst_agg'] = g_in.ndata['agg_feat']  # 目标节点聚合特征

            # 3. 更新节点特征（融合源+目标聚合结果）
            #node_agg = torch.cat([g.ndata['src_agg'], g.ndata['dst_agg']], dim=2)
            node_agg = g.ndata['dst_agg']
            g.ndata['final_agg'] = self.activation(self.W_node(node_agg))

            # 4. 生成新边特征（源聚合+目标聚合+原始边特征）
            u, v = g.edges()
            edge_features = torch.cat([
                g.ndata['final_agg'][u],  # 源节点聚合特征
                g.edata['h']              # 原始边特征（保留细节）
            ], dim=2)
            final_edge_feat = self.W_edge(edge_features)

            return g.ndata['final_agg'], final_edge_feat


class SAGE(nn.Module):
    def __init__(self, ndim_in, ndim_out, edim,  activation):
      super(SAGE, self).__init__()
      self.layers = nn.ModuleList()
      self.layers.append(SAGELayer(ndim_in, edim, ndim_out, F.relu))#只添加了一层，所以只聚合一跳邻居信息

    def forward(self, g, nfeats, efeats, corrupt=False):
      if corrupt: #试图扰动策略
        e_perm = torch.randperm(g.number_of_edges()) #生成[0, n-1]之间的随机索引
        #n_perm = torch.randperm(g.number_of_nodes())
        efeats = efeats[e_perm] #边与节点关系打乱了
        #nfeats = nfeats[n_perm]
      for i, layer in enumerate(self.layers):
        #nfeats = layer(g, nfeats, efeats)
        nfeats, e_feats = layer(g, nfeats, efeats)
      #return nfeats.sum(1)
      return nfeats.sum(1), e_feats.sum(1)

class Discriminator1(nn.Module):
    def __init__(self, n_hidden):
      super(Discriminator1, self).__init__()
      self.weight = nn.Parameter(torch.Tensor(n_hidden, n_hidden))
      self.reset_parameters()

    def uniform(self, size, tensor):
      bound = 1.0 / math.sqrt(size)
      if tensor is not None:
        tensor.data.uniform_(-bound, bound)

    def reset_parameters(self):
      size = self.weight.size(0)
      self.uniform(size, self.weight)

    def forward(self, features, summary):
      features = torch.matmul(features, torch.matmul(self.weight, summary))
      return features

class Discriminator(nn.Module):
    def __init__(self, n_hidden):
        super(Discriminator, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(n_hidden, n_hidden))
        self.reset_parameters()
        self.activation = nn.Sigmoid()  # 添加激活函数

    def uniform(self, size, tensor):
        bound = 1.0 / math.sqrt(size)
        if tensor is not None:
            tensor.data.uniform_(-bound, bound)

    def reset_parameters(self):
        size = self.weight.size(0)
        self.uniform(size, self.weight)

    def forward(self, features, summary):
        # 确保维度匹配
        batch_size = features.shape[0]
        scores = torch.zeros(batch_size, device=features.device)
        for i in range(batch_size):
            # 对每个样本，计算其与对应summary行的双线性得分
            scores[i] = torch.matmul(features[i], torch.matmul(self.weight, summary[i]))

        return scores 


class DGI(nn.Module):
    def __init__(self, ndim_in, ndim_out, edim, activation):
      super(DGI, self).__init__()
      self.encoder = SAGE(ndim_in, ndim_out, edim,  F.relu)
      #self.discriminator = Discriminator(128)
      self.discriminator = Discriminator1(64)
      self.loss = nn.BCEWithLogitsLoss()
      self.linear_layer = nn.Linear(ndim_in, 64)  # 添加线性层

    def forward(self, g, n_features, e_features):
      positive = self.encoder(g, n_features, e_features, corrupt=False) #[15, 128]和[226, 256]
      negative = self.encoder(g, n_features, e_features, corrupt=True) #[63442, 256]

      positive = positive[1]#获取边的嵌入
      negative = negative[1]#获取负样本边的嵌入

      summary = torch.sigmoid(positive.mean(dim=0))#原图的全局表示

      #global_weights = torch.softmax(positive, dim=0)  # 根据正样本嵌入计算权重
      #summary = (global_weights * positive).sum(dim=0)  # 加权全局嵌入

      #transformed_e_features = self.linear_layer(e_features)
      #summary = torch.sigmoid(transformed_e_features)
      #summary = summary.squeeze(1)
      positive_loss = self.discriminator(positive, summary)#positive-[226,256] e_features-[226, 1, 39], summary-[256]
      negative_loss = self.discriminator(negative, summary)
      
      pos_sim = F.cosine_similarity(positive, summary.unsqueeze(0)).mean()
      neg_sim = F.cosine_similarity(negative, summary.unsqueeze(0)).mean()
      print(f"Positive与summary相似度: {pos_sim.item():.4f}, Negative相似度: {neg_sim.item():.4f}")
      
      l1 = self.loss(positive_loss, torch.ones_like(positive_loss)) #正样本对预测得分比较高
      l2 = self.loss(negative_loss, torch.zeros_like(negative_loss)) #负样本对预测得分比较低

      return l1 + l2
    









  