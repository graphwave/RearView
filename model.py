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
      #self.W_edge = nn.Linear(128 * 2+edims, 256)
      self.W_edge = nn.Linear(ndim_out+edims, 64)
      #self.W_edge = nn.Linear(ndim_out, 256)#
      self.reset_parameters()

    def reset_parameters(self):
      gain = nn.init.calculate_gain('relu')
      nn.init.xavier_uniform_(self.W_node.weight, gain=gain)
      nn.init.xavier_uniform_(self.W_edge.weight, gain=gain)

    def message_func(self, edges):
      return {'m':  edges.data['h']}

    def forward(self, g_dgl, nfeats, efeats):
      with g_dgl.local_scope():
        g = g_dgl
        g.ndata['h'] = nfeats
        g.edata['h'] = efeats
        #g.update_all(self.message_func, fn.mean('m', 'h_neigh'))

        g.update_all(self.message_func, fn.mean('m', 'h_out_neigh'))
        
        #g.ndata['h'] = F.relu(self.W_apply(torch.cat([g.ndata['h'], g.ndata['h_neigh']], 2)))#
        


        g_in = dgl.reverse(g, copy_edata=True) 
        g_in.update_all(
            message_func=lambda edges: {'m_in': edges.data['h']},
            reduce_func=fn.mean('m_in', 'h_in_neigh')
        )

        g.ndata['h_in_neigh'] = g_in.ndata['h_in_neigh']
        g.ndata['h'] = self.activation(self.W_node(
                torch.cat([g.ndata['h_out_neigh'], g.ndata['h_in_neigh']], dim=2)
                #torch.cat([g.ndata['h'], g.ndata['h_in_neigh']], dim=2)
                #g.ndata['h_in_neigh']
            ))
        # Compute edge embeddings
        u, v = g.edges()
        #edge = self.W_edge(torch.cat((g.srcdata['h'][u], g.dstdata['h'][v]), 2))
        #edge = self.W_edge(torch.cat((g.srcdata['h'][u], g.dstdata['h'][v], efeats), 2))
        edge_features = torch.cat(( g.ndata['h'][u], efeats), dim=2) 

        edge = self.W_edge(edge_features)
        return g.ndata['h'], edge



class SAGELayer(nn.Module):
    def __init__(self, ndim_in, edims, ndim_out, activation):
        super(SAGELayer, self).__init__()
        self.W_node = nn.Linear(ndim_in, ndim_out) 
        self.activation = activation
        self.W_edge = nn.Linear(ndim_out + edims, 64)  
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_uniform_(self.W_node.weight, gain=gain)
        nn.init.xavier_uniform_(self.W_edge.weight, gain=gain)


    def message_func(self, edges):

        anomaly_weight = edges.data['vae_anomaly_prob']

        anomaly_weight = anomaly_weight.unsqueeze(1)  

        anomaly_weight = anomaly_weight.unsqueeze(-1)
        weighted_edge_feat = edges.data['h'] * anomaly_weight 
        return {
            'weighted_m': weighted_edge_feat,
            'weight': anomaly_weight 
        }


    def reduce_func(self, nodes):

        weighted_sum = nodes.mailbox['weighted_m'].sum(dim=1)  # shape: [num_nodes,1,edim]

        total_weight = nodes.mailbox['weight'].sum(dim=1)      # shape: [num_nodes,1]


        weighted_avg = weighted_sum / (total_weight + 1e-8) 
        return {'agg_feat': weighted_avg}  


    def forward(self, g_dgl, nfeats, efeats):
        with g_dgl.local_scope():
            g = g_dgl
            g.ndata['h'] = nfeats 
            g.edata['h'] = efeats 




            g.update_all(self.message_func, self.reduce_func)
            g.ndata['src_agg'] = g.ndata['agg_feat'] 


            g_in = dgl.reverse(g, copy_edata=True)  
            g_in.update_all(self.message_func, self.reduce_func)
            g.ndata['dst_agg'] = g_in.ndata['agg_feat'] 


            #node_agg = torch.cat([g.ndata['src_agg'], g.ndata['dst_agg']], dim=2)
            node_agg = g.ndata['dst_agg']
            g.ndata['final_agg'] = self.activation(self.W_node(node_agg))


            u, v = g.edges()
            edge_features = torch.cat([
                g.ndata['final_agg'][u], 
                g.edata['h']             
            ], dim=2)
            final_edge_feat = self.W_edge(edge_features)

            return g.ndata['final_agg'], final_edge_feat


class SAGE(nn.Module):
    def __init__(self, ndim_in, ndim_out, edim,  activation):
      super(SAGE, self).__init__()
      self.layers = nn.ModuleList()
      self.layers.append(SAGELayer(ndim_in, edim, ndim_out, F.relu))

    def forward(self, g, nfeats, efeats, corrupt=False):
      if corrupt: 
        e_perm = torch.randperm(g.number_of_edges())
        #n_perm = torch.randperm(g.number_of_nodes())
        efeats = efeats[e_perm]
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
        self.activation = nn.Sigmoid() 

    def uniform(self, size, tensor):
        bound = 1.0 / math.sqrt(size)
        if tensor is not None:
            tensor.data.uniform_(-bound, bound)

    def reset_parameters(self):
        size = self.weight.size(0)
        self.uniform(size, self.weight)

    def forward(self, features, summary):

        batch_size = features.shape[0]
        scores = torch.zeros(batch_size, device=features.device)
        for i in range(batch_size):

            scores[i] = torch.matmul(features[i], torch.matmul(self.weight, summary[i]))

        return scores 


class DGI(nn.Module):
    def __init__(self, ndim_in, ndim_out, edim, activation):
      super(DGI, self).__init__()
      self.encoder = SAGE(ndim_in, ndim_out, edim,  F.relu)
      #self.discriminator = Discriminator(128)
      self.discriminator = Discriminator1(64)
      self.loss = nn.BCEWithLogitsLoss()
      self.linear_layer = nn.Linear(ndim_in, 64)  

    def forward(self, g, n_features, e_features):
      positive = self.encoder(g, n_features, e_features, corrupt=False) #[15, 128]和[226, 256]
      negative = self.encoder(g, n_features, e_features, corrupt=True) #[63442, 256]

      positive = positive[1]
      negative = negative[1]

      summary = torch.sigmoid(positive.mean(dim=0))

      #global_weights = torch.softmax(positive, dim=0) 
      #summary = (global_weights * positive).sum(dim=0)  

      #transformed_e_features = self.linear_layer(e_features)
      #summary = torch.sigmoid(transformed_e_features)
      #summary = summary.squeeze(1)
      positive_loss = self.discriminator(positive, summary)#positive-[226,256] e_features-[226, 1, 39], summary-[256]
      negative_loss = self.discriminator(negative, summary)
      
      pos_sim = F.cosine_similarity(positive, summary.unsqueeze(0)).mean()
      neg_sim = F.cosine_similarity(negative, summary.unsqueeze(0)).mean()

      
      l1 = self.loss(positive_loss, torch.ones_like(positive_loss)) 
      l2 = self.loss(negative_loss, torch.zeros_like(negative_loss)) 

      return l1 + l2
    









  
