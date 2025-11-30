import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import softmax as scatter_softmax
from torch_geometric.nn import HeteroConv, GCNConv, global_mean_pool, global_add_pool, GATConv, GraphConv, MessagePassing, global_max_pool, GATv2Conv
import math
from torch_geometric.nn.dense.linear import Linear

from torch_geometric.utils import (
    add_self_loops,
    is_torch_sparse_tensor,
    remove_self_loops,
    softmax,
)

class cnn_extractor(nn.Module):
    def __init__(self, feature_kernel, input_size):
        super(cnn_extractor, self).__init__()
        self.convs = nn.ModuleList(
            [nn.Conv1d(input_size, feature_num, kernel)
             for kernel, feature_num in feature_kernel.items()])

    def forward(self, input_data):
        share_input_data = input_data.permute(0, 2, 1)
        feature = [conv(share_input_data) for conv in self.convs]
        feature = [torch.max_pool1d(f, f.shape[-1]) for f in feature]
        feature = torch.cat(feature, dim=1)
        feature = feature.view([-1, feature.shape[1]])
        return feature

class GraphEdgeWeight(nn.Module):
    def __init__(self, args, hidden_dim, list_of_edge_types):
        super(GraphEdgeWeight, self).__init__()
        self.hidden_dim = hidden_dim
        self.list_of_edge_types = list_of_edge_types
        
        self.parent_encoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(p=args.dropout))
        
        self.child_encoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(p=args.dropout))
        
        self.edge_weight_mlp = nn.Sequential(
            nn.Linear(3 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid())
    
    def get_node_features_by_name(self, graph, name):
        return graph.x_dict.get(name, None)

    def forward(self, graph):
        for edge_type, edge_index in graph.edge_index_dict.items():
            
            if edge_type not in self.list_of_edge_types:
                continue
            
            src_type, _, dst_type = edge_type
            
            par_fea = self.get_node_features_by_name(graph, src_type)[edge_index[0]]
            cld_fea = self.get_node_features_by_name(graph, dst_type)[edge_index[1]]
            enc_parent_fea = self.parent_encoder(par_fea)
            enc_child_fea = self.child_encoder(cld_fea)
            
            diff_fea = torch.abs(par_fea - cld_fea)
            
            edge_features = torch.cat([enc_parent_fea, enc_child_fea, diff_fea], dim=-1)
            graph[edge_type].edge_weight = self.edge_weight_mlp(edge_features).squeeze(-1)
        
        return graph


class GraphEdgeAttr(nn.Module):
    def __init__(self, args, hidden_dim, list_of_edge_types):
        super(GraphEdgeAttr, self).__init__()
        self.hidden_dim = hidden_dim
        self.list_of_edge_types = list_of_edge_types
        
        self.parent_encoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(p=args.dropout))
        
        self.child_encoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(p=args.dropout))
        
        self.edge_weight_mlp = nn.Sequential(
            nn.Linear(3 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim))
    
    def get_node_features_by_name(self, graph, name):
        return graph.x_dict.get(name, None)

    def forward(self, graph):
        for edge_type, edge_index in graph.edge_index_dict.items():
            
            if edge_type not in self.list_of_edge_types:
                continue
            
            src_type, _, dst_type = edge_type
            
            par_fea = self.get_node_features_by_name(graph, src_type)[edge_index[0]]
            cld_fea = self.get_node_features_by_name(graph, dst_type)[edge_index[1]]
            enc_parent_fea = self.parent_encoder(par_fea)
            enc_child_fea = self.child_encoder(cld_fea)
            
            diff_fea = torch.abs(par_fea - cld_fea)
            
            edge_features = torch.cat([enc_parent_fea, enc_child_fea, diff_fea], dim=-1)
            graph[edge_type].edge_attr = self.edge_weight_mlp(edge_features)
        
        return graph


class HeteroGNNLayer(nn.Module):
    def __init__(self, args, feature_dim, list_of_edge_types, conv_type=GraphConv, norm=False):
        super(HeteroGNNLayer, self).__init__()
        
        self.conv_type = None
        node_type = set()
        conv_dic = {}
        for edge_type in list_of_edge_types:
            node_type.add(edge_type[0])
            node_type.add(edge_type[2])
            if conv_type in [GATv2Conv]:
                conv_dic[edge_type] = conv_type(feature_dim, feature_dim // args.num_heads, heads=args.num_heads, add_self_loops=False, edge_dim=feature_dim)
                self.conv_type = "GATv2Conv"
            elif conv_type in [GraphConv]:
                conv_dic[edge_type] = conv_type(feature_dim, feature_dim)
                self.conv_type = "GraphConv"
        
        self.hetero_gnn_layer = HeteroConv(conv_dic)
        self.norm = norm
        if self.norm:
            self.hetero_norm = nn.ModuleDict({k: nn.LayerNorm(feature_dim) for k in list(node_type)})
        self.dropout = args.gnn_dropout
        if self.conv_type in ["GATv2Conv"]:
            self.graph_edge_attr_updater = GraphEdgeAttr(args, hidden_dim=feature_dim, list_of_edge_types=list_of_edge_types)
        elif self.conv_type in ["GraphConv"]:
            self.graph_edge_weight_updater = GraphEdgeWeight(args, hidden_dim=feature_dim, list_of_edge_types=list_of_edge_types)
    
    def forward(self, batch):
        if self.conv_type in ["GATv2Conv"]:
            batch = self.graph_edge_attr_updater(batch)
        elif self.conv_type in ["GraphConv"]:
            batch = self.graph_edge_weight_updater(batch)
        
        x_dict = batch.x_dict
        edge_index_dict = batch.edge_index_dict
        
        if self.conv_type in ["GATv2Conv"]:
            edge_attr_dict = batch.edge_attr_dict
            x_dict = self.hetero_gnn_layer(x_dict, edge_index_dict, edge_attr_dict=edge_attr_dict)
        elif self.conv_type in ["GraphConv"]:
            edge_weight_dict = batch.edge_weight_dict
            x_dict = self.hetero_gnn_layer(x_dict, edge_index_dict, edge_weight_dict=edge_weight_dict)
        
        if self.norm:
            for node_type in x_dict.keys():
                x_dict[node_type] = self.hetero_norm[node_type](x_dict[node_type])
        
        if self.dropout > 0:
            for node_type in x_dict.keys():
                x_dict[node_type] = F.dropout(x_dict[node_type], p=self.dropout, training=self.training)
        
        batch.x_dict = x_dict
        
        return batch


class Global_aware_Aggregator(nn.Module):
    def __init__(self, args, hidden_dim, node_type):
        super(Global_aware_Aggregator, self).__init__()        
        self.virtual_root = nn.Parameter(torch.randn(1, hidden_dim))
        self.node_type = list(set(node_type))
        self.weight_calculater = nn.ModuleList([nn.Linear(hidden_dim, 1) for _ in self.node_type])
        self.alpha = args.agg_alpha
        
    def forward(self, graph, news_embeddings):
        for node in self.node_type:
            node_emb = graph.x_dict[node]
            node_logit = self.weight_calculater[self.node_type.index(node)](node_emb)
            node_weight = scatter_softmax(node_logit, graph[node].batch)
            merged_news_embeddings = news_embeddings + self.virtual_root
            graph.x_dict[node] = node_emb + node_weight * merged_news_embeddings[graph[node].batch] * self.alpha
        
        return graph

def robust_mean_pool(x, batch):
    device = x.device
    unique_batches = torch.unique(batch, sorted=True)
    num_graphs = len(unique_batches)
    
    pooled = torch.zeros(num_graphs, x.size(1), device=device, dtype=x.dtype)
    
    for i, b in enumerate(unique_batches):
        mask = (batch == b)
        pooled[i] = x[mask].mean(dim=0)
    
    return pooled
    
class HeteroGNNPooling(nn.Module):
    def __init__(self, hidden_dim, node_type, pooling_type='mean', agg_output_dim=None):
        super(HeteroGNNPooling, self).__init__()
        self.node_type = list(set(node_type))
        self.pooling_type = pooling_type
        self.pooling = global_mean_pool if pooling_type == 'mean' else global_add_pool
        if agg_output_dim is None:
            agg_output_dim = hidden_dim
        self.agg_mean_mlp = MLP(hidden_dim * len(list(set(node_type))), hidden_dim, agg_output_dim, 3)
    


    def forward(self, graph, news_embeddings=None):
        pooled_features = []
        for node in graph.x_dict.keys():
            if node not in self.node_type:
                continue
            # BUG
            pooled_feature = self.pooling(graph.x_dict[node], graph[node].batch)
            # try:
            #     batch = graph[node].batch.long()
            #     max_batch = batch.max().item()
            #     pooled_feature = global_mean_pool(graph.x_dict[node], batch, size=max_batch + 1)
            # except RuntimeError as e:
            #     print(f"Pooling error: {e}")
            #     pooled_feature = robust_mean_pool(graph.x_dict[node], graph[node].batch)

            pooled_features.append(pooled_feature)
        
        for node in graph.x_dict.keys():
            if node in self.node_type:
                continue
            pooled_feature = self.pooling(graph.x_dict[node], graph[node].batch) * 0.
            pooled_features[-1] += pooled_feature
        
        if news_embeddings is not None:
            return self.agg_mean_mlp(torch.cat(pooled_features, dim=-1)) + news_embeddings
        else:
            return self.agg_mean_mlp(torch.cat(pooled_features, dim=-1))


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

class GraphNodeEdgeAttr(nn.Module):
    def __init__(self, args, hidden_dim, list_of_node_types, list_of_edge_types):
        super(GraphNodeEdgeAttr, self).__init__()
        self.hidden_dim = hidden_dim
        self.list_of_node_types = list_of_node_types
        self.list_of_edge_types = list_of_edge_types
        n_ntype = len(list_of_node_types)
        n_etype = len(list_of_edge_types)

        self.node_type_embedding = nn.Embedding(n_ntype, hidden_dim)
        self.edge_type_embedding = nn.Embedding(n_etype, hidden_dim)

        self.merge_node_edge_attr = nn.Sequential(nn.Linear(2*hidden_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))

    def forward(self, graph):
        for edge_type, edge_index in graph.edge_index_dict.items():
            if edge_type not in self.list_of_edge_types:
                continue

            src_type, _, dst_type = edge_type
            par_node_emb = self.node_type_embedding.weight[self.list_of_node_types.index(src_type)][None].repeat_interleave(edge_index.size(1), dim=0)
            cld_node_emb = self.node_type_embedding.weight[self.list_of_node_types.index(dst_type)][None].repeat_interleave(edge_index.size(1), dim=0)
            edge_emb = self.edge_type_embedding.weight[self.list_of_edge_types.index(edge_type)][None].repeat_interleave(edge_index.size(1), dim=0)

            graph[edge_type].edge_attr = [
                self.merge_node_edge_attr(torch.cat([par_node_emb, edge_emb], dim=1)),
                self.merge_node_edge_attr(torch.cat([cld_node_emb, edge_emb], dim=1))
            ]
        
        return graph

class GraphNodeTextualEdgeAttr(nn.Module):
    def __init__(self, args, hidden_dim, list_of_node_types, list_of_edge_types):
        super(GraphNodeTextualEdgeAttr, self).__init__()
        self.hidden_dim = hidden_dim
        self.list_of_node_types = list_of_node_types
        self.list_of_edge_types = list_of_edge_types
        n_ntype = len(list_of_node_types)
        n_etype = len(list_of_edge_types)

        self.node_type_embedding = nn.Embedding(n_ntype, hidden_dim)
        self.edge_type_embedding = nn.Embedding(n_etype, hidden_dim)

        self.merge_node_edge_attr = nn.Sequential(nn.Linear(2*hidden_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))

    def forward(self, graph):
        # print(f"graph.edge_index_dict.items(): {graph.edge_index_dict.items()}")
        for edge_type, edge_index in graph.edge_index_dict.items():
            if edge_type not in self.list_of_edge_types:
                continue
            if edge_type == ('sentence', 'to', 'env') or edge_type == ('env', 'to', 'sentence'):
                continue
            
            src_type, _, dst_type = edge_type
            par_node_emb = self.node_type_embedding.weight[self.list_of_node_types.index(src_type)][None].repeat_interleave(edge_index.size(1), dim=0)
            cld_node_emb = self.node_type_embedding.weight[self.list_of_node_types.index(dst_type)][None].repeat_interleave(edge_index.size(1), dim=0)
            edge_emb = self.edge_type_embedding.weight[self.list_of_edge_types.index(edge_type)][None].repeat_interleave(edge_index.size(1), dim=0)

            graph[edge_type].edge_attr = [
                self.merge_node_edge_attr(torch.cat([par_node_emb, edge_emb], dim=1)),
                self.merge_node_edge_attr(torch.cat([cld_node_emb, edge_emb], dim=1))
            ]
        
        return graph

class UnifiedGraphConv(MessagePassing):
    def __init__(self, args, emb_dim, list_of_edge_types, head_count=8, aggr="add", add_self_loops=True):
        super(UnifiedGraphConv, self).__init__(aggr=aggr)
        self.args = args
        self.emb_dim = emb_dim
        self.head_count = head_count
        self.add_self_loops = add_self_loops
        self.list_of_edge_types = list_of_edge_types

        self.res = Linear(emb_dim, emb_dim, bias=False, weight_initializer='glorot')

        # attention
        assert emb_dim % head_count == 0
        self.dim_per_head = emb_dim // head_count
        self.linear_key = nn.Linear(emb_dim, head_count * self.dim_per_head)
        self.linear_msg = nn.Linear(emb_dim, head_count * self.dim_per_head)
        self.linear_query = nn.Linear(emb_dim, head_count * self.dim_per_head)

        # final MLP
        self.mlp = nn.Sequential(nn.Linear(emb_dim, emb_dim), nn.BatchNorm1d(emb_dim), nn.LeakyReLU(), nn.Linear(emb_dim, emb_dim))

    def forward(self, x_dict, edge_index_dict, edge_attr_dict):
        for edge_type in self.list_of_edge_types:

            src, _, dst = edge_type
            x_l, x_r = x_dict[src], x_dict[dst]

            edge_index = edge_index_dict[edge_type]
            edge_attr = edge_attr_dict[edge_type]
            
            res = self.res(x_r)
            # x_r_agg = x_r[edge_index[1].unique()]
            # res = self.res(x_r_agg)

            # if x_r.shape != res.shape:
            #     print(f"x_l.shape: {x_l.shape}, x_r.shape: {x_r.shape}")
            #     print(f"res.shape: {res.shape}")

            alpha, msg = self.edge_updater(edge_index, x=(x_l, x_r), edge_attr=edge_attr)
            aggr_out = self.propagate(edge_index, x=(x_l, x_r), alpha=alpha, msg=msg, size=(x_l.size(0), x_r.size(0)))
            if aggr_out.shape != res.shape:
                print(f"x_l.shape: {x_l.shape}, x_r.shape: {x_r.shape}")
                print(f"aggr_out.shape: {aggr_out.shape}, res.shape: {res.shape}")

            out = self.mlp(aggr_out + res)
            # out = self.mlp(aggr_out)

            x_dict[dst] = out
        
        return x_dict

    def message(self, alpha, msg):
        out = msg * alpha.view(-1, self.head_count, 1)
        return out.view(-1, self.head_count * self.dim_per_head)
    
    def edge_update(self, x_j, x_i, edge_attr, edge_index): #i: tgt, j:src
        query = self.linear_query(x_j + edge_attr[0]).view(-1, self.head_count, self.dim_per_head)
        key   = self.linear_key(x_i + edge_attr[1]).view(-1, self.head_count, self.dim_per_head)
        msg = self.linear_msg(x_j).view(-1, self.head_count, self.dim_per_head)
        
        query = query / math.sqrt(self.dim_per_head)
        scores = (query * key).sum(dim=2)
        src_node_index = edge_index[0]
        alpha = softmax(scores, src_node_index)

        return alpha, msg


class UnifiedGNNLayer(nn.Module):
    def __init__(self, args, feature_dim, list_of_edge_types):
        super(UnifiedGNNLayer, self).__init__()

        node_type = set()
        for edge_type in list_of_edge_types:
            node_type.add(edge_type[0])
            node_type.add(edge_type[2])
        list_of_node_types = list(node_type)

        self.graph_node_edge_attr = GraphNodeEdgeAttr(args, hidden_dim=feature_dim, list_of_node_types=list_of_node_types, list_of_edge_types=list_of_edge_types)
        self.gnn_layer = UnifiedGraphConv(args, feature_dim, list_of_edge_types=list_of_edge_types)

        self.dropout = args.gnn_dropout
    
    def forward(self, batch):
        batch = self.graph_node_edge_attr(batch)
        
        x_dict = batch.x_dict
        edge_index_dict = batch.edge_index_dict
        edge_attr_dict = batch.edge_attr_dict

        x_dict = self.gnn_layer(x_dict, edge_index_dict, edge_attr_dict)
        
        if self.dropout > 0:
            for node_type in x_dict.keys():
                x_dict[node_type] = F.dropout(x_dict[node_type], p=self.dropout, training=self.training)
        
        batch.x_dict = x_dict
        # batch.edge_index_dict = edge_index_dict
        # batch.edge_attr_dict = edge_attr_dict
        
        return batch

class UnifiedTextualGNNLayer(nn.Module):
    def __init__(self, args, feature_dim, list_of_edge_types):
        super(UnifiedTextualGNNLayer, self).__init__()

        node_type = set()
        for edge_type in list_of_edge_types:
            node_type.add(edge_type[0])
            node_type.add(edge_type[2])
        list_of_node_types = list(node_type)

        self.graph_node_edge_attr = GraphNodeTextualEdgeAttr(args, hidden_dim=feature_dim, list_of_node_types=list_of_node_types, list_of_edge_types=list_of_edge_types)
        self.gnn_layer = UnifiedGraphConv(args, feature_dim, list_of_edge_types=list_of_edge_types)

        self.dropout = args.gnn_dropout
    
    def forward(self, batch):
        batch = self.graph_node_edge_attr(batch)
        x_dict = batch.x_dict
        edge_index_dict = batch.edge_index_dict
        edge_attr_dict = batch.edge_attr_dict

        x_dict = self.gnn_layer(x_dict, edge_index_dict, edge_attr_dict)
        
        if self.dropout > 0:
            for node_type in x_dict.keys():
                x_dict[node_type] = F.dropout(x_dict[node_type], p=self.dropout, training=self.training)
        
        batch.x_dict = x_dict
        return batch

