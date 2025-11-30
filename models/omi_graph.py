import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import HeteroGNNLayer, Global_aware_Aggregator, HeteroGNNPooling, MLP, copy, UnifiedGNNLayer, UnifiedTextualGNNLayer
from transformers import BertModel
from torch_geometric.data import HeteroData, Batch
from torch_geometric.nn import GraphConv


class OmiGraph(nn.Module):
    def __init__(self, args):
        super(OmiGraph, self).__init__()

        self.bert_dim = args.bert_dim
        self.feature_dim = args.feature_dim
        self.sentence_connect_strategy = args.sentence_connect_strategy
        self.sentence_connect_window_size = args.sentence_connect_window_size

        self.bert_model = BertModel.from_pretrained(args.backbone_model_path).requires_grad_(False)
        print(f"Loading backbone model from {args.backbone_model_path}")
        for name, param in self.bert_model.named_parameters():
            if name.startswith("encoder.layer.11"):
                param.requires_grad = True
            else:
                param.requires_grad = False
    
        self.bert_to_feature = nn.Linear(self.bert_dim, self.feature_dim)
        self.env_to_feature = nn.Linear(self.bert_dim, self.feature_dim)
        self.int_to_feature = nn.Linear(args.bert_dim, self.feature_dim)
        if len(args.exist_detector) > 0:
            self.commission_to_feature = nn.Linear(args.commission_dim, self.feature_dim)
        
        if 'sem' in args.use_graph_type:
            self.SemHeteroGNN = _get_clones(HeteroGNNLayer(args, self.feature_dim, [
                ('sentence', 'to', 'sentence')
                ], conv_type=GraphConv), args.num_sem_gnn_layer)
        if not args.not_use_global_update:
            self.news_specific_aggregator = Global_aware_Aggregator(args, hidden_dim=self.feature_dim, node_type=['sentence'])
        self.SemHeteroGNNPool = HeteroGNNPooling(hidden_dim=self.feature_dim, node_type=['sentence'], pooling_type='mean')

        self.cls_mlp_semantic = MLP(self.feature_dim, hidden_dim=128, output_dim=2, num_layers=2)
        self.cls_mlp_final = MLP(self.feature_dim, hidden_dim=128, output_dim=2, num_layers=2)
        
        if args.ab_path:
            self.MergeHeteroGNN = _get_clones(UnifiedGNNLayer(args, self.feature_dim, [
                ('env', 'to', 'sentence'), 
                ('sentence', 'to', 'env')
                ]), args.num_merge_gnn_layer)
        else:
            if args.env_link_strategy == 'intent':
                self.MergeHeteroGNN = _get_clones(UnifiedTextualGNNLayer(args, self.feature_dim, [
                    ('sentence', 'to', 'env'), ('env', 'to', 'sentence'), ('sentence', 'to', 'sentence')
                    ]), args.num_merge_gnn_layer)
            else:
                self.virtual_node = nn.Parameter(torch.randn(args.num_virtual_subnode, self.feature_dim))
                self.MergeHeteroGNN = _get_clones(UnifiedGNNLayer(args, self.feature_dim, [
                    ('virtual', 'to', 'sentence'), ('virtual', 'to', 'env'),
                    ('sentence', 'to', 'virtual'), ('env', 'to', 'virtual'), ('sentence', 'to', 'sentence')
                    ]), args.num_merge_gnn_layer)
        
        self.OmiHeteroGNNPool = HeteroGNNPooling(hidden_dim=self.feature_dim, node_type=['sentence'], pooling_type='mean')

        if len(args.exist_detector) > 0:
            self.cls_mlp_commission = MLP(self.feature_dim, hidden_dim=128, output_dim=2, num_layers=2)
        self.loss_bce = nn.CrossEntropyLoss()

        self.tanh = nn.Tanh()

    def forward(self, args, batch_news):
        # ====== Data preparation ======
        news_token_ids = batch_news['news_token_id'].squeeze(1)
        news_masks = batch_news['news_mask'].squeeze(1)

        sentence_token_ids = batch_news['sentence_token_id']
        sentence_masks = batch_news['sentence_mask']

        env_token_ids = batch_news['env_token_id']
        env_masks = batch_news['env_mask']

        intent_token_ids = batch_news['omi_intent_token_id']
        intent_masks = batch_news['omi_intent_mask']
        if len(args.exist_detector) > 0:
            commission_embeddings = batch_news['commission_embedding']

        sentence_env_links = batch_news['sentence_env_link']

        labels = batch_news['label']
        news_ids = batch_news['news_id']
                
        news_embeddings = self.get_text_features(news_token_ids, news_masks)
        
        sem_graph_list = self.init_sem_graph(sentence_token_ids, sentence_masks)
        sem_graph_batch = Batch.from_data_list(sem_graph_list)
        
        if 'sem' in args.use_graph_type:
            for idx, gnn in enumerate(self.SemHeteroGNN):
                resdual_x_dict = _copy_tensor_dict(sem_graph_batch.x_dict)
                sem_graph_batch = gnn(sem_graph_batch)
                for node_k, node_v in sem_graph_batch.x_dict.items():
                    sem_graph_batch.x_dict[node_k] = F.leaky_relu(node_v)
                    sem_graph_batch.x_dict[node_k] = sem_graph_batch.x_dict[node_k] + resdual_x_dict[node_k]
                if not args.not_use_global_update:
                    sem_graph_batch = self.news_specific_aggregator(sem_graph_batch, news_embeddings)
            pooled_gnn_features = self.SemHeteroGNNPool(sem_graph_batch) #[bs, 256]
            preds_semantic_graph = self.cls_mlp_semantic(pooled_gnn_features).squeeze(1)

        if args.ab_env:
            return preds_semantic_graph
        
        if args.env_link_strategy == 'intent':
            omi_graph_list = self.init_omi_textual_edge_graph(sem_graph_batch, env_token_ids, env_masks, intent_token_ids, intent_masks, sentence_env_links)
        else:
            omi_graph_list = self.init_mix_graph(sem_graph_batch, env_token_ids, env_masks)
        omi_graph_batch = Batch.from_data_list(omi_graph_list)
        
        for idx, gnn in enumerate(self.MergeHeteroGNN):
            resdual_x_dict = _copy_tensor_dict(omi_graph_batch.x_dict)
            omi_graph_batch = gnn(omi_graph_batch)
            for node_k, node_v in omi_graph_batch.x_dict.items():
                omi_graph_batch.x_dict[node_k] = F.leaky_relu(node_v)
                omi_graph_batch.x_dict[node_k] = omi_graph_batch.x_dict[node_k] + resdual_x_dict[node_k]

        pooled_gnn_features = self.OmiHeteroGNNPool(omi_graph_batch)
        preds_final = self.cls_mlp_final(pooled_gnn_features).squeeze(1)
                
        if len(args.exist_detector) > 0:
            commission_features = self.commission_to_feature(commission_embeddings)
            preds_commission = self.cls_mlp_commission(commission_features).squeeze(1)
            output = preds_final * self.tanh(preds_commission)
        else:
            output = preds_final
        
        return output
    
    def init_sem_graph(self, sentence_token_ids, sentence_masks):
        news_sen_embs = []
        for sen_ids, sen_mask in zip(sentence_token_ids, sentence_masks):
            valid_sentences = sen_ids[sen_mask.sum(dim=1) > 0]
            valid_masks = sen_mask[sen_mask.sum(dim=1) > 0]
            sen_emb = self.bert_model(input_ids=valid_sentences, attention_mask=valid_masks).last_hidden_state
            sen_emb = sen_emb[:, 0]
            sen_feature = self.bert_to_feature(sen_emb)
            news_sen_embs.append(sen_feature)            
        
        semantic_graph_list = []
        for news_sen_emb in news_sen_embs:
            graph = HeteroData()
            graph['sentence'].x = news_sen_emb
            if self.sentence_connect_strategy == 'full':
                graph['sentence', 'to', 'sentence'].edge_index = torch.cartesian_prod(torch.arange(news_sen_emb.size(0)), torch.arange(news_sen_emb.size(0))).t().to(news_sen_emb.device)
            elif self.sentence_connect_strategy == 'seq':
                graph['sentence', 'to', 'sentence'].edge_index = torch.cat([
                    torch.stack([torch.arange(news_sen_emb.size(0))[:-1], torch.arange(news_sen_emb.size(0) - 1) + 1], dim=0),
                    torch.stack([torch.arange(news_sen_emb.size(0))[:-1], torch.arange(news_sen_emb.size(0) - 1) + 1], dim=0).flip(0)
                ], dim=-1).to(news_sen_emb.device)
            elif self.sentence_connect_strategy == 'window':
                edge_index = []
                n_sen = news_sen_emb.size(0)
                window_size = self.sentence_connect_window_size
                # from left to right
                for idx in range(n_sen):
                    for offset in range(1, window_size + 1):
                        if idx + offset < n_sen: 
                            edge_index.append([idx, idx + offset])
                        else:
                            break
                edge_index = torch.tensor(edge_index, dtype=torch.long, device=news_sen_emb.device).t()
                # merge from left to right
                graph['sentence', 'to', 'sentence'].edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=-1)
            else:
                raise NotImplementedError
            
            semantic_graph_list.append(graph)
        
        return semantic_graph_list
    
    def init_omi_textual_edge_graph(self, batched_sem_graph, env_token_ids, env_masks, intent_token_ids, intent_masks, sen_env_links):
        env_sen_embs = []
        for sen_ids, sen_mask in zip(env_token_ids, env_masks):
            valid_sentences = sen_ids[sen_mask.sum(dim=1) > 0]
            valid_masks = sen_mask[sen_mask.sum(dim=1) > 0]
            sen_emb = self.bert_model(input_ids=valid_sentences, attention_mask=valid_masks).last_hidden_state
            sen_emb = sen_emb[:, 0]
            sen_feature = self.env_to_feature(sen_emb)
            env_sen_embs.append(sen_feature)
        
        int_embs = []
        for int_ids, int_mask in zip(intent_token_ids, intent_masks):
            valid_ints = int_ids[int_mask.sum(dim=1) > 0]
            valid_masks = int_mask[int_mask.sum(dim=1) > 0]
            int_emb = self.bert_model(input_ids=valid_ints, attention_mask=valid_masks).last_hidden_state
            int_emb = int_emb[:, 0]
            int_feature = self.int_to_feature(int_emb)
            int_embs.append(int_feature)

        int_graph_list = []
        for sem_graph, env_emb, intent_emb, sen_env_link in zip(batched_sem_graph.to_data_list(), env_sen_embs, int_embs, sen_env_links):
            int_graph = HeteroData()
            int_x_dic = {}
            int_x_dic.update(sem_graph.x_dict)
            int_graph.set_value_dict('x', int_x_dic)
            int_edge_index_dic = {}
            int_edge_index_dic.update(sem_graph.edge_index_dict)
            int_graph.set_value_dict('edge_index', int_edge_index_dic)

            int_graph['env'].x = env_emb

            sen_env_link = sen_env_link[:, sen_env_link[0] != -1]
            int_graph['sentence', 'to', 'env'].edge_index = sen_env_link
            int_graph['env', 'to', 'sentence'].edge_index = sen_env_link.flip(0)
            int_graph['env', 'to', 'sentence'].edge_attr = intent_emb
            int_graph['sentence', 'to', 'env'].edge_attr = intent_emb

            int_graph_list.append(int_graph)
        
        return int_graph_list

    def init_mix_graph(self, batched_sem_graph, env_token_ids, env_masks):
        env_sen_embs = []
        for sen_ids, sen_mask in zip(env_token_ids, env_masks):
            valid_sentences = sen_ids[sen_mask.sum(dim=1) > 0]
            valid_masks = sen_mask[sen_mask.sum(dim=1) > 0]
            sen_emb = self.bert_model(input_ids=valid_sentences, attention_mask=valid_masks).last_hidden_state
            # CLS token as sentence embedding
            sen_emb = sen_emb[:, 0]
            sen_feature = self.env_to_feature(sen_emb)
            env_sen_embs.append(sen_feature)

        mix_graph_list = []
        for sem_graph, env_emb in zip(batched_sem_graph.to_data_list(), env_sen_embs):
            mix_graph = HeteroData()
            sem_x_dic = {}
            sem_x_dic.update(sem_graph.x_dict)
            mix_graph.set_value_dict('x', sem_x_dic)
            sem_edge_index_dic = {}
            sem_edge_index_dic.update(sem_graph.edge_index_dict)
            mix_graph.set_value_dict('edge_index', sem_edge_index_dic)

            # print(mix_graph)
            # print(f"mix_graph.edge_index_dict: {mix_graph.edge_index_dict}")
            # exit()

            mix_graph['env'].x = env_emb
            mix_graph['virtual'].x = self.virtual_node
            mix_graph['virtual', 'to', 'sentence'].edge_index = self.get_full_connected_edge(self.virtual_node.size(0), mix_graph.x_dict['sentence'].size(0)).to(self.virtual_node.device)
            mix_graph['virtual', 'to', 'env'].edge_index = self.get_full_connected_edge(self.virtual_node.size(0), mix_graph.x_dict['env'].size(0)).to(self.virtual_node.device)
            mix_graph['sentence', 'to', 'virtual'].edge_index = self.get_full_connected_edge(mix_graph.x_dict['sentence'].size(0), self.virtual_node.size(0)).to(self.virtual_node.device)
            mix_graph['env', 'to', 'virtual'].edge_index = self.get_full_connected_edge(mix_graph.x_dict['env'].size(0), self.virtual_node.size(0)).to(self.virtual_node.device)

            mix_graph_list.append(mix_graph)
        
        return mix_graph_list

    def get_full_connected_edge(self, num_nodes_1, num_nodes_2):
        edge_index = torch.cartesian_prod(torch.arange(num_nodes_1), torch.arange(num_nodes_2)).t()
        return edge_index
    
    def get_text_features(self, input_token_ids, input_masks):
        text_emb = self.bert_model(input_ids=input_token_ids, attention_mask=input_masks).last_hidden_state
        text_emb = text_emb[:, 0]
        text_emb = self.bert_to_feature(text_emb)
        return text_emb
        
    def get_criterion(self, outputs, targets):
        loss_dic = {}
        loss_dic["loss_cls"] = self.loss_bce(outputs, targets)
            
        return loss_dic

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

def _copy_tensor_dict(t_dic):
    copy_dic = {}
    for k, v in t_dic.items():
        copy_dic[k] = v.clone()
    return copy_dic