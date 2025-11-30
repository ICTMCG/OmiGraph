import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
# === Model-related modules ===
from transformers import BertModel as BertModel_
from torch_geometric.utils import softmax as scatter_softmax


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

class BertModel(nn.Module):
    def __init__(self, args):
        super(BertModel, self).__init__()

        self.bert_dim = args.bert_dim
        self.feature_dim = args.feature_dim

        model_path = args.backbone_model_path
        self.bert_model = BertModel_.from_pretrained(model_path).requires_grad_(False)
        for name, param in self.bert_model.named_parameters():
            if name.startswith("encoder.layer.11"): \
                param.requires_grad = True
            else:
                param.requires_grad = False
        
        self.bert_to_feature = nn.Linear(self.bert_dim, self.feature_dim)
        self.cls_mlp = MLP(self.feature_dim, hidden_dim=128, output_dim=2, num_layers=2)
        
        # loss weight
        self.criterion_weight_dict = {
            "loss_cls": args.loss_cls
        }

        # loss function
        self.loss_bce = nn.CrossEntropyLoss()
        self.loss_alpha = args.loss_alpha

    def forward(self, args, batch_news):
        # ====== Data preparation ======
        news_token_ids = batch_news['news_token_id'].squeeze(1)
        news_masks = batch_news['news_mask'].squeeze(1)
        
        news_embeddings = self.get_gloabl_news_features(news_token_ids, news_masks)
        preds_news_semantic = self.cls_mlp(news_embeddings).squeeze(1)
        
        return preds_news_semantic
    
    def get_gloabl_news_features(self, news_token_ids, news_masks):
        news_emb = self.bert_model(input_ids=news_token_ids, attention_mask=news_masks).last_hidden_state
        news_emb = news_emb[:, 0]
        news_emb = self.bert_to_feature(news_emb)
        return news_emb

    def get_criterion(self, outputs, targets):
        loss_dic = {}
        loss_dic["loss_cls"] = self.loss_bce(outputs, targets)
        return loss_dic

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])