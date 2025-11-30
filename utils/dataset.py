import json
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer

class OmiDataset(Dataset):
    def __init__(self, args, data_list, tokenizer):
        self.data_list = data_list
        self.tokenizer = tokenizer
        self.max_sen_length = args.max_sen_length
        self.max_news_length = args.max_news_length

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        news = self.data_list[idx]
        news_content = news['content']
        sentence_texts = news['sentence']
        environment_texts = news['environment']
        omi_intent = news['omi_intent']
        label = news['label']

        sentence_env_link = news['sentence_env_link']

        tokenized_sentences = self.tokenizer(sentence_texts, padding='max_length', truncation=True, max_length=self.max_sen_length, return_tensors='pt')
        sentence_token_ids = tokenized_sentences['input_ids']
        sentence_masks = tokenized_sentences['attention_mask']

        tokenized_envs = self.tokenizer(environment_texts, padding='max_length', truncation=True, max_length=self.max_sen_length, return_tensors='pt')
        env_token_ids = tokenized_envs['input_ids']
        env_masks = tokenized_envs['attention_mask']

        tokenized_omi_intent = self.tokenizer(omi_intent, padding='max_length', truncation=True, max_length=self.max_sen_length, return_tensors='pt')
        omi_intent_token_ids = tokenized_omi_intent['input_ids']
        omi_intent_masks = tokenized_omi_intent['attention_mask']

        tokenized_news = self.tokenizer(news_content, padding='max_length', truncation=True, max_length=self.max_news_length, return_tensors='pt')
        news_token_ids = tokenized_news['input_ids']
        news_masks = tokenized_news['attention_mask']        

        # print(f"sentence_token_ids: {sentence_token_ids.shape}")
        # exit()
        
        return {
            'news_id':  torch.tensor(news['news_id']).float(),
            'news_token_id': news_token_ids,
            'news_mask': news_masks,

            'sentence_token_id': sentence_token_ids,
            'sentence_mask': sentence_masks,

            'env_token_id': env_token_ids,
            'env_mask': env_masks,

            'omi_intent_token_id': omi_intent_token_ids,
            'omi_intent_mask': omi_intent_masks,

            'label': torch.tensor(label).long(),

            'sentence_env_link': torch.tensor(sentence_env_link, dtype=torch.long),
        }

class OmiPlusDataset(Dataset):
    def __init__(self, args, data_list, tokenizer, commission_embeddings):
        self.data_list = data_list
        self.tokenizer = tokenizer
        self.commission_embeddings = commission_embeddings
        self.max_sen_length = args.max_sen_length
        self.max_news_length = args.max_news_length

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        news = self.data_list[idx]
        news_content = news['content']
        sentence_texts = news['sentence']
        environment_texts = news['environment']
        omi_intent = news['omi_intent']
        label = news['label']

        sentence_env_link = news['sentence_env_link']

        tokenized_sentences = self.tokenizer(sentence_texts, padding='max_length', truncation=True, max_length=self.max_sen_length, return_tensors='pt')
        sentence_token_ids = tokenized_sentences['input_ids']
        sentence_masks = tokenized_sentences['attention_mask']

        tokenized_envs = self.tokenizer(environment_texts, padding='max_length', truncation=True, max_length=self.max_sen_length, return_tensors='pt')
        env_token_ids = tokenized_envs['input_ids']
        env_masks = tokenized_envs['attention_mask']

        tokenized_omi_intent = self.tokenizer(omi_intent, padding='max_length', truncation=True, max_length=self.max_sen_length, return_tensors='pt')
        omi_intent_token_ids = tokenized_omi_intent['input_ids']
        omi_intent_masks = tokenized_omi_intent['attention_mask']

        tokenized_news = self.tokenizer(news_content, padding='max_length', truncation=True, max_length=self.max_news_length, return_tensors='pt')
        news_token_ids = tokenized_news['input_ids']
        news_masks = tokenized_news['attention_mask']

        if len(self.commission_embeddings) > 0:
            commission_embeddings = self.commission_embeddings[idx] 
            commission_embeddings = torch.stack(commission_embeddings)  # Shape: [num_commissions, embedding_dim]

        # print(f"sentence_token_ids: {sentence_token_ids.shape}")
        # exit()
        
        return {
            'news_id':  torch.tensor(news['news_id']).float(),
            'news_token_id': news_token_ids,
            'news_mask': news_masks,

            'sentence_token_id': sentence_token_ids,
            'sentence_mask': sentence_masks,

            'env_token_id': env_token_ids,
            'env_mask': env_masks,

            'omi_intent_token_id': omi_intent_token_ids,
            'omi_intent_mask': omi_intent_masks,

            'commission_embedding': commission_embeddings,

            'label': torch.tensor(label).long(),

            'sentence_env_link': torch.tensor(sentence_env_link, dtype=torch.long),
        }
    

class NewsDataset(Dataset):
    def __init__(self, args, data_list, tokenizer):
        self.data_list = data_list
        self.tokenizer = tokenizer
        self.max_news_length = args.max_news_length

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        news = self.data_list[idx]
        news_content = news['content']
        label = news['label']

        tokenized_news = self.tokenizer(news_content, padding='max_length', truncation=True, max_length=self.max_news_length, return_tensors='pt')
        news_token_ids = tokenized_news['input_ids']
        news_masks = tokenized_news['attention_mask']
        
        return {
            'news_id':  torch.tensor(news['news_id']).float(),
            'news_token_id': news_token_ids,
            'news_mask': news_masks,
            'label': torch.tensor(label).long(),
        }

def omigraph_collate_fn(batch):
    def get_max_sequence_length(tensor_list):
        max_len = 0
        for tensor in tensor_list:
            max_len = max(max_len, tensor.size(0))
        return max_len
    
    def pad_to_max_sequence_length(tensor_list, padding_value):
        max_len = get_max_sequence_length(tensor_list)
        padded_list = []
        mask = []
        for tensor in tensor_list:
            pad_size = max_len - tensor.size(0)
            if pad_size > 0:
                padding = torch.full((pad_size, *tensor.size()[1:]), padding_value, dtype=tensor.dtype)
                padded_list.append(torch.cat([tensor, padding], dim=0))
                mask.append(torch.cat([torch.ones_like(tensor), torch.zeros_like(padding)], dim=0))
            else:
                padded_list.append(tensor)
                mask.append(torch.ones_like(tensor))
        return torch.stack(padded_list), torch.stack(mask)
    
    # padding to the max sequence length of each tensor list within a batch
    news_tonek_ids = pad_to_max_sequence_length([item['news_token_id'] for item in batch], padding_value=0)[0]
    news_masks = pad_to_max_sequence_length([item['news_mask'] for item in batch], padding_value=0)[0]
    sentence_token_ids = pad_to_max_sequence_length([item['sentence_token_id'] for item in batch], padding_value=0)[0]
    sentence_masks = pad_to_max_sequence_length([item['sentence_mask'] for item in batch], padding_value=0)[0]
    env_token_ids = pad_to_max_sequence_length([item['env_token_id'] for item in batch], padding_value=0)[0]
    env_masks = pad_to_max_sequence_length([item['env_mask'] for item in batch], padding_value=0)[0]
    
    omi_intent_token_ids = pad_to_max_sequence_length([item['omi_intent_token_id'] for item in batch], padding_value=0)[0]
    omi_intent_masks = pad_to_max_sequence_length([item['omi_intent_mask'] for item in batch], padding_value=0)[0]
    # print(f"omi_intent_token_ids shape: {omi_intent_token_ids.shape}")
    # exit()

    # no need for padding or truncation
    labels = torch.tensor([item['label'] for item in batch])
    news_ids = torch.tensor([item['news_id'] for item in batch])
    
    sentence_env_link = pad_to_max_sequence_length([item['sentence_env_link'] for item in batch], padding_value=-1)[0]
    sentence_env_link = sentence_env_link.permute(0, 2, 1)

    return_dict = {
        'news_id': news_ids,
        'news_token_id': news_tonek_ids,
        'news_mask': news_masks, 
        'sentence_token_id': sentence_token_ids, # [bs, 32, 32]
        'sentence_mask': sentence_masks, # [bs, 32, 32]
        'env_token_id': env_token_ids, # [bs, 32, 32]
        'env_mask': env_masks, # [bs, 32, 32]
        'omi_intent_token_id': omi_intent_token_ids, # [bs, 1, 32]
        'omi_intent_mask': omi_intent_masks, # [bs, 1, 32]

        'label': labels, # [bs]
        'sentence_env_link': sentence_env_link, # [bs, 2, 1]
    }
    # print(f"return_dict={return_dict}")
    return return_dict


def omigraph_plus_collate_fn(batch):
    def get_max_sequence_length(tensor_list):
        max_len = 0
        for tensor in tensor_list:
            max_len = max(max_len, tensor.size(0))
        return max_len
    
    def pad_to_max_sequence_length(tensor_list, padding_value):
        max_len = get_max_sequence_length(tensor_list)
        padded_list = []
        mask = []
        for tensor in tensor_list:
            pad_size = max_len - tensor.size(0)
            if pad_size > 0:
                padding = torch.full((pad_size, *tensor.size()[1:]), padding_value, dtype=tensor.dtype)
                padded_list.append(torch.cat([tensor, padding], dim=0))
                mask.append(torch.cat([torch.ones_like(tensor), torch.zeros_like(padding)], dim=0))
            else:
                padded_list.append(tensor)
                mask.append(torch.ones_like(tensor))
        return torch.stack(padded_list), torch.stack(mask)
    
    # padding to the max sequence length of each tensor list within a batch
    news_tonek_ids = pad_to_max_sequence_length([item['news_token_id'] for item in batch], padding_value=0)[0]
    news_masks = pad_to_max_sequence_length([item['news_mask'] for item in batch], padding_value=0)[0]
    sentence_token_ids = pad_to_max_sequence_length([item['sentence_token_id'] for item in batch], padding_value=0)[0]
    sentence_masks = pad_to_max_sequence_length([item['sentence_mask'] for item in batch], padding_value=0)[0]
    env_token_ids = pad_to_max_sequence_length([item['env_token_id'] for item in batch], padding_value=0)[0]
    env_masks = pad_to_max_sequence_length([item['env_mask'] for item in batch], padding_value=0)[0]
    
    omi_intent_token_ids = pad_to_max_sequence_length([item['omi_intent_token_id'] for item in batch], padding_value=0)[0]
    omi_intent_masks = pad_to_max_sequence_length([item['omi_intent_mask'] for item in batch], padding_value=0)[0]
    commission_embeddings = pad_to_max_sequence_length([item['commission_embedding'] for item in batch], padding_value=0)[0]

    labels = torch.tensor([item['label'] for item in batch])
    news_ids = torch.tensor([item['news_id'] for item in batch])
    
    sentence_env_link = pad_to_max_sequence_length([item['sentence_env_link'] for item in batch], padding_value=-1)[0]
    sentence_env_link = sentence_env_link.permute(0, 2, 1)

    return_dict = {
        'news_id': news_ids,
        'news_token_id': news_tonek_ids,
        'news_mask': news_masks, 
        'sentence_token_id': sentence_token_ids, # [bs, 32, 32]
        'sentence_mask': sentence_masks, # [bs, 32, 32]
        'env_token_id': env_token_ids, # [bs, 32, 32]
        'env_mask': env_masks, # [bs, 32, 32]
        'omi_intent_token_id': omi_intent_token_ids, # [bs, 1, 32]
        'omi_intent_mask': omi_intent_masks, # [bs, 1, 32]
        'commission_embedding': commission_embeddings,

        'label': labels, # [bs]
        'sentence_env_link': sentence_env_link, # [bs, 2, 1]
    }
    # print(f"return_dict={return_dict}")
    return return_dict

def collate_fn(batch):
    def get_max_sequence_length(tensor_list):
        max_len = 0
        for tensor in tensor_list:
            max_len = max(max_len, tensor.size(0))
        return max_len
    
    def pad_to_max_sequence_length(tensor_list, padding_value):
        max_len = get_max_sequence_length(tensor_list)
        padded_list = []
        mask = []
        for tensor in tensor_list:
            pad_size = max_len - tensor.size(0)
            if pad_size > 0:
                padding = torch.full((pad_size, *tensor.size()[1:]), padding_value, dtype=tensor.dtype)
                padded_list.append(torch.cat([tensor, padding], dim=0))
                mask.append(torch.cat([torch.ones_like(tensor), torch.zeros_like(padding)], dim=0))
            else:
                padded_list.append(tensor)
                mask.append(torch.ones_like(tensor))
        return torch.stack(padded_list), torch.stack(mask)
    
    # padding to the max sequence length of each tensor list within a batch
    news_tonek_ids = pad_to_max_sequence_length([item['news_token_id'] for item in batch], padding_value=0)[0]
    news_masks = pad_to_max_sequence_length([item['news_mask'] for item in batch], padding_value=0)[0]    

    labels = torch.tensor([item['label'] for item in batch])
    news_ids = torch.tensor([item['news_id'] for item in batch])
    
    return {
        'news_id': news_ids,
        'news_token_id': news_tonek_ids,
        'news_mask': news_masks,
        'label': labels, # [64]
    }

def dataset_creator(args, split='train'):
    if args.model.lower() in ['bert']:
        data_path = f"{args.dataset_path}/{split}.json"
    elif args.model.lower() in ['omigraph']:
        if 'no_env' in args.prompt_type:
            data_path = f"{args.dataset_path}/{split}_formatted.json"
        else:
            data_path = f"{args.dataset_path}/{split}_formatted.json"
    else:
        data_path = f"{args.dataset_path}/{split}_32_32.json"
    data_list = json.load(open(data_path, 'r', encoding='utf-8'))    
    
    if args.model.lower() in ['omigraph']:
        commission_embeddings = torch.tensor([])
        if len(args.exist_detector) > 0:
            commission_embeddings = torch.load(f"{args.dataset_path}/{split}_{args.exist_detector}.pt", weights_only=False, map_location='cpu')
            print(f"Loading commission embeddings from {data_path.replace('.json', f'_{args.exist_detector}.pt')}, shape: {commission_embeddings.shape}")
    
    tokenizer = BertTokenizer.from_pretrained(args.backbone_model_path)
    print(f"Loading tokenizer from {args.backbone_model_path}")


    if args.model.lower() in ['omigraph']:
        if len(args.exist_detector) > 0:
            dataset = OmiPlusDataset(args,
                                    data_list, 
                                    tokenizer,
                                    commission_embeddings)
        else:
            dataset = OmiDataset(args,
                                data_list, 
                                tokenizer)
    else:
        dataset = NewsDataset(args,
                              data_list, 
                              tokenizer)

    return dataset