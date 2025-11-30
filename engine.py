import math
import os
import sys
from typing import Iterable
import torch
import json
import utils.misc as utils
import numpy as np
import torch.distributed as dist
from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score, roc_auc_score

def to_device(item, device):
    if isinstance(item, torch.Tensor):
        return item.to(device)
    elif isinstance(item, list):
        return [to_device(i, device) for i in item]
    elif isinstance(item, dict):
        return {k: to_device(v, device) for k,v in item.items()}
    else:
        raise NotImplementedError("Call if you use other containers! type: {}".format(type(item)))


def train_one_epoch(model: torch.nn.Module, data_loader: Iterable, 
                    optimizer: torch.optim.Optimizer, device: torch.device, 
                    epoch: int, lr_scheduler = None, max_norm: float = 0, args=None, criterion_weight_dict=None):
    
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = args.print_freq

    for samples in metric_logger.log_every(data_loader, print_freq, header):
        samples = to_device(samples, device)
        outputs = model(args, samples)
        if args.distributed:
            model_without_ddp = model.module
        else:
            model_without_ddp = model
        loss_dict = model_without_ddp.get_criterion(outputs, samples["label"])
        weight_dict = criterion_weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        # original backward function
        optimizer.zero_grad()
        losses.backward()
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()
        # lr_scheduler.step()

        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)


@torch.no_grad()
def gather_together(data):
    world_size = utils.get_world_size()
    if world_size < 2:
        return [data]
    dist.barrier()
    gather_data = [None for _ in range(world_size)]
    dist.all_gather_object(gather_data, data)
    return gather_data


@torch.no_grad()
def evaluate(model, data_loader, device, args=None):
    model.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    print_freq = args.print_freq

    y_true, y_pred, news_ids = [], [], []
    for samples in metric_logger.log_every(data_loader, print_freq, header):
        samples = to_device(samples, device)
        labels = samples["label"]
        outputs = model(args, samples)
        news_id = samples["news_id"]
        
        if isinstance(outputs, tuple):
            outputs = outputs[0]
        y_pred.extend(outputs.softmax(dim=1)[:, 1].flatten().tolist())
        y_true.extend(labels.flatten().tolist())
        news_ids.extend(news_id.flatten().tolist())
    
    merge_y_true = []
    for data in gather_together(y_true):
        merge_y_true.extend(data)
    
    merge_y_pred = []
    for data in gather_together(y_pred):
        merge_y_pred.extend(data)
    
    y_true, y_pred = np.array(merge_y_true), np.array(merge_y_pred)
    if args.save_preds_logits:
        prediction_file = os.path.join(args.output_dir + f"/{args.custom_log_name}.txt")
        with open(prediction_file, 'w') as f:
            for nid, true, pred in zip(news_ids, y_true, y_pred):
                # pred = 1 if pred > 0.5 else 0
                f.write(f"{int(nid)},\t{true},\t{pred}\n")

    all_metrics = {}
    try:
        all_metrics['auc'] = roc_auc_score(y_true, y_pred, average='macro')
        all_metrics['spauc'] = roc_auc_score(y_true, y_pred, average='macro', max_fpr=0.1)
    except:
        all_metrics['auc'] = 0
        all_metrics['spauc'] = 0
    y_pred = y_pred > 0.5
    all_metrics['mac_f1'] = f1_score(y_true, y_pred, average='macro')
    all_metrics['f1_real'], all_metrics['f1_fake'] = f1_score(y_true, y_pred, average=None)
    all_metrics['recall'] = recall_score(y_true, y_pred, average='macro')
    all_metrics['recall_real'], all_metrics['recall_fake'] = recall_score(y_true, y_pred, average=None)
    all_metrics['precision'] = precision_score(y_true, y_pred, average='macro')
    all_metrics['precision_real'], all_metrics['precision_fake'] = precision_score(y_true, y_pred, average=None)
    all_metrics['acc'] = accuracy_score(y_true, y_pred)
    
    for k, v in all_metrics.items():
        print(f"{k}: {v}")
    
    return json.dumps(all_metrics), all_metrics, all_metrics['mac_f1']