import os
import datetime
import time
import torch
import random
import numpy as np
import argparse
from pathlib import Path
from utils.misc import get_rank, init_distributed_mode, save_on_master, is_main_process, cleanup_dist
from utils.dataset import dataset_creator, collate_fn, omigraph_collate_fn, omigraph_plus_collate_fn
from torch.utils.data import DataLoader, DistributedSampler, RandomSampler, SequentialSampler
from engine import train_one_epoch, evaluate
from models import build_model
from utils.utils import Recorder
import logging

def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
        
    parser.add_argument('--seed', type=int, default=100)
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--test_train', action='store_true')

    # dataset parameters
    parser.add_argument('--dataset_path', type=str, default='../data', help='')
    parser.add_argument('--dataset_name', type=str, default='newsenv-en')
    parser.add_argument('--backbone_model', type=str, default='bert')
    parser.add_argument('--gen_model', type=str, default='Llama-3.1-8B-Instruct')
    parser.add_argument('--exist_detector', type=str, default='')
    parser.add_argument('--max_sen_length', type=int, default=8)
    parser.add_argument('--max_env_num', type=int, default=32)
    parser.add_argument('--max_news_length', type=int, default=256)

    # training parameters
    parser.add_argument('--epoch', type=int, default=20)
    parser.add_argument('--start_epoch', type=int, default=0)
    parser.add_argument('--batchsize', type=int, default=64)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--beta1', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=0.)
    parser.add_argument('--model_max_norm', type=float, default=0.)
    parser.add_argument('--lr_drop', type=int, default=50)
    parser.add_argument('--gamma', type=float, default=0.1)

    # model
    parser.add_argument('--model', type=str, default='Bert')
    parser.add_argument('--bert_dim', type=int, default=768)
    parser.add_argument('--intent_dim', type=int, default=768)
    parser.add_argument('--feature_dim', type=int, default=256)
    parser.add_argument('--commission_dim', type=int, default=256)
    parser.add_argument("--prompt_type", type = str, default="", help="")
    parser.add_argument("--use_graph_type", type = str, default="omi", help="sem, omi")
    
    parser.add_argument('--num_sem_gnn_layer', type=int, default=2)
    parser.add_argument('--num_merge_gnn_layer', type=int, default=2)
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--agg_alpha', type=float, default=0.5)
    parser.add_argument('--dropout', type=float, default=0.)
    parser.add_argument('--gnn_dropout', type=float, default=0.)
    parser.add_argument('--sentence_connect_strategy', type=str, default='window', choices=['full', 'window', 'seq'])
    parser.add_argument('--env_link_strategy', type=str, default='intent', choices=['intent', 'ab'])
    parser.add_argument('--sentence_connect_window_size', type=int, default=2)
    parser.add_argument('--num_virtual_subnode', type=int, default=8)


    # loss
    parser.add_argument('--loss_cls', type=float, default=1.0)

    # output
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--pretrained_model', type=str, default="")
    parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument('--output_dir', default= 'results')
    parser.add_argument('--save_checkpoint_interval', default=10, type=int)
    parser.add_argument('--custom_name', type=str, default='test')
    parser.add_argument('--custom_log_name', type=str, default='1')
    parser.add_argument('--print_freq', default=50, type=int)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--rank', default=0, type=int,
                        help='number of distributed processes')
    parser.add_argument("--local_rank", type=int, help='local rank for DistributedDataParallel')

    # Ablaition study
    parser.add_argument('--save_preds_logits', action='store_true')
    parser.add_argument('--ablation', action='store_true')
    parser.add_argument('--ab_env', action='store_true')
    parser.add_argument('--ab_path', action='store_true')
    parser.add_argument('--not_use_global_update', action='store_true')
    return parser

def _init_logger(args):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh = logging.FileHandler(os.path.join(args.output_dir + f"/{args.custom_log_name}.log"))
    print(f"Logging to {os.path.join(args.output_dir + f'/{args.custom_log_name}.log')}")
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger

def main(args):
    args.dataset_path = f"{args.dataset_path}/{args.dataset_name}"
    init_distributed_mode(args)
    plm_root_file = '/data/shared/LLMs/'
    if args.output_dir:
        args.output_dir = os.path.join(args.output_dir, args.dataset_name, args.custom_name)
        os.makedirs(args.output_dir, exist_ok=True)
    output_dir = Path(args.output_dir)

    if is_main_process():
        logger = _init_logger(args)
    else:
        logger = logging.getLogger()

    if args.dataset_name in ['weibo', 'newsenv-ch']:
        args.backbone_model_path = f"{plm_root_file}/bert-base-chinese"
    else:
        args.backbone_model_path =f"{plm_root_file}/bert-base-uncased"
    # set device
    device = torch.device(args.device)

    # fix the seed
    seed = args.seed + get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)

    model_name = args.model.lower()
    if model_name in ["omigraph"]:
        if len(args.exist_detector) > 0:
            data_collate_fn = omigraph_plus_collate_fn
        else:
            data_collate_fn = omigraph_collate_fn
    else:
        data_collate_fn = collate_fn

    # model
    model = build_model(args)
    criterion_weight_dict = {
        "loss_cls": args.loss_cls,
    }
    if args.distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = model.to(device)
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=False)
        model_without_ddp = model.module
    
    if args.eval:
        checkpoint_path = output_dir / f'checkpoint_best_macf1.pth'
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])

        dataset_test = dataset_creator(args, "test")
        if args.distributed:
            sampler_test = DistributedSampler(dataset_test, shuffle=False)
        else:
            sampler_test = SequentialSampler(dataset_test)
        dataloader_test = DataLoader(dataset_test, args.batchsize, sampler=sampler_test, drop_last=False, num_workers=args.num_workers, collate_fn=data_collate_fn)
        print(f"Evaluating the model from {checkpoint_path}...")
        
        output_strs, all_metrics_dict, cur_f1 = evaluate(model, dataloader_test, device, args=args)
        if is_main_process():
            logger.info(f"Best model for test:\n{checkpoint_path}\n")
            logger.info(f"Test results:\n{output_strs}\n")
            print(f"Test results:\n{output_strs}\nFINISHED!\n{'='*20}")
        return

    dataset_val = dataset_creator(args, "val")
    if args.distributed:
        sampler_val = DistributedSampler(dataset_val, shuffle=False)
    else:
        sampler_val = SequentialSampler(dataset_val)
    dataloader_val = DataLoader(dataset_val, args.batchsize, sampler=sampler_val, drop_last=False, num_workers=args.num_workers, collate_fn=data_collate_fn)

    dataset_train = dataset_creator(args, "train")
    if args.distributed:
        sampler_train = DistributedSampler(dataset_train)
    else:
        sampler_train = RandomSampler(dataset_train)
    batch_sampler_train = torch.utils.data.BatchSampler(sampler_train, args.batchsize, drop_last=True)
    dataloader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train, num_workers=args.num_workers, collate_fn=data_collate_fn)
    
    optimizer = torch.optim.Adam([{"params": [p for n, p in model.named_parameters() if p.requires_grad]}], lr=args.lr, betas=(args.beta1, 0.999), weight_decay=args.weight_decay)
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop, gamma=args.gamma)
    
    if args.resume:
        checkpoint = torch.load(args.pretrained_model, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])             

        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            import copy
            p_groups = copy.deepcopy(optimizer.param_groups)
            optimizer.load_state_dict(checkpoint['optimizer'])
            for pg, pg_old in zip(optimizer.param_groups, p_groups):
                pg['lr'] = pg_old['lr']
                pg['initial_lr'] = pg_old['initial_lr']
            print(optimizer.param_groups)
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            # this is a hack for doing experiment that resume from checkpoint and also modify lr scheduler (e.g., decrease lr in advance).
            args.override_resumed_lr_drop = True
            if args.override_resumed_lr_drop:
                print('Warning: (hack) args.override_resumed_lr_drop is set to True, so args.lr_drop would override lr_drop in resumed lr_scheduler.')
                lr_scheduler.step_size = args.lr_drop
                lr_scheduler.base_lrs = list(map(lambda group: group['initial_lr'], optimizer.param_groups))
            lr_scheduler.step(lr_scheduler.last_epoch)
            args.start_epoch = checkpoint['epoch'] + 1    

    best_f1 = 0
    best_epoch = 0
    best_metrics = ''
    if is_main_process():
        logger.info(f"{'=' * 10} {args.custom_log_name} {'=' * 10}")
        logger.info(f"Output directory: {output_dir}")
        logger.info(f"Arguments:{args}")
        logger.info(f"Number of parameters: {n_parameters}")
        logger.info(f"Model: {args.model}, Backbone: {args.backbone_model}, Gen Model: {args.gen_model}\n")
        logger.info(f"Dataset: {args.dataset_name}, Training size: {len(dataset_train)}, Validation size: {len(dataset_val)}\n")
        if args.resume:
            logger.info(f"Resuming from checkpoint: {args.pretrained_model}")
        logger.info(f"{'=' * 10} {args.custom_log_name} {'=' * 10}")
    for epoch in range(args.start_epoch, args.epoch):
        epoch_start_time = time.time()
        if args.distributed:
            sampler_train.set_epoch(epoch)
        train_one_epoch(model, dataloader_train, optimizer, device, epoch, lr_scheduler, args.model_max_norm, args=args, criterion_weight_dict=criterion_weight_dict)
        
        lr_scheduler.step()
        
        epoch_time = time.time() - epoch_start_time
        epoch_time_str = str(datetime.timedelta(seconds=int(epoch_time)))
        print('Epoch training time {}'.format(epoch_time_str))
        
        output_strs, all_metrics_dict, cur_f1 = evaluate(model, dataloader_val, device, args=args)
        if args.test_train and epoch % 5 == 0:
            output_strs_train, _, _ = evaluate(model, dataloader_train, device, args=args)

        if args.output_dir and is_main_process():
            logger.info(f"Epoch {epoch}:" + output_strs + "\n")
            if args.test_train and epoch % 5 == 0:
                logger.info(f"-- Test trainset in Epoch {epoch}:" + output_strs_train + "\n")
            logger.info(f"Current epoch [{epoch}] (mac_f1) = {cur_f1}, while former max = {best_f1} in epoch[{best_epoch}]")
            if cur_f1 > best_f1:
                logger.info(f'From {best_f1} (in epoch {best_epoch}) update to {cur_f1} (in epoch {epoch})')
                checkpoint_path = output_dir / f'checkpoint_best_macf1.pth'
                save_on_master({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }, checkpoint_path)
                best_metrics = output_strs
                best_epoch = epoch
                best_f1 = cur_f1
                logger.info(f"=== Epoch [{epoch}] max(mac_f1)={best_f1} ===")
                logger.info(f"Best epoch {best_epoch}: {best_metrics}")

    if args.distributed:
        torch.distributed.barrier()
        cleanup_dist()

    logger.info(f"\n\n{'='*10}\nTraining {args.custom_log_name} completed.\nBest epoch {best_epoch}: {best_metrics}\n{'='*10}\n")
    print(f"\n\n{'='*10}\nTraining {args.custom_log_name} completed.\nBest epoch {best_epoch}: {best_metrics}\n{'='*10}\n")
    
    return

if __name__ == "__main__":    
    parser = get_args_parser()
    args = parser.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)