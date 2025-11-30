#!/bin/bash

python main.py \
--batchsize 48 \
--epoch 20 \
--model OmiGraph \
--backbone_model bert \
--output_dir results_test \
--dataset_name newsenv-en \
--custom_name test \
--custom_log_name test \
--lr 2e-5 \
--seed 3456 \
--env_link_strategy intent