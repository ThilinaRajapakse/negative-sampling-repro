#!/bin/bash

python train_scripts/pretrain/pretrain_ance.py
python train_scripts/pretrain/pretrain_dpr_base.py
python train_scripts/pretrain/pretrain_dpr_bm.py
python train_scripts/pretrain/pretrain_ict_p.py
python train_scripts/pretrain/pretrain_ict_q.py
python train_scripts/pretrain/pretrain_tas_p.py
python train_scripts/pretrain/pretrain_tas_q.py

python train_scripts/finetune/finetune_ance.py
python train_scripts/finetune/finetune_dpr_base.py
python train_scripts/finetune/finetune_dpr_bm.py
python train_scripts/finetune/finetune_ict_p.py
python train_scripts/finetune/finetune_ict_q.py
python train_scripts/finetune/finetune_tas_p.py
python train_scripts/finetune/finetune_tas_q.py
