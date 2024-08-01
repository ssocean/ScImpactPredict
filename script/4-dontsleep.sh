#!/bin/bash

#SBATCH -p gpu

### 该作业的作业名
#SBATCH --job-name=llama1

### 该作业需要1个节点
#SBATCH --nodes=1

### 该作业需要32个CPU
#SBATCH --ntasks=32

### 申请8块GPU卡
#SBATCH --gres=gpu:4




sleep infinity
