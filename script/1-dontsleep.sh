#!/bin/bash

#SBATCH -p gpu

### 该作业的作业名
#SBATCH --job-name=llama

### 该作业需要1个节点
#SBATCH --nodes=1

### 该作业需要32个CPU
#SBATCH --ntasks=8

### 申请8块GPU卡
#SBATCH --gres=gpu:1




sleep infinity
