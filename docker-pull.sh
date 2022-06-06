#!/bin/bash
curl https://nvidia.github.io/nvidia-docker/centos7/nvidia-docker.repo > /etc/yum.repos.d/nvidia-docker.repo &
yum -y install nvidia-container-toolkit &
docker run --gpus all -tid --ipc=host -v /home/wxh:/home/wxh pytorch/pytorch:latest
