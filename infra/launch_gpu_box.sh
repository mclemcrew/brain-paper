#!/usr/bin/env bash
# Launch (or re-launch) the rAvI v2 GPU dev box in the dev account.
# Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 22.04): Docker + NVIDIA container toolkit preinstalled.
set -euo pipefail
PROFILE=${PROFILE:-dev-sso}; REGION=us-east-1
AMI=$(aws ssm get-parameter --name /aws/service/deeplearning/ami/x86_64/base-oss-nvidia-driver-gpu-ubuntu-22.04/latest/ami-id --profile $PROFILE --region $REGION --query Parameter.Value --output text)
TYPE=${TYPE:-g6e.xlarge}            # 1x L40S 48 GB, 4 vCPU, 32 GB RAM, ~$1.86/h on-demand
SUBNET=${SUBNET:-subnet-0333003d3d9af15bd}   # public subnet in vpc-06afea3b89ea0172b; 1a default, 1b subnet-07a28ab4941191cc7, 1c subnet-0c58a9a6ab6801ff7
SG=sg-098f440b8bf1a5b44               # ravi-v2-ssh (port 22 from Michael's IP)
KEY=fxjepa-oct2026                    # private key at ~/.ssh/fxjepa-oct2026.pem
PROFILE_ARG=""; [[ -n "${INSTANCE_PROFILE:-}" ]] && PROFILE_ARG="--iam-instance-profile Name=$INSTANCE_PROFILE"
aws ec2 run-instances --profile $PROFILE --region $REGION \
  --image-id "$AMI" --instance-type "$TYPE" --key-name $KEY \
  --subnet-id $SUBNET --security-group-ids $SG $PROFILE_ARG \
  --block-device-mappings '[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":300,"VolumeType":"gp3","DeleteOnTermination":true}}]' \
  --metadata-options HttpTokens=required \
  --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=ravi-v2-gpu},{Key=Project,Value=ravi-v2},{Key=Owner,Value=mclemens@uaudio.com},{Key=ExpiresOn,Value=2026-11-30},{Key=map-migrated,Value=mig6AX58RZ218}]" \
  --query 'Instances[0].{id:InstanceId,az:Placement.AvailabilityZone,type:InstanceType}' --output json
