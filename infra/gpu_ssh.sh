#!/usr/bin/env bash
# SSH/scp/rsync to the rAvI GPU box through the EC2 Instance Connect Endpoint
# (direct port-22 from the UA network is reset by a middlebox; the EICE tunnels over HTTPS).
#   infra/gpu_ssh.sh                 -> interactive shell
#   infra/gpu_ssh.sh 'cmd'           -> run a command
#   infra/gpu_ssh.sh rsync SRC DST   -> rsync; use gpu: prefix for the remote side, e.g. gpu:/data/
set -euo pipefail
INSTANCE=${GPU_INSTANCE:-i-08a479e674e43c4ef}
KEY=${GPU_KEY:-$HOME/.ssh/fxjepa-oct2026.pem}
PROXY="aws ec2-instance-connect open-tunnel --instance-id $INSTANCE --profile dev-sso --region us-east-1"
SSH_OPTS=(-i "$KEY" -o "ProxyCommand=$PROXY" -o StrictHostKeyChecking=accept-new -o ServerAliveInterval=30 -o IdentitiesOnly=yes)
if [[ "${1:-}" == "rsync" ]]; then shift
  args=(); for a in "$@"; do args+=("${a/gpu:/ubuntu@$INSTANCE:}"); done
  # macOS bash 3.2: build the -e string by hand (no ${var@Q})
  exec rsync -az -e "ssh -i $KEY -o ProxyCommand='$PROXY' -o StrictHostKeyChecking=accept-new -o IdentitiesOnly=yes" "${args[@]}"
fi
exec ssh "${SSH_OPTS[@]}" "ubuntu@$INSTANCE" "$@"
