#!/usr/bin/env bash
# Run make with supported proxy/trust configuration in Codex Cloud.
set -euo pipefail
cd "$(dirname "$0")/.."
export DOCKER_CONFIG="${DOCKER_CONFIG:-/tmp/apertus-ost-docker}"
task_build_args=()
task_run_args=()
if [[ -n "${HTTPS_PROXY:-}" ]]; then
    task_proxy_host=$(python -c 'import os,urllib.parse; print(urllib.parse.urlsplit(os.environ["HTTPS_PROXY"]).hostname)')
    task_proxy_ip=$(python -c 'import os,socket,urllib.parse; u=urllib.parse.urlsplit(os.environ["HTTPS_PROXY"]); print(socket.getaddrinfo(u.hostname,u.port,family=socket.AF_INET)[0][4][0])')
    task_build_args+=(--add-host "$task_proxy_host:$task_proxy_ip" --build-arg HTTP_PROXY --build-arg HTTPS_PROXY)
    task_run_args+=(--add-host "$task_proxy_host:$task_proxy_ip" -e HTTP_PROXY -e HTTPS_PROXY -e NO_PROXY)
fi
if [[ -n "${SSL_CERT_FILE:-}" && -f "$SSL_CERT_FILE" ]]; then
    task_build_args+=(--secret "id=cloud_ca,src=$SSL_CERT_FILE")
    task_run_args+=(-v "$SSL_CERT_FILE:/run/cloud_ca:ro" -e SSL_CERT_FILE=/run/cloud_ca)
fi
export DOCKER_BUILD_ARGS="${task_build_args[*]}"
export DOCKER_RUN_ARGS="${task_run_args[*]}"
exec make "$@"
