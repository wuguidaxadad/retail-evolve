#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
case "${1:-}" in
  deploy)
    kubectl apply -f deploy/k8s/namespace.yaml
    kubectl -n retail-evolve get secret retail-secrets >/dev/null
    kubectl apply -f deploy/k8s/config.yaml -f deploy/k8s/storage.yaml
    kubectl -n retail-evolve rollout status statefulset/mysql --timeout=300s
    kubectl -n retail-evolve rollout status statefulset/redis --timeout=300s
    kubectl -n retail-evolve delete job retail-bootstrap --ignore-not-found --wait=true
    kubectl apply -f deploy/k8s/bootstrap.yaml
    kubectl -n retail-evolve wait --for=condition=complete job/retail-bootstrap --timeout=300s
    kubectl apply -f deploy/k8s/api.yaml -f deploy/k8s/worker.yaml
    kubectl -n retail-evolve rollout status deployment/api --timeout=300s
    kubectl -n retail-evolve rollout status deployment/worker --timeout=300s
    ;;
  status) kubectl -n retail-evolve get pods,svc,pvc,jobs ;;
  open) kubectl -n retail-evolve port-forward svc/api 8766:8765 --address 127.0.0.1 ;;
  *) echo 'Usage: bash scripts/k8s.sh {deploy|status|open}'; exit 2 ;;
esac
