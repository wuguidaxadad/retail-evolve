# Source this file from the project root before using project-local container tools.
RETAIL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)"
export PATH="$RETAIL_ROOT/.runtime/tools:$RETAIL_ROOT/.runtime/tools/docker:$RETAIL_ROOT/.runtime/tools/lima/bin:$PATH"
export RETAIL_VM_HOME="${RETAIL_VM_HOME:-$HOME/Documents/Codex/.retail-vm}"
export COLIMA_HOME="$RETAIL_VM_HOME/colima"
export LIMA_HOME="$RETAIL_VM_HOME/lima"
export DOCKER_CONFIG="$RETAIL_ROOT/.runtime/docker-config"
export KUBECONFIG="$RETAIL_ROOT/.runtime/kubeconfig"
