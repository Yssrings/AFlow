#!/bin/bash
#SBATCH --partition=IAI_SLURM_3090
#SBATCH --job-name=qwen3-aflow
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --qos=singlegpu
#SBATCH --cpus-per-task=10
#SBATCH --time 72:00:00

set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/AFlow}"
CONDA_HOME="${CONDA_HOME:-$HOME/miniconda3}"
CONDA_ENV="${CONDA_ENV:-aflow}"
MODEL_PATH="${MODEL_PATH:-${REPO_DIR}/../SimpleMem/weights/Qwen3-8B}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-Qwen3-8B}"
PORT="${PORT:-8000}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-16}"

AFLOW_DATASET="${AFLOW_DATASET:-HumanEval}"
AFLOW_SAMPLE="${AFLOW_SAMPLE:-2}"
AFLOW_INITIAL_ROUND="${AFLOW_INITIAL_ROUND:-1}"
AFLOW_MAX_ROUNDS="${AFLOW_MAX_ROUNDS:-3}"
AFLOW_VALIDATION_ROUNDS="${AFLOW_VALIDATION_ROUNDS:-1}"
AFLOW_OPTIMIZED_PATH="${AFLOW_OPTIMIZED_PATH:-workspace_vllm_qwen3_8b}"

cd "${REPO_DIR}"
source "${CONDA_HOME}/bin/activate" "${CONDA_ENV}"

export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export VLLM_NO_USAGE_STATS=1
export TOKENIZERS_PARALLELISM=false

echo "Running on node: $(hostname)"
echo "Repo: ${REPO_DIR}"
echo "Model path: ${MODEL_PATH}"
echo "Served model name: ${SERVED_MODEL_NAME}"
echo "AFlow dataset: ${AFLOW_DATASET}"

if [[ ! -d "${MODEL_PATH}" ]]; then
    echo "Missing local model directory: ${MODEL_PATH}" >&2
    exit 1
fi

if [[ ! -d "data/datasets" ]]; then
    echo "Missing offline data/datasets directory. Pre-stage AFlow datasets before submitting this job." >&2
    exit 1
fi

CONFIG_PATH="config/config2.yaml"
CONFIG_BACKUP=""
if [[ -f "${CONFIG_PATH}" ]]; then
    CONFIG_BACKUP="${CONFIG_PATH}.slurm.${SLURM_JOB_ID:-manual}.bak"
    cp "${CONFIG_PATH}" "${CONFIG_BACKUP}"
fi

cleanup() {
    echo "Stopping vLLM..."
    if [[ -n "${VLLM_PID:-}" ]]; then
        kill "${VLLM_PID}" 2>/dev/null || true
        wait "${VLLM_PID}" 2>/dev/null || true
    fi
    if [[ -n "${CONFIG_BACKUP}" && -f "${CONFIG_BACKUP}" ]]; then
        mv "${CONFIG_BACKUP}" "${CONFIG_PATH}"
    fi
}
trap cleanup EXIT

cat > "${CONFIG_PATH}" <<EOF
models:
  "${SERVED_MODEL_NAME}":
    api_type: "openai"
    base_url: "http://127.0.0.1:${PORT}/v1"
    api_key: "EMPTY"
    temperature: 0.6
    top_p: 0.95
EOF

mkdir -p logs "${AFLOW_OPTIMIZED_PATH}"

vllm serve "${MODEL_PATH}" \
    --served-model-name "${SERVED_MODEL_NAME}" \
    --host 127.0.0.1 \
    --port "${PORT}" \
    --dtype float16 \
    --max-model-len "${MAX_MODEL_LEN}" \
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
    --max-num-seqs "${MAX_NUM_SEQS}" \
    --trust-remote-code \
    --default-chat-template-kwargs '{"enable_thinking": true}' \
    > "logs/vllm_${SERVED_MODEL_NAME}_aflow.log" 2>&1 &

VLLM_PID=$!
echo "vLLM PID: ${VLLM_PID}"

echo "Waiting for vLLM server..."
for i in {1..180}; do
    if ! kill -0 "${VLLM_PID}" 2>/dev/null; then
        echo "vLLM exited before becoming ready. Last log lines:" >&2
        tail -n 80 "logs/vllm_${SERVED_MODEL_NAME}_aflow.log" >&2 || true
        exit 1
    fi
    if curl -fsS "http://127.0.0.1:${PORT}/v1/models" >/dev/null; then
        echo "vLLM is ready."
        break
    fi
    sleep 5
done

curl -fsS "http://127.0.0.1:${PORT}/v1/models"
echo

echo "Starting AFlow optimization and validation evaluation..."
CUDA_VISIBLE_DEVICES="" \
python -u run.py \
    --dataset "${AFLOW_DATASET}" \
    --sample "${AFLOW_SAMPLE}" \
    --optimized_path "${AFLOW_OPTIMIZED_PATH}" \
    --initial_round "${AFLOW_INITIAL_ROUND}" \
    --max_rounds "${AFLOW_MAX_ROUNDS}" \
    --validation_rounds "${AFLOW_VALIDATION_ROUNDS}" \
    --if_force_download false \
    --opt_model_name "${SERVED_MODEL_NAME}" \
    --exec_model_name "${SERVED_MODEL_NAME}"
