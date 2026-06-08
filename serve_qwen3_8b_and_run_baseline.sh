#!/bin/bash
#SBATCH --partition=IAI_SLURM_3090
#SBATCH --job-name=qwen3-baseline
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
MAX_MODEL_LEN="${MAX_MODEL_LEN:-9182}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-1}"
MAX_CONCURRENT_TASKS="${MAX_CONCURRENT_TASKS:-1}"
LOG_ROOT="${LOG_ROOT:-experiments/baseline_vllm_qwen3_8b}"
MATH_PATH="${MATH_PATH:-data/datasets/math_validate.jsonl}"
HUMANEVAL_PATH="${HUMANEVAL_PATH:-data/datasets/humaneval_validate.jsonl}"

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

if [[ ! -d "${MODEL_PATH}" ]]; then
    echo "Missing local model directory: ${MODEL_PATH}" >&2
    exit 1
fi

if [[ ! -f "${MATH_PATH}" ]]; then
    echo "Missing offline dataset file: ${MATH_PATH}" >&2
    exit 1
fi

if [[ ! -f "${HUMANEVAL_PATH}" ]]; then
    echo "Missing offline dataset file: ${HUMANEVAL_PATH}" >&2
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
    extra_body:
      chat_template_kwargs:
        enable_thinking: true
EOF

mkdir -p logs "${LOG_ROOT}"
VLLM_LOG="${REPO_DIR}/logs/vllm_${SERVED_MODEL_NAME}_baseline.log"
touch "${VLLM_LOG}"
echo "vLLM log: ${VLLM_LOG}"

vllm serve "${MODEL_PATH}" \
    --served-model-name "${SERVED_MODEL_NAME}" \
    --host 127.0.0.1 \
    --port "${PORT}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --max-num-seqs "${MAX_NUM_SEQS}" \
    --trust-remote-code \
    > "${VLLM_LOG}" 2>&1 &

VLLM_PID=$!
echo "vLLM PID: ${VLLM_PID}"

echo "Waiting for vLLM server..."
for i in {1..180}; do
    if ! kill -0 "${VLLM_PID}" 2>/dev/null; then
        echo "vLLM exited before becoming ready. Last log lines:" >&2
        tail -n 80 "${VLLM_LOG}" >&2 || true
        exit 1
    fi
    if curl -fs "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1; then
        echo "vLLM is ready."
        break
    fi
    sleep 5
done

curl -fsS "http://127.0.0.1:${PORT}/v1/models"
echo

echo "Starting Qwen3-8B single-base-model CoT baseline..."
CUDA_VISIBLE_DEVICES="" \
python -u scripts/run_baseline_env.py \
    --models "${SERVED_MODEL_NAME}" \
    --math_path "${MATH_PATH}" \
    --humaneval_path "${HUMANEVAL_PATH}" \
    --log_root "${LOG_ROOT}" \
    --max_concurrent_tasks "${MAX_CONCURRENT_TASKS}"
