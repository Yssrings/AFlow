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
MAX_MODEL_LEN="${MAX_MODEL_LEN:-6000}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.75}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-4}"
MAX_TOKENS="${MAX_TOKENS:-2048}"

AFLOW_DATASETS="${AFLOW_DATASETS:-${AFLOW_DATASET:-HumanEval,MATH}}"
AFLOW_SAMPLE="${AFLOW_SAMPLE:-2}"
AFLOW_INITIAL_ROUND="${AFLOW_INITIAL_ROUND:-1}"
AFLOW_MAX_ROUNDS="${AFLOW_MAX_ROUNDS:-30}"
AFLOW_VALIDATION_ROUNDS="${AFLOW_VALIDATION_ROUNDS:-1}"
AFLOW_MAX_CONCURRENT_TASKS="${AFLOW_MAX_CONCURRENT_TASKS:-4}"
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
echo "AFlow datasets: ${AFLOW_DATASETS}"

if [[ ! -d "${MODEL_PATH}" ]]; then
    echo "Missing local model directory: ${MODEL_PATH}" >&2
    exit 1
fi

if [[ ! -d "data/datasets" ]]; then
    echo "Missing offline data/datasets directory. Pre-stage AFlow datasets before submitting this job." >&2
    exit 1
fi

seed_aflow_workflow() {
    local dataset="$1"
    local source_workflow_dir="${SOURCE_WORKFLOW_DIR:-workspace/${dataset}/workflows}"
    local target_workflow_dir="${AFLOW_OPTIMIZED_PATH}/${dataset}/workflows"

    if [[ -f "${target_workflow_dir}/round_1/graph.py" && -f "${target_workflow_dir}/template/operator.json" ]]; then
        return
    fi

    echo "Seeding initial AFlow workflow for ${dataset} from ${source_workflow_dir}"
    if [[ ! -f "${source_workflow_dir}/round_1/graph.py" ]]; then
        echo "Missing source workflow seed: ${source_workflow_dir}/round_1/graph.py" >&2
        exit 1
    fi
    if [[ ! -f "${source_workflow_dir}/template/operator.json" ]]; then
        echo "Missing source workflow template: ${source_workflow_dir}/template/operator.json" >&2
        exit 1
    fi

    mkdir -p "${target_workflow_dir}/round_1" "${target_workflow_dir}/template"
    cp -a "${source_workflow_dir}/round_1/." "${target_workflow_dir}/round_1/"
    cp -a "${source_workflow_dir}/template/." "${target_workflow_dir}/template/"
    touch "${AFLOW_OPTIMIZED_PATH}/__init__.py"
    mkdir -p "${AFLOW_OPTIMIZED_PATH}/${dataset}"
    touch "${AFLOW_OPTIMIZED_PATH}/${dataset}/__init__.py"
    touch "${target_workflow_dir}/__init__.py"

    if [[ -f "${target_workflow_dir}/results.json" ]]; then
        mv "${target_workflow_dir}/results.json" "${target_workflow_dir}/results.json.bak.${SLURM_JOB_ID:-manual}"
    fi
}

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
    max_tokens: ${MAX_TOKENS}
    extra_body:
      chat_template_kwargs:
        enable_thinking: true
EOF

mkdir -p logs "${AFLOW_OPTIMIZED_PATH}"
VLLM_LOG="${REPO_DIR}/logs/vllm_${SERVED_MODEL_NAME}_aflow.log"
touch "${VLLM_LOG}"
echo "vLLM log: ${VLLM_LOG}"

vllm serve "${MODEL_PATH}" \
    --served-model-name "${SERVED_MODEL_NAME}" \
    --host 127.0.0.1 \
    --port "${PORT}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
    --max-num-seqs "${MAX_NUM_SEQS}" \
    --enforce-eager \
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

IFS=',' read -r -a AFLOW_DATASET_LIST <<< "${AFLOW_DATASETS}"

for dataset in "${AFLOW_DATASET_LIST[@]}"; do
    dataset="$(echo "${dataset}" | xargs)"
    if [[ -z "${dataset}" ]]; then
        continue
    fi

    seed_aflow_workflow "${dataset}"

    echo "Starting AFlow optimization and validation evaluation for ${dataset}..."
    CUDA_VISIBLE_DEVICES="" \
    python -u run.py \
        --dataset "${dataset}" \
        --sample "${AFLOW_SAMPLE}" \
        --optimized_path "${AFLOW_OPTIMIZED_PATH}" \
        --initial_round "${AFLOW_INITIAL_ROUND}" \
        --max_rounds "${AFLOW_MAX_ROUNDS}" \
        --validation_rounds "${AFLOW_VALIDATION_ROUNDS}" \
        --max_concurrent_tasks "${AFLOW_MAX_CONCURRENT_TASKS}" \
        --if_force_download false \
        --opt_model_name "${SERVED_MODEL_NAME}" \
        --exec_model_name "${SERVED_MODEL_NAME}"
done

echo "AFlow optimization finished for: ${AFLOW_DATASETS}"
