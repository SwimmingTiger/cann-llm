#!/bin/bash
# dopt(pytorch) 三阶段 —— 按官方文档（CANN LLM 大语言模型解决方案.md）的 run.sh 配方 ✓
set -u
ROOT=~/q38/dopt_work
mkdir -p $ROOT
cat > $ROOT/config.yaml <<'YAML'
kd:
  enable: False
  loss: mse
  micro_batch_size: 1
  gradient_accumulation_steps: 1
  weight_decay: 0.0
  num_epochs: 1
  learning_rate: !!float 1e-4
  lr_scheduler_type: cosine
  trainable_keys:
    - quant_alpha
    - norm
  no_split_module_classes:
    - Qwen3_5DecoderLayer
    - Qwen3_5GatedDeltaNet
dataset:
  train_files:
  train_samples: 256
  ptq_samples: 256
extra_training_config:
  fp16: False
cutoff_len: 128
num_samples: 64
quant_param_2: False
embedding_separate: True
YAML
D=~/ddk/tools/tools_dopt/dopt_pytorch_py3
OUT=$ROOT/train_output
mkdir -p $OUT
cp $ROOT/config.yaml $OUT/ 2>/dev/null
[ -f $ROOT/dopt_config.json ] || cp ~/q38/dopt_out/dopt_config.json $ROOT/dopt_config.json
cd "$D" || exit 9
for st in stage1 stage2 stage3; do
  echo "=== $st $(date +%H:%M:%S) ==="
  PYTHONPATH=$D timeout 5400 ~/q38env/bin/python dopt/dopt_lm/opt_main.py \
    --model-path /home/hu60/q38 \
    --dopt-config $ROOT/dopt_config.json \
    --optimize-config $ROOT/config.yaml \
    --quant-stage $st --block-size 128 --output-dir $OUT 2>&1 | tail -5
  ls -l $OUT 2>/dev/null | tail -3
done
echo "ALL-STAGES-DONE"
