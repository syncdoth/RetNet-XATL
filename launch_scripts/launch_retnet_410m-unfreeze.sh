RANK=$1
NUM_NODE=${2-4}
MASTER_ADDR=${3-31}  # 172.27.14.14
MODE=${4-skip_reten}
RESUME_STEP=${5-"latest"}

export HF_DATASETS_OFFLINE=0
export TOKENIZERS_PARALLELISM=true

if [ $NUM_NODE -gt 1 ]; then
launcher="sudo -E env "PATH=$PATH" lightning run model \
    --node-rank=$RANK  \
    --main-address=$MASTER_ADDR \
    --accelerator=cuda \
    --devices=8 \
    --num-nodes=$NUM_NODE \
    scripts/train.py"
else
launcher="sudo -E env "PATH=$PATH" python scripts/train.py"
fi

# setup args
args="--devices 8 --num_of_nodes $NUM_NODE \
    --model_name RetNet-410m --random_seed 3407 \
    --micro_batch_size 16 --global_batch_size 256 \
    --tokenizer EleutherAI/pythia-410m \
    --log_train_loss_per_batch True \
    --dataset pile_dedup --activation_checkpointing False \
    --learning_rate 3e-4 --min_lr 3e-5 --max_step 145000 \
    --use_ddp True \
    --checkpoint_path checkpoints \
    --copy_exp True --freeze_copied_weights False"

# set config and copy_model_path from the $MODE
if [ $MODE = "skip_reten" ]; then
    args="$args --skip_reten True"
elif [ $MODE = "skip_reten-skip_oproj" ]; then
    args="$args --skip_reten True --skip_oproj True"
elif [ $MODE = "skip_reten-skip_oproj-skip_mlp" ]; then
    args="$args --skip_reten True --skip_oproj True --skip_mlp True"
fi
args="$args --copy_unfreeze_from $RESUME_STEP \
    --copy_ckpt_dir checkpoints/RetNet-410m-bs1024-pile_dedup-copy_exp-$MODE-freeze"


eval "$launcher $args"