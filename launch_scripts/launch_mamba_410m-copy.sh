RANK=$1
NUM_NODE=${2-4}
MASTER_ADDR=${3-31}  # 172.27.14.14
FREEZE=${4-nofreeze}
HYBRID=${5-False}
MODE=${6-skip_reten}
RESUME=${7-False}

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

if [ $HYBRID = "True" ]; then
    model_name="StripedMamba-410m-Hybrid"
else
    model_name="StripedMamba-410m-expand2"
fi

args="--devices 8 --num_of_nodes $NUM_NODE \
    --model_name $model_name --random_seed 3407 \
    --micro_batch_size 16 --global_batch_size 256 \
    --tokenizer EleutherAI/pythia-410m \
    --log_train_loss_per_batch True \
    --dataset pile_dedup --activation_checkpointing False \
    --learning_rate 3e-4 --min_lr 3e-5 --max_step 145000 \
    --checkpoint_path checkpoints \
    --use_ddp True --resume $RESUME --copy_exp True"


# set config and copy_model_path from the $MODE
if [ $MODE = "skip_reten" ]; then
    args="$args --skip_reten True"
elif [ $MODE = "skip_reten-skip_oproj" ]; then
    args="$args --skip_reten True --skip_oproj True"
elif [ $MODE = "skip_reten-skip_oproj-skip_mlp" ]; then
    args="$args --skip_reten True --skip_oproj True --skip_mlp True"
fi
if [ $HYBRID = "True" ]; then
    MODE="$MODE-Hybrid"
fi

args="$args --copy_model_path checkpoints/pythia-410m-deduped-200B/retnet/lit_model-$MODE.bin"

# control freezing of copied weights
if [ $FREEZE = "freeze" ]; then
    args="$args --freeze_copied_weights True"
else
    echo "not freezing the copied weights."
fi

eval "$launcher $args"
