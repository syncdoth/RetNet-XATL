RANK=$1
NUM_NODE=${2-4}
MASTER_ADDR=${3-31}  # 172.27.14.14

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

args="--devices 8 --num_of_nodes $NUM_NODE \
    --model_name pythia-410m --random_seed 3407 \
    --micro_batch_size 16 --global_batch_size 256 \
    --tokenizer EleutherAI/pythia-410m \
    --log_train_loss_per_batch True \
    --checkpoint_path checkpoints \
    --dataset pile_dedup --activation_checkpointing False \
    --learning_rate 3e-4 --min_lr 3e-5 --max_step 145000 \
    --use_ddp True"

eval "$launcher $args"