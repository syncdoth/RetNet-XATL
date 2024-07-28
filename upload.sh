iter=${1-060000}
do_upload=${2-True}
model=${3-"RetNet-410m"}
extra_config=${4-""}

default_config="bs1024-pile_dedup"
full_config="${default_config}${extra_config}"

CKPT_DIR=checkpoints
sudo -E env "PATH=$PATH" python scripts/convert_and_upload.py \
    --checkpoint_name iter-$iter-ckpt.pth \
    --out_dir $CKPT_DIR/$model-$full_config \
    --model_name $model --do_upload $do_upload
