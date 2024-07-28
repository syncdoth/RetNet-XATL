BUILD_XFORMERS_FROM_SOURCE=${1-0}
BUILD_FLASH_ATTN_FROM_SOURCE=${2-1}

eval "$(conda shell.bash hook)" && source ~/miniconda3/etc/profile.d/conda.sh && conda activate lit_llm

conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia -y

pip uninstall ninja -y && pip install ninja -U
if [ $BUILD_XFORMERS_FROM_SOURCE = 1 ]; then
    pip install -v -U git+https://github.com/facebookresearch/xformers.git@main#egg=xformers
else
    pip install xformers --index-url https://download.pytorch.org/whl/cu121
fi

pip install packaging
cd ~/installs
git clone https://github.com/Dao-AILab/flash-attention
cd flash-attention

if [ $BUILD_FLASH_ATTN_FROM_SOURCE = 1 ]; then
    python setup.py install
else
    pip install flash-attn --no-build-isolation
fi
# still need to build c kernels
cd csrc/rotary && pip install .
cd ../layer_norm && pip install .
cd ../xentropy && pip install .
cd ../fused_dense_lib && pip install .
cd ../.. && rm -rf flash-attention

cd ~/lit_llm_train
pip install -r requirements.txt
