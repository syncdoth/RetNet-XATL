## Pretrain TinyLlama

### Installation
We expect you have CUDA 12.1 installed.
#### Install Pytorch & xformers & flash-attn.
```bash
conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia -y

# build from source:
# pip uninstall ninja -y && pip install ninja -U
# pip install -v -U git+https://github.com/facebookresearch/xformers.git@main#egg=xformers
pip install xformers --index-url https://download.pytorch.org/whl/cu121


cd ~/installs
git clone https://github.com/Dao-AILab/flash-attention
cd flash-attention

# install flash-attn
pip install packaging
pip install flash-attn --no-build-isolation
# depending on your environment, you may want to do the following:
# CUDA_HOME=/usr/local/cuda-12.1 pip install flash-attn --no-build-isolation
# build from source
# python setup.py install

# You still need to build the c-kernels
cd csrc/rotary && pip install .
cd ../layer_norm && pip install .
cd ../xentropy && pip install .
cd ../fused_dense_lib && pip install .
cd ../.. && rm -rf flash-attention
```

#### Install Remaining Dependencies
```
pip install -r requirements.txt
```
to install other dependencies.

Then you are ready to go 🎉!

#### For eval

To run zero-shot evaluations of models (corresponding to Table 3 of the paper), we use the lm-evaluation-harness library.

1. Pull the lm-evaluation-harness repo by git submodule update --init --recursive. We use the big-refactor branch.
2. Install lm-evaluation-harness:

```bash
pip install -e 3rdparty/lm-evaluation-harness
# On Python 3.10 you might need to manually install the latest version of promptsource:
pip install git+https://github.com/bigscience-workshop/promptsource.git.
```

Run evaluation with:
```bash
python evals/lm_harness_eval.py --model retnet --model_args pretrained=NucleusAI/retnet-410m --tasks lambada_openai,hellaswag,piqa,arc_easy,arc_challenge,winogrande --device cuda --batch_size 64
python evals/lm_harness_eval.py --model hf --model_args pretrained=EleutherAI/pythia-410m --tasks lambada_openai,hellaswag,piqa,arc_easy,arc_challenge,winogrande --device cuda --batch_size 64
```
To match the results on the mamba-2.8b-slimpj:

```bash
python evals/lm_harness_eval.py --model mamba --model_args pretrained=NucleusAI/retnet-410m --tasks boolq,piqa,hellaswag,winogrande,arc_easy,arc_challenge,openbookqa,race,truthfulqa_mc2 --device cuda --batch_size 64
python evals/lm_harness_eval.py --model mamba --model_args pretrained=NucleusAI/retnet-410m --tasks mmlu --num_fewshot 5 --device cuda --batch_size 64
```

### Data Preparation

#### Download Datasets
Download the Slimpajama and Starcoderdata datasets to your chosen directory.
```bash
cd /path/to/dataset
git lfs install
git clone https://huggingface.co/datasets/cerebras/SlimPajama-627B
git clone https://huggingface.co/datasets/bigcode/starcoderdata
```
The SlimPajama dataset eats 893GB diskspace and the starcoderdata takes 290GB.

#### Tokenize data
Use the provided scripts to tokenize the datasets and divide them into chunks.
```bash
python scripts/prepare_starcoder.py --source_path /path/to/starcoderdata/ --tokenizer_path data/llama --destination_path data/slim_star_combined --split train --percentage 1.0
python scripts/prepare_slimpajama.py --source_path /path/to/SlimPajama --tokenizer_path data/llama  --destination_path data/slim_star_combined --split validation --percentage 1.0
python scripts/prepare_slimpajama.py --source_path /path/to/SlimPajama --tokenizer_path data/llama  --destination_path data/slim_star_combined --split train --percentage 1.0
```
The processed data will take 1.8T storage.

### Pretraining
If your setup comprises two nodes, each with 8 GPUs, you can initiate pretraining with the following commands:

On node 1:
```
lightning run model \
    --node-rank=0  \
    --main-address=172.16.101.5 \
    --accelerator=cuda \
    --devices=8 \
    --num-nodes=2 \
    scripts/train.py --devices 8 --train_data_dir data/slim_star  --val_data_dir data/slim_star
```
On node 2:
```
lightning run model \
    --node-rank=1  \
    --main-address=172.16.101.5 \
    --accelerator=cuda \
    --devices=8 \
    --num-nodes=2 \
    scripts/train.py --devices 8 --train_data_dir data/slim_star   --val_data_dir data/slim_star
```
You can follow [these instructions](https://lightning.ai/docs/fabric/stable/guide/multi_node/slurm.html) if you have a slurm cluster.

