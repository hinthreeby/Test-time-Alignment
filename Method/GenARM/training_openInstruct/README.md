# Installation
The following is for training Tulu2 model, based on the [open-instruct](https://github.com/allenai/open-instruct) codebase.  

First, installing in a *bare environment* (no Cuda image).

```
conda create --name tulu python=3.10
conda activate tulu
```
Before installing, if not in a Docker container with NVCC installed, you should run:
```
conda install cuda-nvcc=12.1 -c nvidia
```
Then, install `torch==2.3.0` by

```
conda install pytorch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 pytorch-cuda=12.1 -c pytorch -c nvidia
```

To run training, evaluation, or inference for our finetuned models, you need to install the required packages by running the following command (after installing pytorch):

```bash
pip install -r requirements.txt
```

You also need wandb for logging. To login (if you have not already done so), fine you wandb API key on the wandb website and then

```
wandb login
```

# Autoregressive RM full-finetuning

To fully fine-tune `allenai/tulu-2-7b` for Autoregressive Reward Model:
```
bash scripts/arm_train_with_accelerate.sh
```
You need 80G GPU for this task. 

Also, you can train a DPO model using:

```
bash scripts/dpo_train_with_accelerate.sh 
```