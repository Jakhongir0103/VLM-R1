# Set up
Copy `.env_example` to `.env` and fill in the environment variables.

## Intall the environment
If you don't have pixi installed, run:
```
curl -fsSL https://pixi.sh/install.sh | sh
```
Start an interactive job
```
Sinteract -c10 -g gpu:1 -t 1:0:0 -m 32G
```
Then cd to this repository and install the environment with:
```
pixi install
```
You can enter it with `pixi shell` or always prepend `pixi run` to your commands.
Now you can exit the interactive job.

# Running on Izar
For a 3 hour session with 32GB of memory and one GPU, run the following command:
```
Sinteract -c10 -g gpu:1 -t 3:0:0 -m 32G
```

You can verify cuda works as expected:
```
pixi run python -c "import torch; print(torch.cuda.is_available())"
```

# Download dataset
Run [./notebooks/create_dataset.ipynb](./notebooks/create_dataset.ipynb) to download the Visual Spatial Reasoning dataset along with the images. The final dataset has the following columns: [image_path, caption, label, relation, subj, obj]
e.g.
```
{
    'image_path': '/home/saydalie/project/VLM-R1/data/images/vsr/000000558388.jpg',
    'caption': 'The cake is next to the person.',
    'label': 1,
    'relation': 'next to',
    'subj': 'cake',
    'obj': 'person'
}
```

I modify the dataset format inside [./src/open-r1-multimodal/src/open_r1/grpo.py](./src/open-r1-multimodal/src/open_r1/grpo.py) in the lines 127-131 to the following:
```
{
    'problem': 'Is the following statement true: The cake is next to the person.',
    'solution': 'True'
}
```

# GRPO/SFT Training
The following script is to run the GRPO training [./src/open-r1-multimodal/src/open_r1/grpo.py](./src/open-r1-multimodal/src/open_r1/grpo.py). You can also go through the code to understand how it is working. More important parts are how the reward functions are being passed and how the dataset is being formatted. [./scripts/run_grpo_lora.sh](./scripts/run_grpo_lora.sh) is a bash script to run the GRPO using LoRA and all other hyperparameters.

sbatch script to submit a job to run grpo:

#!/bin/bash

```
#!/bin/bash
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 8
#SBATCH --mem 16G
#SBATCH --gres=gpu:1
#SBATCH --time 12:00:00                     # maximum time limit is 12h. We need to rerun the jobs every 12 hours.
#SBATCH --output=./logs/slurm-%j.out
#SBATCH --account=cs-503
#SBATCH --qos=cs-503

cd $HOME/VLM-R1/
echo $PWD
source ./.env
pixi run bash scripts/run_grpo_lora.sh      # to run GRPO
pixi run bash scripts/run_sft_lora.sh       # to run SFT
```

# Reward functions
The rewards functions are declared inside [./src/open-r1-multimodal/src/open_r1/rewards/rewards.py](./src/open-r1-multimodal/src/open_r1/rewards/rewards.py). Right now I am using `accuracy_reward` and `format_reward`. Any other rewards can be declared here, and called from the `grpo.py` script.
