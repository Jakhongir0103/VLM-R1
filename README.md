# Set up
## Venv
1. Start an interactive job on scitas.
2. Create a python virtual environment.
3. run `bash setup.sh` to install the dependencies.

2 and 3 are done only once. The next time we submit a job, we can just activate the virtual environment that has already be created with all the dependencies (`source <venv_name>bin/activate`)
## Pixi
```
pixi install
```
# Running izar
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

# GRPO Training
The following script is to run the GRPO training [./src/open-r1-multimodal/src/open_r1/grpo.py](./src/open-r1-multimodal/src/open_r1/grpo.py). You can also go through the code to understand how it is working. More important parts are how the reward functions are being passed and how the dataset is being formatted. [./scripts/run_grpo_lora.sh](./scripts/run_grpo_lora.sh) is a bash script to run the GRPO using LoRA and all other hyperparameters.

sbatch script to submit a job to run grpo:

#!/bin/bash

```
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 8
#SBATCH --mem 16G
#SBATCH --gres=gpu:1
#SBATCH --time 12:00:00                                     # maximum time limit is 12h. We need to rerun the jobs every 12 hours.
#SBATCH --output=./logs/slurm-%j.out
#SBATCH --account=cs-503
#SBATCH --qos=cs-503

source /home/saydalie/venvs/course_py-3.10/bin/activate     # change this to where your python envrionment is located
cd /home/saydalie/project/VLM-R1/scripts                    # change this to where your repo is located
bash run_grpo_lora.sh
```

# Reward functions
The rewards functions are declared inside [./src/open-r1-multimodal/src/open_r1/rewards/rewards.py](./src/open-r1-multimodal/src/open_r1/rewards/rewards.py). Right now I am using `accuracy_reward` and `format_reward`. Any other rewards can be declared here, and called from the `grpo.py` script.