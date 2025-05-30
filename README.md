<font size=4><div align='center'>[[📄 Tech Report](pdf/Visual_Intelligence_Tech_Report.pdf)] [[🤗 checkpoints](https://huggingface.co/collections/Jakh0103/visual-intelligence-68398719ee0d35e8b553b5c9)]</div></font>


# 🛠️Set up
Copy `.env_example` to `.env` and fill in the environment variables.

### Intall the environment
For reproducable environments, we use a conda compatible tool called Pixi. If you don't have Pixi installed, run:
```
curl -fsSL https://pixi.sh/install.sh | sh
```
You can start an interactive job with
```
Sinteract -c10 -t 1:0:0 -m 32G
```
Then cd to this repository and install the environment with:
```
pixi install
```
You can enter it with `pixi shell` or always prepend `pixi run` to your commands.

### Running on Izar
For a 3 hour session with 32GB of memory and one GPU, run the following command:
```
Sinteract -c10 -g gpu:1 -t 3:0:0 -m 32G
```

You can verify cuda works as expected:
```
pixi run python -c "import torch; print(torch.cuda.is_available())"
```

# 💽Download dataset
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

We modify the dataset format inside [./src/open-r1-multimodal/src/open_r1/grpo.py](./src/open-r1-multimodal/src/open_r1/grpo.py) in the lines 127-131 to the following:
```
{
    'problem': 'Is the following statement true: The cake is next to the person.',
    'solution': 'True'
}
```

# 💪🏻General GRPO/SFT Training
The following script is to run the GRPO training [./src/open-r1-multimodal/src/open_r1/grpo.py](./src/open-r1-multimodal/src/open_r1/grpo.py). You can also go through the code to understand how it is working. More important parts are how the reward functions are being passed and how the dataset is being formatted. [./scripts/run_grpo_lora.sh](./scripts/run_grpo_lora.sh) is a bash script to run the GRPO using LoRA and all other hyperparameters.

sbatch script to submit a job to run grpo:


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
pixi run bash scripts/evaluate.sh           # evaluate
```

### Reward functions
The rewards functions are declared inside [./src/open-r1-multimodal/src/open_r1/rewards/rewards.py](./src/open-r1-multimodal/src/open_r1/rewards/rewards.py). Right now I am using `accuracy_reward` and `format_reward`. Any other rewards can be declared here, and called from the `grpo.py` script.

# 💻Projects break-down
## Bias Mitigation Datasets
All code related to creating biased datasets is in `notebooks/Bias Project/`.

### SmolVLM Adaptation
The original code for VLM-R1 is compatible with Qwen and InternVL. We had to create a separate module [vlm_modules/smolvlm_module.py](src/open-r1-multimodal/src/open_r1/vlm_modules/smolvlm_module.py) to adapt the code for SmolVLM. Moreover, [Idefics3 model](https://github.com/huggingface/transformers/blob/main/src/transformers/models/idefics3/processing_idefics3.py) (contains [the conditional generator](https://github.com/huggingface/transformers/blob/main/src/transformers/models/idefics3/modeling_idefics3.py#L861) for SmolVLM) does not pass the **image tokens** from the cache during generation. This issue is, according to us, an internal issue (*) of Idefics3ForConditionalGeneration of the Transformers library. We experimented with modifying Idefics3ForConditionalGeneration by re-adding manually image tokens on the fly, but the results were inconclusive. Therefore, we disabled caching for SmolVLM during generation, which solved the issue.

SmolVLM is compatible with the usual SFT training.

Moreover, SmolVLM-500M-Instruct had a hard time outputting True/False answers instead of Yes/No. Therefore, we choose to adapt the evaluation to accept both Yes/No and True/False answers, instead of trying to force the model to output True/False and have a very poor accuracy.

(*) The issue arises in the `inputs_merger` function of [Idefics3 model](https://github.com/huggingface/transformers/blob/main/src/transformers/models/idefics3/processing_idefics3.py). In practice, the function does:
```python
special_image_token_mask = input_ids == self.image_token_id
new_inputs_embeds = inputs_embeds.clone()
image_hidden_states = image_hidden_states.view(-1, image_hidden_states.shape[-1])
image_hidden_states = image_hidden_states.to(inputs_embeds.device, inputs_embeds.dtype)
new_inputs_embeds[special_image_token_mask] = image_hidden_states
return new_inputs_embeds
```
The error occurs because `special_image_token_mask` is empty when generating from the cache. In practice, we observe that during the first forward pass, the `special_image_token_mask` is correctly filled with the image tokens, but afterwards, during the generation from cache, it is empty. This is, according to us, due to the way the cache is handled in Idefics3ForConditionalGeneration, which does not pass the image tokens correctly when generating from cache. Also, we have verified that they are correctly passed from our code. When disabling caching, the `special_image_token_mask` is not empty and the image tokens are correctly passed to the model at each step.



## Soft Prompt Tunning
All code related to soft prompt tunning is in `notebooks/prompt_tunning/`.

To tune a soft prompt to generate the answer directly, run the `notebooks/prompt_tunning/scripts/run_softprompt.sh`. For the reasoning soft prompt tunnig, you need to first create the reasoning dataset with `notebooks/prompt_tunning/dataset_generation.ipynb`. Then you can run the `notebooks/prompt_tunning/scripts/run_softprompt_output.sh`.

## Grounded Reasoning
To train for reasoning on [DrivingVQA](https://huggingface.co/datasets/EPFL-DrivingVQA/DrivingVQA), first download the dataset, and run the following scripts.

```
scripts/prepare_sft.sh      # prepares the DrivingVQA dataset for supervised fine-tuning with reasoning
scripts/run_sft_lora.sh     # runs SFT training
scripts/merge_lora.sh       # merges the LoRA model to the base model, and saves to `output_path`
scripts/run_grpo_lora.sh    # runs GRPO training starting from the model at `output_path` from above
scripts/evaluate.sh         # evaluates the models
```



# 🤝Acknowledgements
We would like to express our gratitude to [VLM-R1](https://github.com/om-ai-lab/VLM-R1) for providing open-source resources that contributed to the development of this project.