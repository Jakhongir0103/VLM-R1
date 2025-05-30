<font size=4><div align='center'>[[📄 Tech Report](pdf/Visual_Intelligence_Tech_Report.pdf)] [[🤗 checkpoints](https://huggingface.co/collections/Jakh0103/visual-intelligence-68398719ee0d35e8b553b5c9)]</div></font>

### 📝 Abstract

*We investigate the use of reasoning through Group Relative Policy Optimization (GRPO) to enhance the visual question answering task in vision-language models (VLMs). Our study evaluates five aspects: reasoning-answer alignment, grounded reasoning with bounding boxes, generalization from synthetic data, bias mitigation, and prompt-based reasoning induction. GRPO improves performance and generalization, particularly for out-of-domain datasets when structured rewards are used. However, reasoning alignment remains imperfect, and prompt tuning presents challenges. Our results highlight both the promise and limitations of reinforcement learning for advancing visual reasoning capabilities in VLMs.*

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

## LLM-as-a-judge to evaluate alignment
All code related to creating biased datasets is in `alignment-calculation_LLM-as-a-judge/`.



for running the LLM-as-a-judge script `alignment-calculation_LLM-as-a-judge/taxonomy-prompt/LLM-as-a-judge_o4mini` you need to set the environmental variable in the terminal.

```bash
export OPENAI_API_KEY="your-api-key-here"

cd alignment-calculation_LLM-as-a-judge/taxonomy-prompt/
```

for the judge evaluation, run the script and the results will be stored inside `alignment_checks_TEST_o4-mini.json`, `alignment_checks_TRAIN_o4-mini.json`, `alignment_checks_VALIDATION_o4-mini.json`

```
python3 LLM-as-a-judge_o4mini.py
```

after the results are present, you can print the accuracy/alignment table used in the report. with the following. Also, the results are printed as a comment at the end of the script.

```
python3 alignment-statistics.py
```

for completness there is also `alignment-calculation_LLM-as-a-judge/simple-prompt/` folder containing the experiments with a basic prompt (with both gpt-4o and o4-mini) and the folder `alignment-calculation_LLM-as-a-judge/simple-prompt/human_labeling` containing the human evaluated samples.

in case there are any problem with the paths every script is organized such as you can change path or the split you want to judge/get statistics from.

```
# ─ Config ─
SPLIT       = "TRAIN" # switch to VALIDATION or TEST (upper case)
BASE_PATH   = Path(f"../responses/base/generated_responses_train.json")
GRPO_PATH   = Path(f"../responses/grpo/generated_responses_train.json")
ALIGN_PATH  = Path(f"alignment_checks_{SPLIT}_o4-mini.json")
```
the folder also contains `alignment-calculation_LLM-as-a-judge/responses/(grpo or base)` the original responses generated  by the models, and passed to the different judge scripts.

## Simulation to Real
To download the Rel3D dataset, check [princeton-vl/Rel3D](https://github.com/princeton-vl/Rel3D).
To download the SpatialSense dataset, check [princeton-vl/SpatialSense](https://github.com/princeton-vl/SpatialSense).

Once the datasets are saved on the disk, we can format the datasets and save them as a Dataset type using the notebooks `rel3d_dataset.ipynb` and `spatialsense_dataset.ipynb`. The SpatialSense notebook should be put in the project folder of the repository of [princeton-vl/SpatialSense](https://github.com/princeton-vl/SpatialSense).

To train the base model using GRPO and SFT, see the following files [grpo_sim2real.py](https://github.com/Jakhongir0103/VLM-R1/blob/main/src/open-r1-multimodal/src/open_r1/evaluate_sim2real.py) and [sft_sim2real.py](https://github.com/Jakhongir0103/VLM-R1/blob/main/src/open-r1-multimodal/src/open_r1/sft_sim2real.py).

We evaluate the model using the `evaluate.py`script. 

## Bias Mitigation
All code related to creating biased datasets is in `notebooks/Bias Project/`.

To train the models using biased datasets, please see the scripts in the `scripts/` folder that end with "text-biased". Moreover, to run the training of SmolVLM, please see the folder `scripts/SmolVLM/`.
The results shown in the paper can be found in the `results/vsr` folder for Qwen2.5 and `results/vsr/SmolVLM` for SmolVLM. The names of the scripts and results are self-explanatory.

### SmolVLM Adaptation
The original code for VLM-R1 is compatible with Qwen and InternVL. We had to create a separate module [vlm_modules/smolvlm_module.py](src/open-r1-multimodal/src/open_r1/vlm_modules/smolvlm_module.py) to adapt the code for SmolVLM. Moreover, [Idefics3 model](https://github.com/huggingface/transformers/blob/main/src/transformers/models/idefics3/processing_idefics3.py) (contains [the conditional generator](https://github.com/huggingface/transformers/blob/main/src/transformers/models/idefics3/modeling_idefics3.py#L861) for SmolVLM) does not pass the **image tokens** from the cache during generation. This issue is, according to us, an internal issue (*) of [Idefics3ForConditionalGeneration](https://github.com/huggingface/transformers/blob/main/src/transformers/models/idefics3/modeling_idefics3.py#L861) of the Transformers library. We experimented with modifying Idefics3ForConditionalGeneration by re-adding manually image tokens on the fly, but the results were inconclusive. Therefore, we disabled caching for SmolVLM during generation, which solved the issue, but drastically slows the generation speed.

SmolVLM is compatible with the usual SFT training.

Moreover, SmolVLM-500M-Instruct had a hard time outputting True/False answers instead of Yes/No. Therefore, we choose to relax the evaluation to accept both Yes/No and True/False answers, instead of trying to force the model to output True/False and have a very poor accuracy.

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