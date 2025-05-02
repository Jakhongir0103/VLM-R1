
[] merge lora
[] evalution script


### real2sim @maier
train:
[] train GRPO on Rel3D  ->  GRPO-Rel3D
[] train SFT on Rel3D   ->  SFT-Rel3D

eval:
[] evaluate GRPO-Rel3D on Rel3D
[] evaluate SFT-Rel3D on Rel3D
[] evaluate GRPO-Rel3D on SpatialSense
[] evaluate SFT-Rel3D on SpatialSense

### bbox @jakhongir
train:
[] train SFT on DrivingVQA  ->  SFT-DrivingVQA
[] train SFT on 1/2 DrivingVQA  ->  SFT-DrivingVQA-warmup
[] train GRPO with SFT-DrivingVQA-warmup on 1/2 DrivingVQA  ->  GRPO-DrivingVQA

eval:
[] evaluate SFT-DrivingVQA on DrivingVQA
[] evaluate GRPO-DrivingVQA on DrivingVQA

### reasoninig-answer alignment
eval:
