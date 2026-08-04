# Generating Responses

To generate responses, run:
```
bash generate_outputs_Beaver.sh
```

# GPT evaluation

To run head-to-head comparison between two models, provide the output files of both models in `gpt4_eval.sh` and run:

```
bash gpt4_eval_Beaver.sh
```

To obtain the frontier of GenARM (or other methods like multi-objective RL), we keep the `file_name_blue` as the output paths of the reference models (`PKU-Alignment/alpaca-7b-reproduced` or `TheBloke/alpaca-lora-65B-GPTQ`). The `file_name_red` should be the output file of different methods with varying reward coefficients. 