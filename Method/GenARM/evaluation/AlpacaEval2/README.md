# Apaca-Eval

## Installation

The head-to-head comparison between two models' responses requires the `alpacaEval` environment below:

```
conda create --name alpacaEval python=3.10
conda activate alpacaEval
pip install alpaca-eval
```

To just use one OpenAI key, 
```
export OPENAI_API_KEY=<your_api_key> # for more complex configs, e.g. using Azure or switching clients see client_configs/README.md 
```

To set up Azure (or multiple OpenAI key), modify `openai_configs.yaml` and `export OPENAI_CLIENT_CONFIG_PATH=./openai_configs.yaml`. 

More details please refer to the official documentation [here](https://github.com/tatsu-lab/alpaca_eval). 

## Generate Responses

For generating responses, use the `genarm` environment. Modify and run
```
bash generate_output_AlpacaEval.sh
```
This bash script can also be adapted to run single model alone, without the reward guidance. 

## Evaluation
To run the evaluation, you need to use the `alpacaEval` environment.

### Compare with GPT4
The following command compares the model outputs to GPT4 by default
```
alpaca_eval --model_outputs MyOutputs.json --output_path ./AE2_result
```

### Compare with tulu-2-7b

To compare with other models such as `allenai/tulu-2-7b` in our main experiments, run the following:

`alpaca_eval --model_outputs MyOutputs.json --reference_outputs Path_to_Tulu_2_7b_Outputs.json --output_path ./AE2_result`