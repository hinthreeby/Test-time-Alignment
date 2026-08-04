file_name_red=Your_Model_Generation_Path.json
eval_path_root=./gpt-evaluation/evaluation/raw_results/

# by default, compare with the base model
file_name_blue=YourPath/alpaca-7b-reproduced/generation.json

python gpt4_eval_Beaver.py --eval_path_root $eval_path_root --file_name_red $file_name_red --file_name_blue $file_name_blue 