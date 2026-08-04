run_name_red=First_Model_Generation_Path.jsonl
run_name_blue=Second_Model_Generation_Path.jsonl

output_dir=gpt-evaluation # the evaluation will be saved at output_dir/f"{run_name_red_}_VS_{run_name_blue_}.json" (see gpt4_eval.py)

python gpt4_eval.py --run_name_red $run_name_red --run_name_blue $run_name_blue --output_dir $output_dir