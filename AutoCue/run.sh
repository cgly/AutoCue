#! /bin/bash

######
##windows 运行参数模板
--dataset "ag"
--read_num 10
--attack_model "mutil"
--use_prompt 1
--prompt_target "target"
--prompt_template "You are currently browsing the {0} News category."
--prompt_location "sent"
--output_dir "adv_results"
--model_idx 1
--attack_loc 'brutal_force'
--use_windows 1
--target_model target_model
--target_model_name 'bert-base-uncased-ag-news'
--mlm_model "albert-base-v2"


--dataset $dataset \
                                  --read_num $read_num \
                                  --attack_model $attack_model \
                                  --use_prompt $use_prompt \
                                  --prompt_target $prompt_target \
                                  --prompt_template "$prompt_template" \
                                  --prompt_location $prompt_location \
                                  --output_dir $output_dir \
                                  --model_idx $model_idx \
                                  --attack_loc $attack_loc\
                                  --use_windows $use_windows \
                                  --target_model $target_model \
                                  --target_model_name $target_model\
                                  --mlm_model $mlm_model
######


#Candidate Template 候选模板
#{0} is the label word slot

#prompt_template="This topic is about {0}."
#prompt_template="This article first appeared on {0} News."
#prompt_template="You are currently browsing the {0} News category."
prompt_template="This article was originally published in {0} News."
#prompt_template="Read more at {0} Week.com"
#prompt_template="This article was originally published on {0}."
#prompt_template="Read the full story at {0} News."


attack_loc='brutal_force'
use_windows=0

#select the target model
target_model='bert-base-uncased-ag-news'

#select the MLM model
mlm_model="albert-base-v2"
#mlm_model="bert-base-uncased"
#mlm_model="distilroberta-base"

#dataset
dataset="ag"
read_num=10

#attack type
attack_model="mutil"  # "single"
use_prompt=1  #0
prompt_target="target"  # "ori" -target or original class label
prompt_location="rear"
#prompt_location='front'
"""
We employed six placement patterns for prompts:
1. Rear: Placed after the input.
2. Front: Placed before the input.
3. Both: Placed both before and after the input.
4. Sent: Placed after the nearest period following the <mask>.
5. Sent_r: Placed after the sentence containing the <mask>.
6. Sent_f: Placed before the sentence containing the <mask>.
"""

output_dir="adv_results4"
model_idx=4


export PYTHONWARNINGS='ignore:semaphore_tracker:UserWarning'
python runAutocue.py --dataset $dataset \
                                  --read_num $read_num \
                                  --attack_model $attack_model \
                                  --use_prompt $use_prompt \
                                  --prompt_target $prompt_target \
                                  --prompt_template "$prompt_template" \
                                  --prompt_location $prompt_location \
                                  --output_dir $output_dir \
                                  --model_idx $model_idx \
                                  --attack_loc $attack_loc\
                                  --use_windows $use_windows \
                                  --target_model $target_model \
                                  --target_model_name $target_model\
                                  --mlm_model $mlm_model


