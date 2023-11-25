# -*- coding: utf-8 -*-
# @Time    : 2023/7/19 15:24
# @Author  : Zhuhe
# @File    : MulT.py
# @Description :
import textattack
import transformers
import csv
import pandas as pd
import wandb
import os
import TA_eval
from textattack.metrics.attack_metrics import (
    AttackQueries,
    AttackSuccessRate,
    WordsPerturbed,
)
def baseinfo(results):
    # Default metrics - calculated on every attack
    attack_success_stats = AttackSuccessRate().calculate(results)
    words_perturbed_stats = WordsPerturbed().calculate(results)
    attack_query_stats = AttackQueries().calculate(results)


    suc_num=attack_success_stats["successful_attacks"]
    fail_num=attack_success_stats["failed_attacks"]
    ski_num=attack_success_stats["skipped_attacks"]
    ori_acc=(attack_success_stats["original_accuracy"]) / 100
    underAttack_acc=(attack_success_stats["attack_accuracy_perc"]) / 100
    Attack_suc_rate=(attack_success_stats["attack_success_rate"]) / 100
    perturbed_rate=(words_perturbed_stats["avg_word_perturbed_perc"]) / 100

    avg_len=words_perturbed_stats["avg_word_perturbed"]
    avg_query=attack_query_stats["avg_num_queries"]

    final_result = {'suc_num':suc_num,
                    'fail_num':fail_num,
                    'ski_num':ski_num,
                    'ori_acc':ori_acc,
                    'underAttack_acc':underAttack_acc,
                    'Attack_suc_rate':Attack_suc_rate,
                    'perturbed_rate':perturbed_rate,
                    'avg_len':avg_len,
                    'avg_query':avg_query}
    return final_result

def read_corpus(path, text_label_pair=False):
    with open(path, encoding='utf8') as f:
        data=[]
        examples = list(csv.reader(f, delimiter='\t', quotechar=None))[1:]
        second_text = False if examples[1][2] == '' else True
        for i in range(len(examples)):
            examples[i][0] = int(examples[i][0])
            if not second_text:
                examples[i][2] = None
            data.append((examples[i][1],examples[i][0]))
    return data

#vitcim_model
#model_path=r".\saved_model\bert_ag_uncased"
#model_path=r".\saved_model\bert_textfooler_ag"
model_path=r".\saved_model\bert-base-uncased-ag-news"
model = transformers.AutoModelForSequenceClassification.from_pretrained(model_path)
tokenizer = transformers.AutoTokenizer.from_pretrained(model_path)
model_wrapper = textattack.models.wrappers.HuggingFaceModelWrapper(model, tokenizer)

#data
data_path=r".\data\ag.tsv"
corpus=read_corpus(data_path)
dataset = textattack.datasets.Dataset(corpus)

#attack info
target_attack_results=[]
classes_num=4

dataset_class_space=range(classes_num)
#dataset_class_space=[0,1]#,2,3]

save_path="save_log"
datasetName="ag"
Attack_mothed="Tar_Texefooler"
read_num=1000

project_name=Attack_mothed+"_"+datasetName
save_file=os.path.join(save_path, project_name)+".csv"

wandb.init(project=project_name)

##############################
####Choose attack method######
##############################
# The attack in the attack_recipes has been modified to a targeted attack
##goal_function = UntargetedClassification(model_wrapper) change to  TargetedClassification(model_wrapper, target_class=cur_target)
attack_mothed=textattack.attack_recipes.TextFoolerJin2019
# attack_mothed=textattack.attack_recipes.PSOZang2020
# attack_mothed=textattack.attack_recipes.PWWSRen2019
# attack_mothed=textattack.attack_recipes.BAEGarg2019
print("models loaded,start attack")
for cur_tar in dataset_class_space:
    print("---------------strat attack "+str(cur_tar)+" class--------------------")
    attack = attack_mothed.build(model_wrapper,cur_target=cur_tar)

    #dataset = textattack.datasets.HuggingFaceDataset("imdb", split="test")

    # Attack 20 samples with CSV logging and checkpoint saved every 5 interval
    attack_args = textattack.AttackArgs(
                    num_examples=read_num,
                    #log_to_csv="save_log/Textfooler-ag-log.csv",
                    checkpoint_interval=500,
                    checkpoint_dir="checkpoints",
                    disable_stdout=True,
                    #enable_advance_metrics = True,
                    #log_to_wandb=config)
                    )
    attacker = textattack.Attacker(attack, dataset, attack_args)
    attack_results=attacker.attack_dataset()
    for result in attack_results:
        target_attack_results.append(result)

result_dict=baseinfo(target_attack_results)
print(result_dict)

attackLog=textattack.loggers.AttackLogManager()
attackLog.add_output_csv(filename=save_file,color_method=None)
#attackLog.enable_wandb(project=project_name)
attackLog.log_results(target_attack_results)
attackLog.flush()

#计算ppl,use,bertScore,grammer_error
usem = textattack.metrics.quality_metrics.USEMetric().calculate(target_attack_results)
ppl = textattack.metrics.quality_metrics.Perplexity().calculate(target_attack_results)
print(usem)
print(ppl)

#ref_ppl, hy_ppl, Bert_sore_F1, sim_score, gramar_err=TA_eval.eval()
#上传到wandb
wandb.log(result_dict)
wandb.log({
    'usem':usem,
    'ppl':ppl
})
wandb.save(save_file)