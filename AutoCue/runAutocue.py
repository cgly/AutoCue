import os
import random
import time
from collections import Counter
import pandas as pd
import nltk
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from transformers import RobertaTokenizer, AutoTokenizer, AlbertForMaskedLM,AlbertTokenizer,BertTokenizer,BertForMaskedLM

from config import load_arguments
from utils.hyper_parameters import class_names, nclasses, thres
from dataloaders.dataloader import read_corpus
from models.similarity_model import USE

from models.BERT_classifier import BERTinfer
from models.attack_location_search import get_target_attack_sequences
from models.attack_operations import *
from models.pipeline import FillMaskPipeline
#from models.Roberta import RobertaForMaskedLM
from transformers import RobertaForMaskedLM
from evaluate import evaluate
from transformers import BertForMaskedLM,BertTokenizer
import wandb


# for token check
import re
punct_re = re.compile(r'\W')
words_re = re.compile(r'\w')

def targetAttack(args,example, predictor, stop_words_set, fill_mask,sim_predictor=None,
           synonym_num=50, attack_second=False, attack_loc=None,
           thres_=None):
    true_label = example[0]
    if attack_second:
        text_ls = example[2].split()
        text2 = example[1]
    else:
        text_ls = example[1].split()
        text2 = example[2]
    # first check the prediction of the original text
    orig_probs = predictor([text_ls], text2).squeeze()
    orig_label = torch.argmax(orig_probs).item()
    orig_prob = orig_probs.max()

    # rafael: 获得原始的预测类别数
    num_classes = orig_probs.shape[0]

    #[' '.join(text_prime), num_changed, orig_label, new_label, num_queries,targetLabel]
    # Misclassification on the first prediction
    if true_label != orig_label:
        return [['', 0, orig_label, orig_label, 0,-1]], [[]]
    num_queries = 1

    #first prediction right,Then attack all other Classes
    targetAttackRes=[]
    targetAttack_logs=[]
    for targetLabel in range(num_classes):
        if targetLabel==true_label:
            continue
        # find attack sequences according to predicted probablity change
        attack_sequences, num_query = get_target_attack_sequences(
            args,targetLabel,text_ls, fill_mask, predictor, sim_predictor,
            orig_probs, orig_label, stop_words_set, punct_re, words_re,
            text2=text2, attack_loc=attack_loc, thres=thres_)
        num_queries=1
        num_queries += num_query

        # perform attack sequences
        target_prob = orig_probs[targetLabel]
        attack_logs = []
        text_prime = text_ls.copy()
        prev_prob = orig_prob
        insertions = []
        merges = []
        forbid_replaces = set()
        forbid_inserts = set()
        forbid_merges = set(range(5))
        num_changed = 0
        new_label = orig_label
        for attack_info in attack_sequences:
            num_queries += synonym_num
            idx = attack_info[0]
            attack_type = attack_info[1]
            orig_token = attack_info[2]
            # check forbid replace operations
            if attack_type == 'insert' and idx in forbid_inserts:
                continue
            if attack_type == 'merge' and idx in forbid_merges:
                continue
            if attack_type == 'replace' and idx in forbid_replaces:
                continue

            # shift the attack index by insertions history
            shift_idx = idx
            for prev_insert_idx in insertions:
                if idx >= prev_insert_idx:
                    shift_idx += 1
            for prev_merge_idx in merges:
                if idx >= prev_merge_idx + 1:
                    shift_idx -= 1

            #targetLabel and target_prob  and prompt
            if attack_type == 'replace':
                synonym, syn_prob, prob_diff, semantic_sim, new_prob, collections = \
                    word_replacement(args,
                        targetLabel,target_prob,shift_idx, text_prime, fill_mask, predictor,
                        prev_prob, orig_label, sim_predictor, text2, thres=thres_)
            elif attack_type == 'insert':
                synonym, syn_prob, prob_diff, semantic_sim, new_prob, collections = \
                    word_insertion(args,
                        targetLabel,target_prob,shift_idx, text_prime, fill_mask, predictor,
                        prev_prob, orig_label, punct_re, words_re, sim_predictor, text2, thres=thres_)
            elif attack_type == 'merge':
                synonym, syn_prob, prob_diff, semantic_sim, new_prob, collections = \
                    word_merge(args,
                        targetLabel,target_prob,shift_idx, text_prime, fill_mask, predictor,
                        prev_prob, orig_label, sim_predictor, text2, thres=thres_)

            if prob_diff < 0:
                #         import ipdb; ipdb.set_trace()
                if attack_type == 'replace':
                    text_prime[shift_idx] = synonym
                    # forbid_inserts.add(idx)
                    # forbid_inserts.add(idx+1)
                    forbid_merges.add(idx - 1)
                    forbid_merges.add(idx)
                elif attack_type == 'insert':
                    text_prime.insert(shift_idx, synonym)
                    # append original attack index
                    insertions.append(idx)
                    forbid_merges.add(idx - 1)
                    # forbid_merges.add(idx)
                    for i in [-1, 1]:
                        forbid_inserts.add(idx + i)
                elif attack_type == 'merge':
                    text_prime[shift_idx] = synonym
                    del text_prime[shift_idx + 1]
                    merges.append(idx)
                    # forbid_inserts.add(idx)
                    forbid_inserts.add(idx + 1)
                    # forbid_inserts.add(idx+2)
                    # forbid_replaces.add(idx-1)
                    forbid_replaces.add(idx)
                    forbid_replaces.add(idx + 1)
                    for i in [-1, 1]:
                        forbid_merges.add(idx + i)
                cur_prob = new_prob[targetLabel].item()
                attack_logs.append([idx, attack_type, orig_token, synonym, syn_prob,
                                    semantic_sim, prob_diff, cur_prob])
                target_prob=cur_prob
                prev_prob = cur_prob
                num_changed += 1
                # if attack successfully!
                # if np.argmax(new_prob) != orig_label:
                #     new_label = np.argmax(new_prob)
                #     break

                #new_label==target category label(the class we want)
                if np.argmax(new_prob) == targetLabel:
                    new_label = np.argmax(new_prob)
                    break
        targetAttackRes.append([' '.join(text_prime), num_changed, orig_label, new_label, num_queries,targetLabel])
        targetAttack_logs.append(attack_logs)

    return targetAttackRes,targetAttack_logs
def target_main(args,plm_path,read_num,dataset='ag'):
    begin_time = time.time()
    num_class=len(class_names.get(dataset))
    # get data to attack
    examples = read_corpus(args.attack_file,read_num=read_num)
    if args.data_size is None:
        args.data_size = len(examples)
    examples = examples[args.data_idx:args.data_idx+args.data_size] # choose how many samples for adversary
    print("Data import finished!")

    # construct the model
    print("Building Model...")
    model = BERTinfer(args.target_model, args.target_model_path,
                      nclasses[args.dataset], args.case,
                      batch_size=args.batch_size,
                      attack_second=args.attack_second)
    predictor = model.text_pred
    print("Model built!")

    # prepare context predictor
    #tokenizer = RobertaTokenizer.from_pretrained("/home/zhuhe/distilroberta-base")
    #model = RobertaForMaskedLM.from_pretrained("/home/zhuhe/distilroberta-base")
    if 'roberta' in str(plm_path):
        tokenizer = RobertaTokenizer.from_pretrained(plm_path)
        model = RobertaForMaskedLM.from_pretrained(plm_path)
    elif 'albert' in str(plm_path):
        tokenizer = AlbertTokenizer.from_pretrained(plm_path)
        model = AlbertForMaskedLM.from_pretrained(plm_path)
    else: #bert
        tokenizer = BertTokenizer.from_pretrained(plm_path)
        model = BertForMaskedLM.from_pretrained(plm_path)
    fill_mask = FillMaskPipeline(model, tokenizer, topk=args.synonym_num)

    # build the semantic similarity module
    use = USE(args.USE_cache_path)

    # start attacking
    num_sample = 0
    orig_failures = 0.
    adv_failures = 0.
    skipped_idx = []
    changed_rates = []
    nums_queries = []
    attack_texts = []
    target_record=[]
    new_texts = []
    label_names = class_names[args.dataset]
    log_file = open(os.path.join(
        args.output_dir,str(args.data_size) + '_results_log'), 'a')
    if args.write_into_tsv:
        folder_path = os.path.join('./data', args.sample_file, args.dataset)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        tsv_name = os.path.join(folder_path, "%d.tsv" % args.data_idx)
        adversarial_file = open(tsv_name, 'w', encoding='utf8')
        header = 'label\ttext1\ttext2\tnum_change\n'
        adversarial_file.write(header)
    else:
        sample_file = open(
            os.path.join(args.output_dir, args.sample_file), 'w', encoding='utf8')

    stop_words_set = set(nltk.corpus.stopwords.words('english'))
    print('Start attacking!')

    #用于类别统计
    class_num = {}  # 类别数
    fail_class_num = {}  # 初始预测失败数
    success_class_num = {}  # 初始预测成功数
    success_class_rate = {}  # 初始预测成功比
    targetAttack_fail_num = {}  # 每一个类别 多少初始预测成功 但是攻击失败的
    targetAttack_num = {}  # 类别分流数
    targetAttack_rate = {}  # 类别分流攻击比 = 类别分流数 / 初始预测成功数 （反映在预测成功的数据中，攻击每一类的成功率）

    idx_list=[]
    for idx, example in enumerate(tqdm(examples)):
        true_label = example[0]
        if example[2] is not None:
            single_sentence = False
            attack_text = example[2] if args.attack_second else example[1]
            ref_text = example[1] if args.attack_second else example[2]
        else:
            single_sentence = True
            attack_text = example[1]
        if len(tokenizer.encode(attack_text)) > args.max_seq_length:
            skipped_idx.append(idx)
            continue

        num_sample += 1*(num_class-1)

        #统计每类数量
        class_num[true_label]=class_num.get(true_label, 0) + 1


        # 攻击的核心代码
        #每次对抗 所有其他类
        targetAttackRes,targetAttack_logs = targetAttack(args,example, predictor, stop_words_set,
                         fill_mask, sim_predictor=use,
                         synonym_num=args.synonym_num,
                         attack_second=args.attack_second,
                         attack_loc=args.attack_loc,
                         thres_=thres[args.dataset])


        #[['', 0, orig_label, orig_label, 0, -1]], [[]]
        for r, l in zip(targetAttackRes, targetAttack_logs):
            new_text, num_changed, orig_label, new_label, num_queries,target_label = r
            attack_logs = l
            #此处的返回值保证了 初始预测 正确
            if true_label != orig_label:
                orig_failures += 1*(num_class-1)
                #统计每一类分类失败的数量
                fail_class_num[true_label]=fail_class_num.get(true_label, 0) + 1
            else:
                nums_queries.append(num_queries)


            changed_rate = 1.0 * num_changed / len(attack_text.split())

            if true_label == orig_label and new_label != target_label:
                targetAttack_fail_num[true_label] = targetAttack_fail_num.get(true_label, 0) + 1

            #此处条件表示 初始预测正确 且 攻击target成功
            if true_label == orig_label and  target_label== new_label:
                sympol=str(true_label)+"-"+str(target_label)
                targetAttack_num[sympol]=targetAttack_num.get(sympol, 0) + 1

                adv_failures += 1

                # 用于记录每个类别的攻击数据
                attack_texts.append(attack_text)
                new_texts.append(new_text)
                changed_rates.append(changed_rate)
                idx_list.append(idx)
                #记录了原始类和目标类
                target_record.append(sympol)

                ######
                if args.write_into_tsv:
                    text1 = new_text.strip()
                    text2 = "" if single_sentence else ref_text.strip()
                    if args.attack_second:
                        tmp = text1
                        text1, text2 = text2, tmp
                    string_ = "%d\t%d\t%s\t%s\t%d\n" % (orig_label, target_label,text1, text2, num_changed)
                    adversarial_file.write(string_)
                else:
                    sample_file.write("Sentence index: %d\n" % idx)
                    if not single_sentence:
                        sample_file.write('ref sent: %s\n' % ref_text)
                    sample_file.write('orig sent ({}):\t{}\nadv sent ({}):\t{}\n'.format(
                        true_label, attack_text, new_label, new_text))
                    sample_file.write('label change: %s ---> %s. num of change: %d\n\n' % \
                        (label_names[orig_label], label_names[new_label], len(attack_logs)))
                    sample_file.write('changed_rate: %f , num_queries: %d\n\n' % \
                                      (changed_rate,num_queries ))
                    for attack_info in attack_logs:
                        output_str = "%d %s %s %s %.4f %.2f %.4f %.4f\n" % tuple(attack_info)
                        sample_file.write(output_str)
                    sample_file.write('\n---------------------------------------------\n')

    #测试分类数据

    success_class_num=dict(Counter(class_num) - Counter(fail_class_num))

    #计算预测成功率 success_class_rate = {}  # 初始预测成功比 success_class_num/ class_num
    for key,val in success_class_num.items():
        success_class_rate[key]=val/class_num[key]
    #计算类别分流率 targetAttack_rate = {}  # 类别分流攻击比 = 类别分流数 / 初始预测成功数 （反映在预测成功的数据中，攻击每一类的成功率）
    # 计算类别分流率 targetAttack_rate = {}  # 类别分流攻击比 = 类别分流数 / 初始预测成功数 （反映在预测成功的数据中，攻击每一类的成功率）

    target_success_class_num = dict(Counter(success_class_num) - Counter(targetAttack_fail_num))
    target_class_rate = {}
    for key, val in success_class_num.items():
        val=val*(num_class-1)
        target_class_rate[key] = (val-targetAttack_fail_num.get(key, 99999)) / val


    for key,val in targetAttack_num.items():
        ori_class=int(key[0])
        targetAttack_rate[key]=val/success_class_num[ori_class]

    #根据攻击类别 计算详细数据
    text_dic = {
        "target_record": target_record,
        "attack_texts": attack_texts,
        "new_texts": new_texts,
        "changed_rates": changed_rates,
        "idx_list":idx_list
    }
    eval_list = pd.DataFrame(text_dic)
    mutil_target_res = {}
    name_list = ['orig_ppl', 'adv_ppl', 'bert_score', 'sim_score', 'gram_err','changed_rate']

    # # 需要按类别统计时启动
    # for val in set(target_record):
    #     current_list = eval_list[eval_list["target_record"] == val]
    #     att_text = current_list["attack_texts"].tolist()
    #     new_text = current_list["new_texts"].tolist()
    #     changed_rate = np.mean(current_list["changed_rates"].tolist())
    #
    #     orig_ppl, adv_ppl, bert_score, sim_score, gram_err = evaluate(att_text, new_text, use, args)
    #     mutil_target_res[val] = dict(zip(name_list, [orig_ppl, adv_ppl, bert_score, sim_score, gram_err,changed_rate]))
    #
    # #退化为单目标攻击的最优结果（从change_rate角度）
    # best_single_result = eval_list.groupby(by='idx_list', as_index=False).min()
    # best_att_text = best_single_result["attack_texts"].tolist()
    # best_new_text = best_single_result["new_texts"].tolist()
    # best_changed_rate = np.mean(best_single_result["changed_rates"].tolist())
    # best_acc=len(best_single_result["idx_list"])*(num_class-1)/(num_sample - orig_failures)
    # orig_ppl, adv_ppl, bert_score, sim_score, gram_err = evaluate(best_att_text, best_new_text, use, args)
    # best_name_list = ['attack_success','orig_ppl', 'adv_ppl', 'bert_score', 'sim_score', 'gram_err', 'changed_rate']
    # best_result=dict(zip(best_name_list, [best_acc,orig_ppl, adv_ppl, bert_score, sim_score, gram_err, best_changed_rate]))
    best_result=''

    #统计输出最终结果
    orig_acc = (1 - orig_failures / num_sample) * 100
    attack_rate = 100 * adv_failures / (num_sample - orig_failures)
    log_file.write(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    message = '\nFor Generated model {} / Target model {} : original accuracy: {:.3f}%, attack success: {:.3f}%, ' \
              'avg changed rate: {:.3f}%, num of queries: {:.1f}, num of samples: {:d}, time: {:.1f}\n'.format(
                  args.sample_file, args.target_model, orig_acc, attack_rate,
                  np.mean(changed_rates)*100, np.mean(nums_queries), num_sample, time.time() - begin_time)
    print(message)
    log_file.write(message)
    torch.cuda.empty_cache()

    ###################
    ####eval过程可能爆内存,所以保存中间结果
    ###################
    # print("")
    # np.savez(r'./tempData/evalData.npz', attack_texts=attack_texts, new_texts=new_texts)
    # print("evalData saved")
    # evalData = np.load(r'./tempData/evalData.npz', allow_pickle=True)
    # attack_texts = (evalData['attack_texts'])
    # new_texts = (evalData['new_texts'])


    orig_ppl, adv_ppl, bert_score, sim_score, gram_err = evaluate(attack_texts, new_texts, use,args)
    #orig_ppl, adv_ppl,  sim_score, gram_err = evaluate(attack_texts, new_texts, use)
    message = 'Original ppl: {:.3f}, Adversarial ppl: {:.3f}, BertScore: {:.3f}, SimScore: {:.3f}, gram_err: {:.3f}\n'. \
        format(orig_ppl, adv_ppl, bert_score, sim_score, gram_err)
    log_file.write(message)


    print("Skipped indices: ", skipped_idx)
    print("Processing time: %d" % (time.time() - begin_time))

    train_conifg = 'target func: {0},Use prompt: {1} ,prompt_template: {2} ,prompt_location: {3} ,prompt_target: {4} , read_num: {5} ,search_mothed: {6}\n'. \
        format('multi class', args.use_prompt, args.prompt_template, args.prompt_location,
               args.attack_model, read_num, args.attack_loc)
    log_file.write(train_conifg)

    parameters=str(thres[args.dataset])+'\n'
    log_file.write(parameters)

    #wandb config
    if use_wandb:
        config = wandb.config  # Initialize config
        config.MaskModel =  args.mlm_model
        config.TargetModel =args.target_model
        config.sample_num=  read_num
        config.attack_loc = args.attack_loc

        #thres
        config.dataset_name=thres[args.dataset].get('dataset_name')
        #
        config.replace_prob = thres[args.dataset].get('replace_prob')
        config.insert_prob = thres[args.dataset].get('insert_prob')
        config.merge_prob = thres[args.dataset].get('merge_prob')
        #
        config.replace_sim = thres[args.dataset].get('replace_sim')
        config.insert_sim = thres[args.dataset].get('insert_sim')
        config.merge_sim = thres[args.dataset].get('merge_sim')
        #
        config.prob_diff = thres[args.dataset].get('prob_diff')
        config.sim_window = thres[args.dataset].get('sim_window')
        #prompt

        config.attack_model = args.attack_model
        config.model_idx = args.model_idx

        config.use_prompt = args.use_prompt
        config.prompt_template=args.prompt_template
        config.prompt_location=args.prompt_location
        config.prompt_target=args.prompt_target
        config.label_names=label_names
        #args
        wandb.log({
            # "Generated model": args.sample_file,
            # "Target model": args.target_model,
            "original accuracy":orig_acc,
            "attack success": attack_rate,
            "avg_changed_rate": np.mean(changed_rates) * 100,
            "num of queries": np.mean(nums_queries),
            "num of samples": num_sample,
            "Original ppl": orig_ppl,
            "Adversarial ppl": adv_ppl,
            "BertScore": bert_score,
            "SimScore": sim_score,
            "gram_err": gram_err,
            "time":time.time() - begin_time,
            "classes_num":str(class_num), ## 各类类别数
            "success_class_num":str(success_class_num),# 初始预测成功数
            "success_class_rate":str(success_class_rate), # 初始预测成功比
            "target_class_rate": str(target_class_rate),  # 攻击成功率
            "targetAttack_num":str(targetAttack_num),# 类别分流数
            "targetAttack_rate":str(targetAttack_rate),# 类别攻击 成功率
            "mutil_target_res":str(mutil_target_res), #eval 测评信息
            "best_result":str(best_result),
            "targetAttack_fail_num":str(targetAttack_fail_num)
        })

        wandb.save(os.path.join(args.output_dir, args.sample_file))

    #os.remove(r'./tempData/evalData.npz')
    print("Attack Finish,delete the evalData")

use_wandb=0
if __name__ == "__main__":

    args = load_arguments()

    #build wandb project name
    project_name="wandbName"+str(args.dataset)
    if use_wandb:
        wandb.init(project=project_name)

    #select MASK LM

    # if args.use_windows:
    #     plm_path= os.path.join('./MLM_model',args.mlm_model)

    plm_path = os.path.join(r'.\MLM_model', args.mlm_model)

    if args.attack_model=='mutil':
        print("start ###target### attack")
        target_main(args,plm_path,read_num=int(args.read_num),dataset=args.dataset)
    else:
        assert "model error!"
    torch.cuda.empty_cache()