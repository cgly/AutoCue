# -*- coding: utf-8 -*-
# @Time    : 2022/7/20 16:13
# @Author  : Zhuhe
# @File    : prompt_zhu_2022.py.py
# @Description :

"""
CLARE Recipe
=============

(Contextualized Perturbation for Textual Adversarial Attack)

"""

import transformers

from textattack import Attack
from textattack.constraints.pre_transformation import (
    RepeatModification,
    StopwordModification,
    MaxModificationRate
)
from textattack.constraints.semantics.sentence_encoders import UniversalSentenceEncoder
from textattack.goal_functions import UntargetedClassification
from textattack.goal_functions import TargetedClassification
from textattack.search_methods import GreedySearch
from textattack.transformations import (
    CompositeTransformation,
    WordInsertionMaskedLM,
    WordMergeMaskedLM,
    WordSwapMaskedLM,
    TarWordInsertionMaskedLM,
    TarWordMergeMaskedLM,
    TarWordSwapMaskedLM,
)

from .attack_recipe import AttackRecipe


class PromptAttack2022_Target(AttackRecipe):
    """Li, Zhang, Peng, Chen, Brockett, Sun, Dolan.

    "Contextualized Perturbation for Textual Adversarial Attack" (Li et al., 2020)

    https://arxiv.org/abs/2009.07502

    This method uses greedy search with replace, merge, and insertion transformations that leverage a
    pretrained language model. It also uses USE similarity constraint.
    """

    @staticmethod
    def build(model_wrapper,cur_target):
        # "This paper presents CLARE, a ContextuaLized AdversaRial Example generation model
        # that produces fluent and grammatical outputs through a mask-then-infill procedure.
        # CLARE builds on a pre-trained masked language model and modifies the inputs in a context-aware manner.
        # We propose three contex-tualized  perturbations, Replace, Insert and Merge, allowing for generating outputs of
        # varied lengths."
        #
        # "We  experiment  with  a  distilled  version  of RoBERTa (RoBERTa_{distill}; Sanh et al., 2019)
        # as the masked language model for contextualized infilling."
        # Because BAE and CLARE both use similar replacement papers, we use BAE's replacement method here.
        #r"E:\PLM\distilroberta-base"
        shared_masked_lm = transformers.RobertaModel.from_pretrained(
            r"E:\code\textcode\Pro-CLARE\MLM_model\distilroberta-base"
        )
        shared_tokenizer = transformers.RobertaTokenizer.from_pretrained(
            r"E:\code\textcode\Pro-CLARE\MLM_model\distilroberta-base"
        )
        #prompt
        ag_label_space= {0: 'World', 1: 'Sports', 2: 'Business', 3: 'Technology'}
        prompot_template="This topic is about {0}."
        tar_prompt=prompot_template.format(ag_label_space[cur_target])

        transformation = CompositeTransformation(
            [

                TarWordSwapMaskedLM(
                    tar_prompt=tar_prompt,
                    method="bae",
                    masked_language_model=shared_masked_lm,
                    tokenizer=shared_tokenizer,
                    max_candidates=50,
                    min_confidence=5e-4,
                ),
                TarWordInsertionMaskedLM(
                    tar_prompt=tar_prompt,
                    masked_language_model=shared_masked_lm,
                    tokenizer=shared_tokenizer,
                    max_candidates=50,
                    min_confidence=0.0,
                ),
                TarWordMergeMaskedLM(
                    tar_prompt=tar_prompt,
                    masked_language_model=shared_masked_lm,
                    tokenizer=shared_tokenizer,
                    max_candidates=50,
                    min_confidence=5e-3,
                ),
            ]
        )

        #
        # Don't modify the same word twice or stopwords.
        #
        constraints = [RepeatModification(), StopwordModification()]

        # "A  common  choice  of sim(·,·) is to encode sentences using neural networks,
        # and calculate their cosine similarity in the embedding space (Jin et al., 2020)."
        # The original implementation uses similarity of 0.7.
        use_constraint = UniversalSentenceEncoder(
            threshold=0.7,
            metric="cosine",
            compare_against_original=True,
            window_size=15,
            skip_text_shorter_than_window=True,
        )
        constraints.append(use_constraint)

        # Goal is untargeted classification.
        # "The score is then the negative probability of predicting the gold label from f, using [x_{adv}] as the input"
        #goal_function = UntargetedClassification(model_wrapper)
        goal_function = TargetedClassification(model_wrapper,target_class=cur_target)
        # "To achieve this,  we iteratively apply the actions,
        #  and first select those minimizing the probability of outputting the gold label y from f."
        #
        # "Only one of the three actions can be applied at each position, and we select the one with the highest score."
        #
        # "Actions are iteratively applied to the input, until an adversarial example is found or a limit of actions T
        # is reached.
        #  Each step selects the highest-scoring action from the remaining ones."
        #
        search_method = GreedySearch()

        return Attack(goal_function, constraints, transformation, search_method)
