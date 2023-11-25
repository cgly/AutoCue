# -*- coding: utf-8 -*-
# @Time    :
# @Author  : Zhuhe
# @File    : prompt_config.py
# @Description :

nclasses = {'ag': 4, 'yelp': 2, 'yahoo': 10, 'dbpedia': 14, 'ag_3': 4,
            'sst-2': 2, 'mnli':3, 'qnli': 2,'imdb':2}
# prompt_template={
#     'ag': "This article first appeared on {0} News.",
#     'yelp': "{0} news:",
#     'dbpedia':,
#     'sst-2': {0: "bad", 1: "delicious"},
#     'mnli': {0: "Contradiction", 1: 'Entailment', 2: "Neutral"},
#     'qnli': {0: "Yes", 1: 'No'},
# }
class_names = {#1'ag': {0: 'World', 1: 'Sports', 2: 'Business', 3: 'Sci-Tech'},#R原始版本:
                #'ag':{0: "Military", 1: "ESPN", 2: "Commercial", 3: "IT"},
            #'ag':{0: "Religion", 1: "Sports", 2: "Market", 3: "Digital"},
            #'ag':{0: "Oriental", 1: "Sports", 2: "Investment", 3: "Computer"},
            #'ag':{0: "Religion", 1: "Sport", 2: "Market", 3: "Digital"},#best
            'ag': {0: 'World', 1: 'Sports', 2: 'Business', 3: 'Technology'},#R:
            #'ag': {0: "MAP", 1: "Sports", 2: "Marketplace", 3: "Microsoft"},
            #'ag':{0: "CNN", 1: "ESPN", 2: "Business", 3: "Wired"},
            #'ag': {0: "Military", 1: "Sport", 2: "Commercial", 3: "IT"},
            'yelp': {0: "Negative", 1: 'Positive'},
            #'yelp': {0: "bad", 1: 'good'},
            'imdb':{0:"negative" ,1:'positive'},
            'dbpedia': {0: 'Company', 1: 'EducationInstitution', 2: "Artist",
                           3: "Athlete", 4: "OfficeHolder", 5: "MeanOfTransportation",
                           6: "Building", 7: "NaturalPlace", 8: "Village", 9: "Animal",
                           10: "Plant", 11: "Album", 12: "Film", 13: "WrittenWork"},#R:
            'yahoo':{0: 'Society & Culture', 1: 'Science & Mathematics', 2: "Health",
                       3: "Education & Reference", 4: "Computers & Internet", 5: "Sports",
                       6: "Business & Finance", 7: "Entertainment & Music", 8: "Family & Relationships", 9: "Politics & Government"},
            'sst-2': {0: "bad", 1: "delicious"},
            'mnli': {0: "Contradiction", 1: 'Entailment', 2: "Neutral"},
            'qnli': {0: "Yes", 1: 'No'},
}
