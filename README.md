# AutoCue
Target text adversarial sample generation
Our project is divided into three parts:

1. AutoCue:Our main experimental code (inspired by the [CLARE](https://github.com/cookielee77/CLARE) and [Textfooler](https://github.com/jind11/TextFooler) projects)
2.  TextAttack: A TextAttack framework modified for targeted attacks, serving as a baseline.
3.  LM-BFF: A method of fine-tuning a PLM from small sample data and generating Prompt.
-------------------
## For AutoCue:
```
#run AutoCue
pip install -r AutoCueRequirements.txt
bash run.sh
```
-------------------
## For TextAttack: 
check the TextAttack/README.md
-------------------
## LM-BFF:
the [LM-BFF](https://github.com/princeton-nlp/LM-BFF) output demo(We only use the templates and prompt words provided by LM-BFF, so LM-BFF and adversarial attack code are independent. LM-BFF only needs to be trained on the same distributed datasets.)

```
0.93750 *cls**sent_0*_Read_more_at*mask*_Week._com.*sep+*	{0: "Religion", 1: "Sports", 2: "Business", 3: "Tech"}
0.92188 *cls**sent_0*_You_are_currently_browsing_the*mask*_News_weblog_archives.*sep+*	{0: "Religion", 1: "Sports", 2: "Business", 3: "Tech"}
0.92188 *cls**sent_0*_Read_the_full_story_on*mask*.com.*sep+*	{0: "military", 1: "sport", 2: "Market", 3: "Tech"}
0.90625 *cls**sent_0*_This_article_originally_appeared_on*mask*_News.*sep+*	{0: "Jerusalem", 1: "Sport", 2: "Investment", 3: "Technology"}
0.90625 *cls**sent_0*_This_story_was_originally_published_by*mask*_News.*sep+*	{0: "Al", 1: "Sports", 2: "Investment", 3: "Digital"}
0.90625 *cls**sent_0*_Read_the_full_story_at*mask*.com.*sep+*	{0: "military", 1: "Sports", 2: "investors", 3: "Tech"}
```
