# TextAttack

We have adopted a widely used text adversarial framework *TextAttack*(https://github.com/QData/TextAttack) and made some modifications to adapt to targeted attacks. We observe the targeted attack performance of existing methods and use those as baseline models.


--------------
```
# The attack in the attack_recipes has been modified to a targeted attack
goal_function = UntargetedClassification(model_wrapper) #old
goal_function = TargetedClassification(model_wrapper, target_class=cur_target) #new
```

```
#Execute targeted attacks
python MulTarAttack.py
```
