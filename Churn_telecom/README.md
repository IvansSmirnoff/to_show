# Clients churn prediction model using catboost with EDA
# Summary

Achieved roc-auc - 0.91+

Great result for such a task: the excellent mark from the reviewer is given when you achieve roc-auc > 0.88

Order:
- __Target__: The clients whose `live_contract` is empty haven't stopped using our telecom services. For them target is 0, for those who decided to quit- 1. After filling the target values the `live_contract` column is filled
- `live_contract` is filled with all the data: those who has had `NaN` now has the contract duration as a result of subtraction of the beginning date and 1st of February 2020
- __New features__ Family_guy: equals to 1 if the client has dependants and a spouse at the same time
- __Learning__
    - catboostclassifier(), parameters are chosen using gridsearchCV
    - Test sample auc-roc = 0.91+
    
    _accuracy is 88%_

# Stack

pandas, matplotlib, sklearn, catboost, numpy, re

# Статус
- [x] Completed
