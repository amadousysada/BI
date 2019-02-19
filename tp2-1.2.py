#!/usr/bin/env python
# -*- coding: utf-8 -*-

from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

from scipy.io import arff
import pandas as pd

data, meta = arff.loadarff("vote.arff")

vote = pd.DataFrame(data)

# convert categorical values into one-hot vectors and ignore ? values
# corresponding to missing values
# ex: handicapped-infants=y -> [1,0], handicapped-infants=n -> [0,1], handicapped-infants=? -> [0,0]
vote_one_hot = pd.get_dummies(vote)

vote_one_hot.drop(vote_one_hot.filter(regex='_\?$',axis=1).columns,axis=1,inplace=True)
frequent_item_sets = apriori(vote_one_hot, min_support=0.4, use_colnames=True)

rules = association_rules(frequent_item_sets, metric="confidence", min_threshold=0.9)
'''x= rules.loc[0]
for r in x:
	print (r)'''
print(rules['confidence'])
exit()


# check that there is no rule implying Republicans
filter(lambda x: "Class_'republican'" in x,rules['antecedents'])
filter(lambda x: "Class_'republican'" in x,rules['consequents'])

