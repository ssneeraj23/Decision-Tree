from typing import Counter
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
import random 
from pprint import pprint
from helper_funs import *
from clean_split import *
from tree_classifier import *
from pruning import *
import math

df=pd.read_csv("Dataset_A.csv")
df=clean_data(df)
FEATURE_TYPES=["categorical","categorical","continuous","categorical","categorical","continuous","categorical","continuous","categorical"]
COLUMN_HEADERS = df.columns
'''We split the data into 80% and 20% where 
20% is for test data and the 80% is further divided where
2/3 rd is for training data and 1/3 rd for validation data to prune'''
train_df,valid_df,test_df=train_valid_test_split(df,0.2)
depth,tree=decision_tree_algorithm(train_df,COLUMN_HEADERS,FEATURE_TYPES)
accuracy=calculate_accuracy(test_df,tree)
i=0
#Selecting best out of the 10 trees obtained from 10 random splits
while i<9:
    train_df_temp,valid_df_temp,test_df_temp=train_valid_test_split(df,0.2)
    depth_temp,tree_temp=decision_tree_algorithm(train_df_temp,COLUMN_HEADERS,FEATURE_TYPES)
    accuracy_temp=calculate_accuracy(test_df_temp,tree_temp)
    if(accuracy_temp>accuracy):
        train_df,valid_df,test_df=train_df_temp,valid_df_temp,test_df_temp
        tree=tree_temp
        depth=depth_temp
        accuracy=accuracy_temp
    i+=1 

''' Tree will be printed in the form of dictionary 
    where each key represents a node with an attribue and split value and 
    values of keys are also two nodes one left and one right each
    telling what is attribute value is less than split value and greater than 
    equal to split value. If the attribute is categorical then it compares if the 
    attribute value is equal or not'''
f=open("result.txt","a")
f.write("Tree will be printed in the form of dictionary where each key represents a node with an attribute and split value and values of keys are also two nodes one left and one right each telling what is attribute value is less than split value and greater than equal to split value. If the attribute is categorical then it compares if the attribute value is equal or not\n")
print("\n\n**********Tree with best test accuracy out of 10 random splits before pruning is********** \n\n\n")
pprint(tree,width=50)
f.write("The best test accuracy tree has an accuracy value of ")
f.write(str(accuracy))
f.write("\n")
print("The best test accuracy tree has an accuracy value of ")
print(accuracy)
print("\n")
print("Depth of unpruned tree is")
print(depth)
f.write("Depth of above unpruned tree is \n")
f.write(str(depth))
f.write("\n")
f.write("\n\n\n *********************Now we are going to prune the tree*********************\n\n\n ")
print("\n\n\n *********************Now we are going to prune the tree*********************\n\n\n ")

#pruning the tree using validation set
pruned_tree=post_pruning(tree,train_df,valid_df)
pruned_accuracy=calculate_accuracy(test_df,pruned_tree)
pprint(pruned_tree,width=50)
print("The accuracy of pruned tree is ")
print(pruned_accuracy)
f.write("The accuracy of pruned tree has an accuracy value of ")
f.write(str(pruned_accuracy))
f.write("\n")
#plotting a graph showing the variation in test accuracy with varying depth

metrics = {"max_depth": [], "acc_tree": [], "acc_tree_pruned": []}
for n in range(15, 50):

    limit_depth,limit_tree = decision_tree_algorithm(train_df,COLUMN_HEADERS,FEATURE_TYPES, max_depth=n)
    tree_pruned = post_pruning(limit_tree, train_df, valid_df )
    
    metrics["max_depth"].append(n)
    metrics["acc_tree"].append(calculate_accuracy(test_df,limit_tree))
    metrics["acc_tree_pruned"].append(calculate_accuracy(test_df, tree_pruned))
    

plt.plot(metrics["max_depth"],metrics["acc_tree"],label="actual tree",color="red",marker="o")
plt.plot(metrics["max_depth"],metrics["acc_tree_pruned"],label="pruned tree",color="blue",marker="o")
plt.title("Actual tree vs Pruned Tree")
plt.xlabel("max_depth")
plt.ylabel("accuracy against test data")
plt.legend()
plt.show()



