
# coding: utf-8

# ## Testing the behaviour of min hashing LSH
# We experiment with Amazon reviews
# 
# ## Dataset(s)
# Recall that your data is a ```.json``` file. Each line describes a review and has the following format:
# 
# ```
# {"reviewerID": "A14CK12J7C7JRK", "asin": "1223000893", "reviewerName": "Consumer in NorCal", "helpful": [0, 0], "reviewText": "I purchased the Trilogy with hoping my two cats, age 3 and 5 would be interested.  The 3 yr old cat was fascinated for about 15 minutes but when the same pictures came on, she got bored.  The 5 year old watched for about a few minutes but then walked away. It is possible that because we have a wonderful courtyard full of greenery and trees and one of my neighbors has a bird feeder, that there is enough going on outside that they prefer real life versus a taped version.  I will more than likely pass this on to a friend who has cats that don't have as much wildlife to watch as mine do.", "overall": 3.0, "summary": "Nice Distraction for my cats for about 15 minutes", "unixReviewTime": 1294790400, "reviewTime": "01 12, 2011"}
# ```
# ## Our analysis goal
# To the purpose of this analysis, we view each user as the set of items s/he reviewed. Given a user u, we are interested in all users, whose baskets (i.e., the sets of items they purchased) have Jaccard similarities with u's that exceed a given threshold $\theta$. The right value for $\theta$ is something that is normally the result of a separate analysis.
# In this lab we consider as  available values of $\theta$ the interval $[0.05,0.15]$ and for each value of $\theta$ we apply a K-fold cross validation with $K=5$,in each of which a different,randomly chosen set of 100 users will be used as a test set.
# 
# ## Our experiment
# Our  experiment concerns the accuracy and efficiency of min-hashing-LSH to the purpose of achieving the goal above. To this purpose, we divide  our dataset into a training and a test set for each fold. 
# We then build an LSH table associated to all users in the training set following the standard approach, as is implemented in the ```datasketch``` package, in particular by the ```MinHash``` and ```MinHashLSH``` classes.
# 
# Next, for each value of threshold,for each fold,for each user in the test, we identify all users whose true (using exhaustive search) or estimated (using min-hashing-LSH) Jaccard similarity is above the desired threshold $\theta$. For each user, we keep track of  precision and recall.Next, we compute average precision and recall over the ```n_test``` users considered for each value of fold.Finally for each threshold the precision and recall will be the average over 5 runs of the algorithm.
# 

# The following code implements the experiment above.We first import the necessary libraries and define a function for exact Jaccard similarity computation.

# In[1]:


import json
import time
import numpy as np
import matplotlib.pyplot as plt
from numpy import random
from datasketch import MinHash, MinHashLSH
from nltk.metrics.scores import (precision, recall)

def jaccard(X,Y):
    intersection = X.intersection(Y)
    return float(len(intersection)) / (len(X) + len(Y) - len(intersection))
    


# We also define an utility function in order to compute precision and recall for each user in the test set.The function return a tuple containing the mean of the precision and recall.

# In[2]:


def prec_rec(test_users,exact_neighbors,apx_neighbors):
    pr = []
    rc = []
    for uid in test_users:
        p = precision(set(exact_neighbors[uid]),set(apx_neighbors[uid]))
        r = recall(set(exact_neighbors[uid]),set(apx_neighbors[uid]))
        if p != None:
            pr.append(p)
        if r != None:
            rc.append(r)
    return (np.mean(pr),np.mean(rc))


# After the setting of thresholds and the number of folds,we create a map where keys are the reviewerID and as values we have a list of PIDs.

# In[5]:


sim_thresholds = np.arange(0.05,0.16,0.01)
pr_avg = []
rc_avg = []
k_fold=5

dataset= open("data")
complete_dict= {} 
for line in dataset:
    jrecord = json.loads(line)
    uid = jrecord["reviewerID"]
    if uid in complete_dict:
        complete_dict[uid].add(jrecord["asin"])
    else:
        complete_dict[uid] = {jrecord["asin"]}

dataset.close()

uIDs=list(complete_dict.keys())
        


# Next,for each threshold and for each fold ,we split the dataset in training and test set,we compute the signature matrix defining the number of hash function equal to $128$ adding all user of the training set,and finally we compute for each user in the test set the exact and aproximate neighbors(LSH),then we compute the  average of precision and the recall.After 5 fold we compute the average of  averages of precision and recall  and store them into two lists.

# In[6]:


for threshold in sim_thresholds:
    count=0
    pr = 0
    rc = 0
    i=0
    while i < k_fold:
        
        test_users = uIDs[count:count+100]
        train_users = uIDs[:count]+uIDs[count+100:]
        count+=100
        lsh = MinHashLSH(threshold=threshold, num_perm=128)
        
        for uid in train_users:
            m = MinHash(num_perm=128)
            for pid in complete_dict[uid]:
                m.update(pid.encode('utf8'))
            lsh.insert(uid,m) 

        #query usando test_users
        apx_neighbors = {}
        exact_neighbors ={}

        for uid in test_users:
            m = MinHash(num_perm=128)
            for pid in complete_dict[uid]:
                m.update(pid.encode('utf8'))
            apx_neighbors[uid]=lsh.query(m)
            exact_neighbors[uid]=[]

            for other in train_users:
                if jaccard(complete_dict[uid],complete_dict[other]) > threshold:
                    exact_neighbors[uid].append(other)

        tuple_pr_rc = prec_rec(test_users,exact_neighbors,apx_neighbors)
        pr += tuple_pr_rc[0]
        rc+= tuple_pr_rc[1]
        i+=1
    pr_avg.append(pr/k_fold)
    rc_avg.append(rc/k_fold)


# Basically ,here we plot the values of precision and recall for each available threshold.

# In[9]:


fig = plt.figure(1)

plt.subplot(211)
plt.plot(sim_thresholds, pr_avg)
plt.ylabel("Precision")
plt.xlabel("Threshold")
plt.axis([0.05, 0.16, 0, 1])
plt.xticks(np.arange(0.05, 0.16, 0.01))

plt.subplot(212)
plt.plot(sim_thresholds, rc_avg)
plt.ylabel("Recall")
plt.xlabel("Threshold")
plt.axis([0.05, 0.16, 0, 1])
plt.xticks(np.arange(0.05, 0.16, 0.01))
plt.show()
fig.savefig('plot_pc_rc_amazon.png')


# we know the precision and recall are defined as:
#  1. $PR=TP/(TP+FP)$
#  2. $RC=TP/(TP+FN)$
# 
# Looking at plots ,we can say that the best value of $\theta$ is $0.05$.In more detail,$\theta = 0.05 $ is the value gives us the minimum number of $FP$ and in the same time the minimum number of $FN$.
