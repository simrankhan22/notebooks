---

<a id="examjanuary-2022-problem4-sms-spam-filtering"></a>
## ExamJanuary_2022.PROBLEM4 – SMS spam filtering
## SMS spam filtering [8p]

In the following problem we will explore SMS spam texts. The dataset is the `SMS Spam Collection Dataset` and we have provided for you a way to load the data. If you run the appropriate cell below, the result will be in the `spam_no_spam` variable. The result is a `list` of `tuples` with the first position in the tuple being the SMS text and the second being a flag `0 = not spam` and `1 = spam`.

1. [3p] Let $X$ be the random variable that represents each SMS text (an entry in the list), and let $Y$ represent whether text is spam or not i.e. $Y \in \{0,1\}$. Thus $\mathbb{P}(Y = 1)$ is the probability that we get a spam. The goal is to estimate:
$$
    \mathbb{P}(Y = 1 | \text{"free" or "prize" is in } X) \enspace .
$$
That is, the probability that the SMS is spam given that "free" or "prize" occurs in the SMS.
Hint: it is good to remove the upper/lower case of words so that we can also find "Free" and "Prize"; this can be done with `text.lower()` if `text` a string.

2. [3p] Provide a "90\%" interval of confidence around the true probability. I.e. use the Hoeffding inequality to obtain for your estimate $\hat P$ of the above quantity. Find $l > 0$ such that the following holds:
$$
    \mathbb{P}(\hat P - l \leq \mathbb{E}[\hat P] \leq \hat P + l) \geq 0.9 \enspace .
$$
3. [2p] Repeat the two exercises above for "free" appearing twice in the SMS.

```{python}

# Run this cell to get the SMS text data
from Utils import load_sms
#from exam_extras import load_sms
spam_no_spam = load_sms()
print(spam_no_spam)

#Typical structure - just for general undestanding of sms data
#spam_no_spam = [
#    ("Free prize now!!!", 1),
#    ("Hey are we meeting today?", 0),
#    ...
#]


# fill in the estimate for part 1 here (should be a number between 0 and 1)

num_contains = 0
num_spam_and_contains = 0

for text, label in spam_no_spam:
    text_lower = text.lower()
    
    if ("free" in text_lower) or ("prize" in text_lower):
        num_contains += 1
        if label == 1:
            num_spam_and_contains += 1

problem4_hatP = num_spam_and_contains / num_contains
print(problem4_hatP)


#problem4_hatP = XXX


# fill in the calculated l from part 2 here
import numpy as np


"""
def hoeffding_conf_interval(data=spam_no_spam, a=0, b=1, conf_level=0.90):
    alpha = 1 - conf_level
    n = len(data)
    epsilon = (b - a) / np.sqrt(2 * n) * np.sqrt(np.log(2 / alpha))
    mean = np.mean(data)
    return (mean - epsilon, mean + epsilon)
print("90% Hoeffding confidence interval:", hoeffding_conf_interval)
"""
n=num_contains
conf_level = 0.9
alpha = 1 - conf_level

problem4_l = np.sqrt((1 / (2 * n)) * np.log(2 / alpha))

lower = problem4_hatP - problem4_l
upper = problem4_hatP + problem4_l

print("90% confidence interval:", (lower, upper))
print("problem4_l:", problem4_l)


# fill in the estimate for hatP for the double free question in part 3 here (should be a number between 0 and 1)
num_contains = 0
num_spam_and_contains = 0

for text, label in spam_no_spam:
    text_lower = text.lower()
    
    if text_lower.count("free") >= 2:
        num_contains += 1
        if label == 1:
            num_spam_and_contains += 1

problem4_hatP2 = num_spam_and_contains / num_contains
print(problem4_hatP2)
#problem4_hatP2 = XXX


# fill in the estimate for l for the double free question in part 3 here
n=num_contains
conf_level = 0.9
alpha = 1 - conf_level

problem4_l2 = np.sqrt((1 / (2 * n)) * np.log(2 / alpha))

lower = problem4_hatP2 - problem4_l2
upper = problem4_hatP2 + problem4_l2

print("90% confidence interval:", (lower, upper))
print("problem4_l:", problem4_l2)
#problem4_l2 = XXX

