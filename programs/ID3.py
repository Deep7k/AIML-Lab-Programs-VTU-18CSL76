"""Write a program to demonstrate the working of the decision tree based ID3 algorithm. Use an appropriate data set
for building the decision tree and apply knowledge to classify a new sample """

from pprint import pprint
import pandas as pd
from pandas import DataFrame

# df_tennis = DataFrame.from
df_tennis = pd.read_csv('ID3.csv')

# df_tennis = DataFrame.from
df_tennis = pd.read_csv('ID3.csv')


# print(df_tennis)

# Calculate the Entropy of given probability
def entropy(probs):
    import math
    return sum([-prob * math.log(prob, 2) for prob in probs])


def entropy_of_list(a_list):  # Entropy calculation of list of discrete val ues(YES / NO)
    from collections import Counter
    cnt = Counter(x for x in a_list)
    print("No and Yes Classes:", a_list.name, cnt)
    num_instances = len(a_list) * 1.0
    probs = [x / num_instances for x in cnt.values()]
    return entropy(probs)  # Call Entropy:


# The initial entropy of the YES/NO attribute for our dataset.
# print(df_tennis['PlayTennis'])
total_entropy = entropy_of_list(df_tennis['PlayTennis'])
print("Entropy of given PlayTennis Data Set:", total_entropy)


def information_gain(df, split_attribute_name, target_attribute_name, trace=0):
    print("Information Gain Calculation of ", split_attribute_name)
    df_split = df.groupby(split_attribute_name)
    '''
 Takes a DataFrame of attributes,and quantifies the entropy of a target
 attribute after performing a split along the values of another attribute.
 '''  # print(df_split.groups)
    for name, group in df_split:
        print(name)
        print(group)
    # Calculate Entropy for Target Attribute, as well as
    # Proportion of Obs in Each Data-Split
    nobs = len(df.index) * 1.0
    # print("NOBS",nobs)
    df_agg_ent = df_split.agg({target_attribute_name: [entropy_of_list, lambda x: len(x) / nobs]})[
        target_attribute_name]
    # print("FAGGED",df_agg_ent)
    df_agg_ent.columns = ['Entropy', 'PropObservations']
    # if traced: # helps understand what fxn is doing:
    # Calculate Information Gain:
    new_entropy = sum(df_agg_ent['Entropy'] * df_agg_ent['PropObservations'])
    old_entropy = entropy_of_list(df[target_attribute_name])
    return old_entropy - new_entropy


# print('Info-gain for Outlook is :'+str( information_gain(df_tennis, 'Outlook', 'PlayTennis')),"\n")
# print('\n Info-gain for Humidity is: ' + str( information_gain(df_tennis,'Humidity', 'PlayTennis')),"\n")
# print('\n Info-gain for Wind is:' + str( information_gain(df_tennis, 'Wind', 'PlayTennis')),"\n")
# print('\n Info-gain for Temperature is:' + str( information_gain(df_tennis, 'Temperature','PlayTennis')),"\n")


def id3(df, target_attribute_name, attribute_names, default_class=None):  # Tally target attribute
    from collections import Counter
    cnt = Counter(x for x in df[target_attribute_name])  # class of YES /NO
    # First check: Is this split of the dataset homogeneous?
    if len(cnt) == 1:
        return next(iter(cnt))
    # Second check: Is this split of the dataset empty?
    # if yes, return a default value
    elif df.empty or (not attribute_names):
        return default_class
        # Otherwise: This dataset is ready to be divvied up!
    else:
        # [index_of_max] # most common value  of  target  attribute in dataset
        default_class = max(cnt.keys())
        # Choose Best Attribute to split on:
        gainz = [information_gain(df, attr, target_attribute_name)
                 for attr in attribute_names]
        index_of_max = gainz.index(max(gainz))
        best_attr = attribute_names[index_of_max]
        # Create an empty tree, to be populated in a moment
        tree = {best_attr: {}}
        remaining_attribute_names = [
            i for i in attribute_names if i != best_attr]
        # Split dataset
        # On each split, recursively call this algorithm.
        # populate the empty tree with subtrees, which
        # are the result of the recursive call
        for attr_val, data_subset in df.groupby(best_attr):
            subtree = id3(data_subset, target_attribute_name,
                          remaining_attribute_names, default_class)
            tree[best_attr][attr_val] = subtree
        return tree


# Predicting Attributes
attribute_names = list(df_tennis.columns)
print("List of Attributes:", attribute_names)
attribute_names.remove('PlayTennis')  # Remove the class attribute
print("Predicting Attributes:", attribute_names)

# Tree Construction

tree = id3(df_tennis, 'PlayTennis', attribute_names)
print("\n\nThe Resultant Decision Tree is :\n")
pprint(tree)


# Classification Accuracy
def classify(instance, tree, default=None):
    attribute = next(iter(tree))  # tree.keys()[0]
    if instance[attribute] in tree[attribute].keys():
        result = tree[attribute][instance[attribute]]
        if isinstance(result, dict):  # this is a tree, delve deeper
            return classify(instance, result)
        else:
            return result  # this is a label
    else:
        return default


df_tennis['predicted'] = df_tennis.apply(classify, axis=1, args=(tree, 'No'))
# classify func allows for a default arg: when tree doesn't have answered for a particular
# combination of attribute-values, we can use 'no' as the default guess
print('Accuracy is:' + str(sum(df_tennis['PlayTennis'] ==
      df_tennis['predicted']) / (1.0 * len(df_tennis.index))))
df_tennis[['PlayTennis', 'predicted']]

# Classification Accuracy: Training/Testing Set training_data = df_tennis.iloc[1:-4] # all but last thousand
# instances test_data = df_tennis.iloc[-4:] # just the last thousand train_tree = id3(training_data, 'PlayTennis',
# attribute_names) test_data['predicted2'] = test_data.loc(classify,axis=1,args=(train_tree,'Yes') ) # <----
# train_data tree print ('\n\n Accuracy is : ' + str( sum(test_data['PlayTennis']==test_data['predicted2'] ) / (
# 1.0*len(test_data.index)) ))

########################################################################################################################
# OUTPUT:
# Ignore single quotes at beginning and end
########################################################################################################################

'''No and Yes Classes: PlayTennis Counter({'yes': 9, 'no': 5})
Entropy of given PlayTennis Data Set: 0.9402859586706309
List of Attributes: ['Outlook', 'Temperature', 'Humidity', 'Wind', 'PlayTennis']
Predicting Attributes: ['Outlook', 'Temperature', 'Humidity', 'Wind']
Information Gain Calculation of  Outlook
overcast
     Outlook Temperature Humidity    Wind PlayTennis
2   overcast         hot     high    weak        yes
6   overcast        cool   normal  strong        yes
11  overcast        mild     high  strong        yes
12  overcast         hot   normal    weak        yes
rain
   Outlook Temperature Humidity    Wind PlayTennis
3     rain        mild     high    weak        yes
4     rain        cool   normal    weak        yes
5     rain        cool   normal  strong         no
9     rain        mild   normal    weak        yes
13    rain        mild     high  strong         no
sunny
   Outlook Temperature Humidity    Wind PlayTennis
0    sunny         hot     high    weak         no
1    sunny         hot     high  strong         no
7    sunny        mild     high    weak         no
8    sunny        cool   normal    weak        yes
10   sunny        mild   normal  strong        yes
No and Yes Classes: PlayTennis Counter({'yes': 4})
No and Yes Classes: PlayTennis Counter({'yes': 3, 'no': 2})
No and Yes Classes: PlayTennis Counter({'no': 3, 'yes': 2})
No and Yes Classes: PlayTennis Counter({'yes': 9, 'no': 5})
Information Gain Calculation of  Temperature
cool
    Outlook Temperature Humidity    Wind PlayTennis
4      rain        cool   normal    weak        yes
5      rain        cool   normal  strong         no
6  overcast        cool   normal  strong        yes
8     sunny        cool   normal    weak        yes
hot
     Outlook Temperature Humidity    Wind PlayTennis
0      sunny         hot     high    weak         no
1      sunny         hot     high  strong         no
2   overcast         hot     high    weak        yes
12  overcast         hot   normal    weak        yes
mild
     Outlook Temperature Humidity    Wind PlayTennis
3       rain        mild     high    weak        yes
7      sunny        mild     high    weak         no
9       rain        mild   normal    weak        yes
10     sunny        mild   normal  strong        yes
11  overcast        mild     high  strong        yes
13      rain        mild     high  strong         no
No and Yes Classes: PlayTennis Counter({'yes': 3, 'no': 1})
No and Yes Classes: PlayTennis Counter({'no': 2, 'yes': 2})
No and Yes Classes: PlayTennis Counter({'yes': 4, 'no': 2})
No and Yes Classes: PlayTennis Counter({'yes': 9, 'no': 5})
Information Gain Calculation of  Humidity
high
     Outlook Temperature Humidity    Wind PlayTennis
0      sunny         hot     high    weak         no
1      sunny         hot     high  strong         no
2   overcast         hot     high    weak        yes
3       rain        mild     high    weak        yes
7      sunny        mild     high    weak         no
11  overcast        mild     high  strong        yes
13      rain        mild     high  strong         no
normal
     Outlook Temperature Humidity    Wind PlayTennis
4       rain        cool   normal    weak        yes
5       rain        cool   normal  strong         no
6   overcast        cool   normal  strong        yes
8      sunny        cool   normal    weak        yes
9       rain        mild   normal    weak        yes
10     sunny        mild   normal  strong        yes
12  overcast         hot   normal    weak        yes
No and Yes Classes: PlayTennis Counter({'no': 4, 'yes': 3})
No and Yes Classes: PlayTennis Counter({'yes': 6, 'no': 1})
No and Yes Classes: PlayTennis Counter({'yes': 9, 'no': 5})
Information Gain Calculation of  Wind
strong
     Outlook Temperature Humidity    Wind PlayTennis
1      sunny         hot     high  strong         no
5       rain        cool   normal  strong         no
6   overcast        cool   normal  strong        yes
10     sunny        mild   normal  strong        yes
11  overcast        mild     high  strong        yes
13      rain        mild     high  strong         no
weak
     Outlook Temperature Humidity  Wind PlayTennis
0      sunny         hot     high  weak         no
2   overcast         hot     high  weak        yes
3       rain        mild     high  weak        yes
4       rain        cool   normal  weak        yes
7      sunny        mild     high  weak         no
8      sunny        cool   normal  weak        yes
9       rain        mild   normal  weak        yes
12  overcast         hot   normal  weak        yes
No and Yes Classes: PlayTennis Counter({'no': 3, 'yes': 3})
No and Yes Classes: PlayTennis Counter({'yes': 6, 'no': 2})
No and Yes Classes: PlayTennis Counter({'yes': 9, 'no': 5})
Information Gain Calculation of  Temperature
cool
  Outlook Temperature Humidity    Wind PlayTennis
4    rain        cool   normal    weak        yes
5    rain        cool   normal  strong         no
mild
   Outlook Temperature Humidity    Wind PlayTennis
3     rain        mild     high    weak        yes
9     rain        mild   normal    weak        yes
13    rain        mild     high  strong         no
No and Yes Classes: PlayTennis Counter({'yes': 1, 'no': 1})
No and Yes Classes: PlayTennis Counter({'yes': 2, 'no': 1})
No and Yes Classes: PlayTennis Counter({'yes': 3, 'no': 2})
Information Gain Calculation of  Humidity
high
   Outlook Temperature Humidity    Wind PlayTennis
3     rain        mild     high    weak        yes
13    rain        mild     high  strong         no
normal
  Outlook Temperature Humidity    Wind PlayTennis
4    rain        cool   normal    weak        yes
5    rain        cool   normal  strong         no
9    rain        mild   normal    weak        yes
No and Yes Classes: PlayTennis Counter({'yes': 1, 'no': 1})
No and Yes Classes: PlayTennis Counter({'yes': 2, 'no': 1})
No and Yes Classes: PlayTennis Counter({'yes': 3, 'no': 2})
Information Gain Calculation of  Wind
strong
   Outlook Temperature Humidity    Wind PlayTennis
5     rain        cool   normal  strong         no
13    rain        mild     high  strong         no
weak
  Outlook Temperature Humidity  Wind PlayTennis
3    rain        mild     high  weak        yes
4    rain        cool   normal  weak        yes
9    rain        mild   normal  weak        yes
No and Yes Classes: PlayTennis Counter({'no': 2})
No and Yes Classes: PlayTennis Counter({'yes': 3})
No and Yes Classes: PlayTennis Counter({'yes': 3, 'no': 2})
Information Gain Calculation of  Temperature
cool
  Outlook Temperature Humidity  Wind PlayTennis
8   sunny        cool   normal  weak        yes
hot
  Outlook Temperature Humidity    Wind PlayTennis
0   sunny         hot     high    weak         no
1   sunny         hot     high  strong         no
mild
   Outlook Temperature Humidity    Wind PlayTennis
7    sunny        mild     high    weak         no
10   sunny        mild   normal  strong        yes
No and Yes Classes: PlayTennis Counter({'yes': 1})
No and Yes Classes: PlayTennis Counter({'no': 2})
No and Yes Classes: PlayTennis Counter({'no': 1, 'yes': 1})
No and Yes Classes: PlayTennis Counter({'no': 3, 'yes': 2})
Information Gain Calculation of  Humidity
high
  Outlook Temperature Humidity    Wind PlayTennis
0   sunny         hot     high    weak         no
1   sunny         hot     high  strong         no
7   sunny        mild     high    weak         no
normal
   Outlook Temperature Humidity    Wind PlayTennis
8    sunny        cool   normal    weak        yes
10   sunny        mild   normal  strong        yes
No and Yes Classes: PlayTennis Counter({'no': 3})
No and Yes Classes: PlayTennis Counter({'yes': 2})
No and Yes Classes: PlayTennis Counter({'no': 3, 'yes': 2})
Information Gain Calculation of  Wind
strong
   Outlook Temperature Humidity    Wind PlayTennis
1    sunny         hot     high  strong         no
10   sunny        mild   normal  strong        yes
weak
  Outlook Temperature Humidity  Wind PlayTennis
0   sunny         hot     high  weak         no
7   sunny        mild     high  weak         no
8   sunny        cool   normal  weak        yes
No and Yes Classes: PlayTennis Counter({'no': 1, 'yes': 1})
No and Yes Classes: PlayTennis Counter({'no': 2, 'yes': 1})
No and Yes Classes: PlayTennis Counter({'no': 3, 'yes': 2})


The Resultant Decision Tree is :

{'Outlook': {'overcast': 'yes',
             'rain': {'Wind': {'strong': 'no', 'weak': 'yes'}},
             'sunny': {'Humidity': {'high': 'no', 'normal': 'yes'}}}}
Accuracy is:1.0'''
