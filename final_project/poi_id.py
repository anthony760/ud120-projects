#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from math import isnan
from sklearn.feature_selection import SelectPercentile, SelectKBest,f_classif
from sklearn.metrics import accuracy_score, precision_score, recall_score


def print_dataset_info(data_set_dict, features_list_array):
    # Print basic information about data
    print('\n---DATASET INFORMATION---')
    print('Total data points in dataset: {}'.format(len(data_set_dict)))
    poi_count = 0
    for name in data_set_dict:
        poi_count += 1 if data_dict[name]['poi'] is True else 0
    print('Total persons of interest: {}'.format(poi_count))
    print('Total persons of non-interest: {}'.format(len(data_set_dict) - poi_count))
    print('Total number of features per person: {}'.format(len(features_list_array)))


def compute_fraction(poi_messages, all_messages):
    fraction = 0.
    if isnan(float(poi_messages)) or isnan(float(all_messages)) or all_messages == 0:
        pass
    else:
        fraction = float(poi_messages) / float(all_messages)
    return fraction


### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
#Removed 'email_address' from features list as it is a string
features_list = ['poi','salary', 'deferral_payments', 'total_payments',
                 'loan_advances', 'bonus', 'restricted_stock_deferred',
                 'deferred_income', 'total_stock_value', 'expenses',
                 'exercised_stock_options', 'other', 'long_term_incentive',
                 'restricted_stock', 'director_fees', 'to_messages',
                 'from_poi_to_this_person', 'from_messages',
                 'from_this_person_to_poi', 'shared_receipt_with_poi']
# You will need to use more features

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

print_dataset_info(data_dict, features_list)

### Task 2: Remove outliers

# Find people missing all values
persons_missing_values = {}
people_to_remove = []
print('\nFinding people missing all data points:')
for name in data_dict:
    missing_values = 0
    for feature in features_list:
        missing_values += 1 if isnan(float(data_dict[name][feature])) else 0
    # Warn if a person is missing all feature values less the POI bool
    if missing_values >= len(features_list) - 1:
        print('WARNING: {} missing {} data points.'.format(name, missing_values))
        people_to_remove.append(name)

# If a person has zero feature values, remove them
if len(people_to_remove) == 0:
    print('\nNo people to remove.')
else:
    print('\nRemoving people missing all data points:')
    for person in people_to_remove:
        print('REMOVING: {}'.format(person))
        data_dict.pop(person)

# Removing 'TOTAL' and 'THE TRAVEL AGENCY IN THE PARK' users as they are not real employees per FindLaw.org spreadsheet
not_real_employees = ['TOTAL', 'THE TRAVEL AGENCY IN THE PARK']
for entity in not_real_employees:
    data_dict.pop(entity)

# Sort and print the total missing values from each feature
features_missing_values = {}
for feature in features_list:
    missing_values = 0
    for name in data_dict:
        missing_values += 1 if isnan(float(data_dict[name][feature])) else 0
    features_missing_values[feature] = missing_values
features_missing_values = sorted(features_missing_values.items(), key=lambda x: x[1], reverse=True)
print('\nFeatures missing values:')
# Start of list of features to remove
features_to_remove = []
for feature in features_missing_values:
    # Calculate percentage missing
    percent_missing = (feature[1]/float(len(data_dict))) * 100
    # Add features to remove list if they are missing more than 80% of data
    features_to_remove.append(feature[0]) if percent_missing >= 80.0 else None
    print('--{}: {}, {:.1f}%'.format(feature[0], feature[1], percent_missing))

# Add 'other' feature to removal list as various payments are mixed into one catchall category
features_to_remove.append('other')

# Create custom list of features
my_feature_list = features_list
print('\nRemoving the following features:')
for feature in features_to_remove:
    print('REMOVING: {}'.format(feature))
    my_feature_list.remove(feature)

# Print updated information about dataset
print_dataset_info(data_dict, features_list)

### Task 3: Create new feature(s)

# Create fraction ratio for messages sent to/from person of interest
for name in data_dict:
    data_point = data_dict[name]

    from_poi_to_this_person = data_point['from_poi_to_this_person']
    to_messages = data_point['to_messages']
    data_point['fraction_from_poi'] = compute_fraction(from_poi_to_this_person, to_messages)

    from_this_person_to_poi = data_point['from_this_person_to_poi']
    from_messages = data_point['from_messages']
    data_point['fraction_to_poi'] = compute_fraction(from_this_person_to_poi, from_messages)

# Add new features to custom feature list
my_feature_list.append('fraction_from_poi')
my_feature_list.append('fraction_to_poi')

### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, my_feature_list, sort_keys=True)
labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers

selector = SelectKBest(f_classif, k=8)
selector.fit_transform(features, labels)
scores = dict(zip(my_feature_list[1:], selector.scores_))
scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
print('\nTop Features (SelectKBest):')
for score in scores:
    print('--{}: {},'.format(score[0], score[1]))
#optimized_features_list = 'poi' + list(map(lambda x: x[0], scores))[0:8]
#print(optimized_features_list)

### Please name your classifier clf for easy export below.

### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
from sklearn.naive_bayes import GaussianNB
clf1 = GaussianNB()

### Task 5: Tune your classifier to achieve better than .3 precision and recall
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info:
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
from sklearn.model_selection import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

# Feature selection

selector = SelectPercentile(f_classif, percentile=30)
selector.fit(features_train, labels_train)
features_train_transformed = selector.transform(features_train)
features_test_transformed = selector.transform(features_test)

print('\nTesting GaussianNB Classifier')
clf1.fit(features_train_transformed, labels_train)
pred1 = clf1.predict(features_test_transformed)
accuracy1 = accuracy_score(pred1, labels_test)
print('Accuracy Score: {}'.format(accuracy1))
precision1 = precision_score(pred1, labels_test)
print('Precision Score: {}'.format(precision1))
recall1 = recall_score(pred1, labels_test)
print('Recall Score: {}'.format(recall1))

print('\nTesting DecisionTree Classifier')
from sklearn.tree import DecisionTreeClassifier
clf2 = DecisionTreeClassifier()
clf2.fit(features_train_transformed, labels_train)
pred2 = clf2.predict(features_test_transformed)
accuracy2 = accuracy_score(pred2, labels_test)
print('Accuracy Score: {}'.format(accuracy2))
precision2 = precision_score(pred2, labels_test)
print('Precision Score: {}'.format(precision2))
recall2 = recall_score(pred2, labels_test)
print('Recall Score: {}'.format(recall2))

clf = clf1 if accuracy1 > accuracy2 else clf2

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, my_feature_list)
