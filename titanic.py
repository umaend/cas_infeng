#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 11:47:41 2019

@author: ursmaendli
"""

# Uebung Titanic

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import export_graphviz
import pydot

# read data:
train = pd.read_csv('../titanic/titanic3_train.csv', sep = ';')
test = pd.read_csv('../titanic/titanic3_test.csv', sep = ';')

# concat two data partitions and define a train/test-attribute:
train['partition'] = 'train'
test['partition'] = 'test'
     
frames = [train, test]
df = pd.concat(frames, sort = False)

# Choose variables for modelling:
df = df[['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'boat', 'body', 'survived', 'partition']]

# alle nan in den Daten bei age und fare mit dem jeweiligen means, gemessen über die Klasse, ersetzen:
means = df.groupby('pclass').mean()

for index, row in means.iterrows():    
    df.loc[(df.age.isnull()) & (df.pclass == index), 'age'] = means.loc[index, 'age']
    df.loc[(df.fare.isnull()) & (df.pclass == index), 'fare'] = means.loc[index, 'fare']

def replace_nan(df, col, value_na, value_not_na):
    '''
    Replaces nan in df.col with value_na
    and not nan with value_not_na
    '''
    mask_na = df[col].isnull()
    mask_not_na = df[col].notnull()
    df.loc[mask_na, col] = value_na
    df.loc[mask_not_na, col] = value_not_na
    return df

# alle nan in body und boat ersetzen:
replace_nan(df, 'boat', 'nein', 'ja')
replace_nan(df, 'body', 'nein', 'ja')

### Random Forest
#siege z.B. hier: https://towardsdatascience.com/random-forest-in-python-24d0893d51c0

# pandas df in numpy array konvertieren.
# labels are the values we want to predict:
labels = np.array(df.loc[df['partition'] == 'train', 'survived'])

# remove labels and partition from the features,
# axis = 1 refers to the columns
features_train = df.loc[df['partition'] == 'train'].drop(['survived', 'partition'], axis = 1)
features_test = df.loc[df['partition'] == 'test'].drop(['survived', 'partition'], axis = 1)

# One-hot encoding:
features_train = pd.get_dummies(features_train)
features_test = pd.get_dummies(features_test)

# für später sichern:
feature_list = list(features_train)

# und ebenfalls in array konvertieren:
features_train = np.array(features_train)
features_test = np.array(features_test)

# kontrollieren:
print('Training Features Shape:', features_train.shape)
print('Training Labels Shape:', labels.shape)
print('Testing Features Shape:', features_test.shape)

# Klassifizierer nehmen, nicht Regressor!
# Random Forest
rf = RandomForestClassifier(n_estimators = 1000, random_state = 42)
rf.fit(features_train, labels)

# an den Testdaten predicten:
predictions = rf.predict(features_test)

#Output ausgeben (Form: key, value)
np.column_stack((features_test[:,0], predictions))
resultat = pd.concat([test[['id']], pd.DataFrame(predictions)], axis = 1, ignore_index=False)
resultat.to_csv('resultat.csv', index = False, sep = ';')


# Einen einzelnen Baum als Beispiel plotten:
tree = rf.estimators_[5]
export_graphviz(tree, out_file = 'tree.dot', feature_names = feature_list, rounded = True, precision = 1)
(graph, ) = pydot.graph_from_dot_file('tree.dot')
graph.write_png('tree.png')