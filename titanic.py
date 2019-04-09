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

# mac:
# df = pd.read_csv('/Users/ursmaendli/Documents/02 zhaw/CAS_InfEng/Scripting/titanic3_train.csv', sep = ';')
# df_test = pd.read_csv('/Users/ursmaendli/Documents/02 zhaw/CAS_InfEng/Scripting/titanic3_test.csv', sep = ';')

# linux:
df = pd.read_csv('/home/umaend/Documents/zhaw/scripting/titanic/titanic3_train.csv', sep = ';')
df_test = pd.read_csv('/home/umaend/Documents/zhaw/scripting/titanic/titanic3_test.csv', sep = ';')


# Schöner machen!
means = df.groupby('pclass').mean()

values_1 = {'fare' : 88.4387, 'age' : 39.3127}
values_2 = {'fare' : 20.8136, 'age' : 29.7977}
values_3 = {'fare' : 13.2293, 'age' : 14.8521}

# ersetzen in train-Datensatz
df_1 = df.loc[df['pclass'] == 1].fillna(value = values_1)
df_2 = df.loc[df['pclass'] == 2].fillna(value = values_2)
df_3 = df.loc[df['pclass'] == 3].fillna(value = values_3)

df = pd.concat([df_1,df_2,df_3])
df = df.loc[df['pclass'] == 3].fillna(value = values_3)

# ersetzen in test-Datensatz (Werte nehmen aus train-Datensatz):
df_1_test = df_test.loc[df['pclass'] == 1].fillna(value = values_1)
df_2_test = df_test.loc[df['pclass'] == 2].fillna(value = values_2)
df_3_test = df_test.loc[df['pclass'] == 3].fillna(value = values_3)

df_test = pd.concat([df_1_test,df_2_test,df_3_test])


# preview von allem:
print(df.to_string())

# Dataset untersuchen und plotten:
df.describe()

df.plot.scatter(x = 'pclass',
                y = 'age',
                c = 'survived')

survived = df.loc[df['survived'] == 1]
died = df.loc[df['survived'] == 0]
survived['age'].plot.hist()
died['age'].plot.hist()
df['age'].plot.hist()

# Choose variables for modelling:
df_model = df[['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'boat', 'body', 'survived']]
df_model_test = df_test[['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'boat', 'body']]

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

# alle nan in body und boat in beiden df ersetzen:
replace_nan(df_model, 'boat', 'nein', 'ja')
replace_nan(df_model_test, 'boat', 'nein', 'ja')
replace_nan(df_model, 'body', 'nein', 'ja')
replace_nan(df_model_test, 'body', 'nein', 'ja')


# One-hot encoding:
df_model = pd.get_dummies(df_model)
df_model_test = pd.get_dummies(df_model_test)

# Random Forest siehe hier:
#https://towardsdatascience.com/random-forest-in-python-24d0893d51c0

# pandas df in numpy array konvertieren.
# labels are the values we want to predict:
labels = np.array(df_model['survived'])

# remove labels from the features,
# axis = 1 refers to the columns
# (nur train-Daten):
features = df_model.drop('survived', axis = 1)

# für später sichern:
feature_list = list(features.columns)

# und ebenfalls in array konvertieren:
features = np.array(features)
features_test = np.array(df_model_test)

# kontrollieren:
print('Training Features Shape:', features.shape)
print('Training Labels Shape:', labels.shape)
print('Testing Features Shape:', features_test.shape)

# Klassifizierer nehmen, nicht Regressor!
# Random Forest
rf = RandomForestClassifier(n_estimators = 1000, random_state = 42)
rf.fit(features, labels)

# an den Testdaten predicten:
predictions = rf.predict(features_test)

#Output ausgeben (Form: key, value)
np.column_stack((features_test[:,0], predictions))
resultat = pd.concat([df_test[['id']], pd.DataFrame(predictions)], axis = 1, ignore_index=False)
resultat.to_csv('resultat.csv', index = False, sep = ';')


# Einen einzelnen Baum als Beispiel plotten:
tree = rf.estimators_[5]
export_graphviz(tree, out_file = 'tree.dot', feature_names = feature_list, rounded = True, precision = 1)
(graph, ) = pydot.graph_from_dot_file('tree.dot')
graph.write_png('tree.png')