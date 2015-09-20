import json
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier

def getItems(recipe):
    itemList = set()
    for a in recipe:
        for b in a.split(" "):
            itemList.add(b)
    return itemList


def getFeatureArray(itemlist,featureList,featureArray):
    items = getItems(itemlist)
    for item in items:
        if item in featureList:
            featureArray[item] = 1
    return featureArray


def assignCuisinId(df):
    cuisineId = []
    cuisine = {}
    for index, row in df.iterrows():
        value = sum(bytearray(row['cuisine'],encoding='utf8'))
        cuisineId.append(value)
        cuisine[value] = row['cuisine']
    return cuisineId,cuisine


def getFeatures(ingredientsDict,limits,numberOfFeatures):
    selectedIngredients = []
    for item in ingredientsDict:
        if limits[0] <= ingredientsDict[item] <= limits[1]:
            selectedIngredients.append(item)
    return selectedIngredients[0:numberOfFeatures]

print("Loading data...")
with open('train.json') as train_data_file:
    train_data_json = json.load(train_data_file)
with open('test.json') as test_data_file:
    test_data_json = json.load(test_data_file)
 
train_data = []
test_data = []

for line in train_data_json:
    train_data.append(line)
for line in test_data_json:
    test_data.append(line)

df_train = pd.DataFrame(train_data)
df_test = pd.DataFrame(test_data)
df_test['cuisine'] = 'Null'

# Feature engineering
print("Selecting ingredients for features...")
recipe = np.array(df_train['ingredients'].values)
ingredients_dict = {}
for i in range(0,len(recipe)):
    for word in recipe[i][:]:
        for item in word.split(" "):
            if len(item)>3:
                try:
                    ingredients_dict[item] += 1
                except:
                    ingredients_dict[item] = 1
ingredients_dict1 = sorted(ingredients_dict.items(), key = lambda x:x[1])
numberOfFeatures = 1000
limits = [100,15000]
selectedIngredients = getFeatures(ingredients_dict,limits,numberOfFeatures)
numberOfFeatures = len(selectedIngredients)

# Assign cuisine names to an interger -> Sum(ascii_values)
print("Assigning cuisine ids...")
cuisineId, cuisine = assignCuisinId(df_train)
df_train['cuisineId'] = cuisineId
df_test['cuisineId'] = 0
index = ['id', 'cuisineId', 'cuisine', 'ingredients']
df_train = df_train[index]
df_test = df_test[index]
df_train_test = pd.concat([df_train,df_test])

# featureArray = pd.Series(0,index=selectedIngredients)
# Cronstruct the feature matrix containing the features of all the entries
print("Constructing feature matrix...")
featureMatrix = np.zeros((len(df_train_test.index),numberOfFeatures),dtype = int)
i = 0
for index,row in df_train_test.iterrows():
    itemList = getItems(row['ingredients'])
    featureMatrix[i,:] = getFeatureArray(itemList,selectedIngredients,pd.Series(0,index = selectedIngredients))
    i += 1


# Add this feature matrix to the data frame
i = 0
for index in selectedIngredients:
    df_train_test[index] = featureMatrix[:,i]
    i += 1

index = selectedIngredients
index.insert(0, 'cuisineId')
df_learn = df_train_test[index]
df_learn_values = df_learn.values


print("Random Forest...")

forest = RandomForestClassifier(n_estimators=100)
forest = forest.fit(df_learn_values[0:len(df_train.index)-1, 0:numberOfFeatures],
                    df_learn_values[0:len(df_train.index)-1, 0])
output = forest.predict(df_learn_values[len(df_train.index):, 0:numberOfFeatures])
cuisineOutput = np.array(output,dtype=object)

for i in range(0,len(df_test.index)-1):
    cuisineOutput[i] = cuisine[output[i]]

output_df = pd.DataFrame([df_test['id'],cuisineOutput], index = ['id','cuisine'])
output_df.to_csv('sample_submission.csv',coloumns=['id','cuisine'])


