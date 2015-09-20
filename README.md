# Kaggle-WhatsCooking-MachineLearning


  Source - [Kaggle - What's Cooking](https://www.kaggle.com/c/whats-cooking)
  
  Objective - Use recipe ingredients to predict the category of cuisine
  
  Language - Python
  
  Packages - NumPy, Pandas, Sci-Kit Learn
  
##### Data file overview

      >>df_train.info()
      <class 'pandas.core.frame.DataFrame'>
      Int64Index: 39774 entries, 0 to 39773
      Data columns (total 3 columns):
      cuisine        39774 non-null object
      id             39774 non-null int64
      ingredients    39774 non-null object
      dtypes: int64(1), object(2)
      
      >>df_train.head()
            cuisine     id                                        ingredients
      0        greek  10259  [romaine lettuce, black olives, grape tomatoes...
      1  southern_us  25693  [plain flour, ground pepper, salt, tomatoes, g...
      2     filipino  20130  [eggs, pepper, salt, mayonaise, cooking oil, g...
      3       indian  22213                [water, vegetable oil, wheat, salt]
      4       indian  13162  [black pepper, shallots, cornflour, cayenne pe...
      
##### Feature Engineering
The list of words in all the ingredients can be used as a set of feature for each cuisine. A condition that the word has to be greater than 3 (>3) was used to get rid of unwanted words like a, on and etc. Also words that appear in the range [100,15000] were selected as features. This is a range that can be tuned, or we can do PCA to reduce the dimensions.

Once this list of ingredients is selected to act as features, assign 1 or 0 to each cuisine's feature list depending on whether the ingredient is present or not in an entry. A new data frame was created as shown below:

        cuisineId     cuisine     lettuce   olive   tomatoes   beef  - - - 
              738     italian           0       1          1      0  - - -
              422        thai           1       1          1      0  - - -
              
The cuisineId = sum(bytearray('cuisine',encoding = 'utf8') through the cuisine is identified by a numerical value. 

##### Random Forest Classifier

Python Sci-Kit learn has built-in random forest classified. Use it to train the forest using the train data with the labels given. The pd.DataFrame df_learn contains both the train and test data combined. This was necessary since features were constructed, and all the entries were given values for these features. So it looks like this:

      cuisineId lettuce   olive   tomatoes   beef  - - - 
            738       0       1          1      0           // Train
            422       1       1          1      0
              .       .       .          .      .      
              0       0       1          0      0           // Test
              0       1       0          0      1

        forest = RandomForestClassifier(n_estimators=100)
        forest = forest.fit(df_learn_values[0:len(df_train.index)-1, 0:numberOfFeatures],
                    df_learn_values[0:len(df_train.index)-1, 0])
        output = forest.predict(df_learn_values[len(df_train.index):, 0:numberOfFeatures])

The output will be a numpy array of integers which are the cuisineId. Use a dictionary to store these ids and cuisine names while assigning these ids. Later use this to get the cuisine name.

##### Kaggle Results

        Accuracy - 0.57230
