#!/usr/bin/env python3

import pandas as pd
import numpy as np
import random
import tensorflow.keras as keras

def preprocess(x):
    return (x - np.mean(x)) / np.std(x)

df1 = pd.read_csv("donor_data.csv", low_memory=False)
df2 = pd.read_csv("survey_data.csv")

all_columns = df2.columns

inner_join = pd.merge(df1, 
                      df2, 
                      on ='Zip', 
                      how ='inner')

inner_join = inner_join.drop("Zip", axis=1)
with_NA = inner_join[inner_join.columns[inner_join.isna().any()]].columns
for column_name in with_NA:
    #print(column_name)
    x = inner_join[column_name].mean()
    inner_join[column_name].fillna(x, inplace = True)

inner_join[inner_join.columns[inner_join.isna().any()]]


# code below gets rid of any columns that didn't have something in it to find the mean of
columns_to_drop = inner_join[inner_join.columns[inner_join.isna().any()]].columns

# drop all N/A columns that are completely empty
if(len(columns_to_drop) != 0): 
    inner_join = inner_join.drop(columns_to_drop, axis =1 )

inner_join = pd.get_dummies(inner_join, columns=['Gender'])

# find the unique majors in this dataset
unique_majors = inner_join.Major.unique()

#create a mapping dictionary for these majors
major_dict = {}
for i in range(len(unique_majors)):
    major_dict[unique_majors[i]] = i
major_dict["Other"] = len(unique_majors)


# replace with column encodings 
inner_join["Major"] = inner_join["Major"].replace(major_dict)

#append column to the end to make my life significantly easier when doing the embedding layer later
save_column = inner_join["Major"]
inner_join.drop("Major", axis=1, inplace=True)
inner_join["Major"] = save_column

# make my life easier by putting the target value on the end
target = inner_join["Donor_Category"]
inner_join.drop("Donor_Category", axis=1, inplace=True)
inner_join["target"] = target

#even up classes
g = inner_join.groupby('target')
inner_join = g.apply(lambda x: x.sample(g.size().min()).reset_index(drop=True))


# get values into a numpy array
X = inner_join.iloc[:, :-1].values
Y = inner_join.iloc[:, -1].values

#seperate numberical data (x1) from the major categorical data (x2)
x1 = X[:,:-1]
x2 = X[:,-1:]

#reshape Y
Y = Y.reshape((len(Y), 1))

#preprocess x
x1 = preprocess(x1)

shuffle = np.random.permutation(x1.shape[0])
x1_shuffled = x1[shuffle,:]
x2_shuffled = x2[shuffle,:]
Y_shuffled = Y[shuffle,:]

index = int(len(x1_shuffled)*0.1)
x1_test = x1_shuffled[-index:,:]
x2_test = x2_shuffled[-index:,:]
Y_test = Y_shuffled[-index:,:]

x1_train = x1_shuffled[:-index,:]
x2_train = x2_shuffled[:-index,:]
Y_train = Y_shuffled[:-index,:]

num_classes = len(np.unique(Y_shuffled))


major_embed_size = 50
embed_dim = 500
ff_dim = 250


input1 = keras.layers.Input(shape=x1.shape[1:])
input2 = keras.layers.Input(shape=x2.shape[1:])

#embed our majors as vectors of size 50
embed_input = keras.layers.Embedding(input_dim=len(major_dict),
                                     output_dim=major_embed_size)(input2)
embed_input = keras.layers.Flatten()(embed_input)

#concatenate our input together
y = keras.layers.Concatenate()([input1,embed_input]) 

y = keras.layers.Dense(embed_dim)(y)
for _ in range(3):
    i = y
    y = keras.layers.Dense(ff_dim,activation='gelu',activity_regularizer=keras.regularizers.l2(0.01))(y)
    y = keras.layers.Dense(embed_dim)(y)
    y = keras.layers.Add()([i,y])
    y = keras.layers.BatchNormalization()(y)
    y = keras.layers.Dropout(0.35)(y)
output = keras.layers.Dense(num_classes, activation=keras.activations.softmax)(y)
model = keras.Model([input1, input2], output)

model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
                  loss=keras.losses.SparseCategoricalCrossentropy(), 
                  metrics=[keras.metrics.SparseCategoricalAccuracy()])

history = model.fit([x1_train, x2_train], Y_train,
                    validation_split=0.2,
                    epochs=50, 
                    batch_size = 64,
                    verbose=1)

print("Validation accuracy:",*["%.8f"%(x) for x in 
                               history.history['val_sparse_categorical_accuracy']])

accuracy = model.evaluate([x1_test, x2_test], Y_test)
print("Final Test:", accuracy[1])