import pandas as pd
import numpy as np

print("Mental health is a fundamental aspect of overall well-being, encompassing our emotional, psychological, and social health. It influences how we think, feel, and act, shaping our ability to handle stress, relate to others, and make decisions. Despite its critical importance, mental health often remains overshadowed by stigma and misinformation, leading to inadequate awareness and support.")

df = pd.read_csv("D:\data set\mental health dataset.csv")
df.drop("number", axis="columns" , inplace=True)
df['State'] = df['State'].str.lower()  # converting target variables to lower case

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df["State_le"]=le.fit_transform(df['State'])

target = df["State_le"]
features = df.drop(["State_le" , "State"] , axis = 1 )

# storing the max and min value of the features
column_name = list(features.columns)
max_feature_value =[]
min_feature_value =[]
for name in column_name:
    max_feature_value.append(features[name].max())
    min_feature_value.append(features[name].min())

# taking User input of features
inputs = []
for i in range(0 , len(features.columns)):
    print("Enter the : " , features.columns[i] , "\n Max Value : " , max_feature_value[i] , " Min Value : " , min_feature_value[i]  )
    value = int(input())
    inputs.append(value)

#model traning 
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators= 35 , max_depth=14 , min_samples_leaf= 2 , min_samples_split=13)
model.fit(features , target)
# Make prediction
prediction = model.predict([np.array(inputs)])[0]
state_name = le.inverse_transform([prediction])[0] if "State_le" in df.columns else prediction

print("Predicted State:", state_name)