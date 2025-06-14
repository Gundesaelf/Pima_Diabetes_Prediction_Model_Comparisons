# |---------------------------------IMPORTS---------------------------------|

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import cross_val_score
from colorama import Fore, init # Because I like my sections to be distinct and easy to read

# |---------------------------------WRANGLING_DATA---------------------------------|

init() # Initialize colorama (for those who don't know)

fname = 'diabetes.csv'
df = pd.read_csv(fname)
pd.set_option('display.max_columns', 9) # After running df.head(), I saw that there's 9 columns and I'd like to see them all

print(Fore.LIGHTGREEN_EX + '\n\nShape:', df.shape, '\n')
print(Fore.LIGHTYELLOW_EX)
print('\nBefore:\n\n', df.head()) # To show me what I'm working with before I start cleaning the data

cols_with_zeros = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI'] # I'm  filling in the missing values in the next couple lines

df[cols_with_zeros] = df[cols_with_zeros].replace(0, np.nan) # Replace 0s with NaN so I can fill with mean/median/mode  without issues
df[cols_with_zeros] = df[cols_with_zeros].fillna(df[cols_with_zeros].mean()) # Mean gives better results than median or mode

print(Fore.LIGHTBLUE_EX)
print('\nAfter:\n\n', df.head()) # To show me what changes I made to the data

# |---------------------------------MODEL---------------------------------|

model =  RandomForestClassifier(class_weight= 'balanced')
X = df.drop(columns= ['Age', 'Outcome'])  # I dropped 'Age' because I found it reduces accuracy
y = df['Outcome']

# |---------------------------------PREDICTION_TEST---------------------------------|

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, random_state= 42)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

for actual, predicted in zip(y_test, y_pred):
    print(Fore.LIGHTMAGENTA_EX)
    print(f'Actual:{actual} || Predicted: {predicted}') # Show the actual and predicted values for each test case

print(Fore.LIGHTCYAN_EX)
print('\n\nClassification Report:\n\n', classification_report(y_test, y_pred))

print(Fore.LIGHTGREEN_EX + f'\n\nAccuracy: {accuracy_score(y_test, y_pred):.2%}\n\n')

# |---------------------------------CROSS_VALIDATION---------------------------------|

scores = cross_val_score(model, X, y, cv=10)  # 10 fold cross-validation
print(f'Mean accuracy: {scores.mean():.2%}')
print(f'Std deviation: {scores.std():.2%}\n\n')
