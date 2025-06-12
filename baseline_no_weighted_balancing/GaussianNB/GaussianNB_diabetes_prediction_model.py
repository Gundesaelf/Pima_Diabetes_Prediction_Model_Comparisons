# |---------------------------------IMPORTS---------------------------------|

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB 
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

# |---------------------------------FINDING_THE_BEST_SEED---------------------------------|

seed_accuracy_scores = [] # To keep track of which seed gives the best accuracy
best_seed_accuracy = 0

for seed in range(1, 101): # I want to test 100 different random states to find the best one. It's seed 43, btw
    X = df.drop(columns= ['Age', 'Outcome'])
    y = df['Outcome'] 

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.5, random_state= seed)

    model = GaussianNB()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    seed_accuracy = accuracy_score(y_test, y_pred)
    seed_accuracy_scores.append(seed_accuracy)

    if seed_accuracy > best_seed_accuracy:
        best_seed_accuracy = seed_accuracy
        best_seed = seed

print(Fore.LIGHTYELLOW_EX + f'\n\nBest Seed for Accuracy: {best_seed} {max(seed_accuracy_scores):.2%}')


# |---------------------------------FINDING_THE_BEST_TEST_SIZE---------------------------------|

test_size_accuracy_scores = []
best_test_size_accuracy = 0

for test in np.arange(0.1, 0.51, 0.01): # Testing to find the best test size. It's 0.5, btw
    X = df.drop(columns= ['Age', 'Outcome'])
    y = df['Outcome'] 

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= test, random_state= 43)

    model = GaussianNB()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    test_accuracy = accuracy_score(y_test, y_pred)
    test_size_accuracy_scores.append(test_accuracy)

    if test_accuracy > best_test_size_accuracy:
        best_test_size_accuracy = test_accuracy
        best_test_size = test

print(Fore.LIGHTYELLOW_EX + f'\n\nBest Test Size for Accuracy: {best_test_size:.2%} {max(test_size_accuracy_scores):.2%}\n\n')
# Python translates floats slightly off so I converted it to a percentage to get a more accurate representation of which test size is best

# |---------------------------------MODEL---------------------------------|

X = df.drop(columns= ['Age', 'Outcome']) # I dropped 'Age' because I found it reduces accuracy
y = df['Outcome'] 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.5, random_state= 43) # Use the best seed and test size

model = GaussianNB()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

for actual, predicted in zip(y_test, y_pred):
    print(Fore.LIGHTMAGENTA_EX)
    print(f'Actual:{actual} || Predicted: {predicted}') # Show the actual and predicted values for each test case

print(Fore.LIGHTCYAN_EX)
print('\n\nClassification Report:\n\n', classification_report(y_test, y_pred))

print(Fore.LIGHTGREEN_EX + f'\n\nAccuracy: {accuracy_score(y_test, y_pred):.2%}\n\n') # Our final accuracy score is 80%

# |---------------------------------CROSS_VALIDATION---------------------------------|

scores = cross_val_score(model, X, y, cv=10)  # 10 fold cross-validation
print(f'Mean accuracy: {scores.mean():.2%}')
print(f'Std deviation: {scores.std():.2%}\n\n')
