import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv('../data/raw/phase1_movie_reviews-train.csv')

df = pd.get_dummies(df['polarity'], drop_first=True)

train, test = train_test_split(df, test_size=0.2)
validation, test = train_test_split(test, test_size=0.5)

def predictallpos(dataframe):
    return dataframe["positive"].sum()/len(dataframe)

print("Accuracies")
print("Training: "+str(predictallpos(train)))
print("Testing: "+str(predictallpos(test)))
print("Validation: " + str(predictallpos(validation)))