from datetime import date, datetime
from functools import reduce
import holidays
import pandas as pd
from sklearn.model_selection import train_test_split
from catboost import CatBoostRegressor, Pool

hk_2017_holidays = holidays.HongKong(years = 2017)
hk_2018_holidays = holidays.HongKong(years = 2018)

dataset = pd.read_csv('train.csv')
dataset["date"] = pd.to_datetime(dataset["date"], format="%d/%m/%Y %H:%M")
dataset["year"] = dataset["date"].apply(lambda x: x.year)
dataset["month"] = dataset["date"].apply(lambda x: x.month)
dataset["day"] = dataset["date"].apply(lambda x: x.day)
dataset["hour"] = dataset["date"].apply(lambda x: x.hour)
dataset["weekday"] = dataset["date"].apply(lambda x: x.isoweekday())
dataset['hour_of_week'] = (dataset["weekday"] * 24 - 24) + dataset["hour"]
dataset['hour_of_month'] = (dataset["month"] * 24 - 24) + dataset["hour"]
dataset["quarter"] = dataset["date"].apply(lambda x: x.quarter)
dataset["season"] = dataset["month"].apply(lambda x: (x%12 + 3)//3)
dataset["holiday"] = dataset["date"].apply(lambda x: x.strftime('%Y-%m-%d') in hk_2017_holidays or x.strftime('%Y-%m-%d') in hk_2018_holidays)*1
y = dataset['speed']

X = dataset.drop(["id", "date", "speed"], axis=1)
X_train, X_eval, y_train, y_eval = train_test_split(X, y, test_size=0.2, random_state=1)

# Initialize CatBoostRegressor
model = CatBoostRegressor(iterations=1100,
                          learning_rate=0.09,
                          depth=10,
                          use_best_model=True,
                          l2_leaf_reg=2)
eval_dataset = Pool(X_eval,
                    y_eval)
# Fit model
model.fit(X_train,
          y_train,
          eval_set=eval_dataset,
          verbose=False)
# Get predictions
print(model.get_best_score())

test_set = pd.read_csv('test.csv')
test_set["date"] = pd.to_datetime(test_set["date"], format="%d/%m/%Y %H:%M")
test_set["year"] = test_set["date"].apply(lambda x: x.year)
test_set["month"] = test_set["date"].apply(lambda x: x.month)
test_set["day"] = test_set["date"].apply(lambda x: x.day)
test_set["hour"] = test_set["date"].apply(lambda x: x.hour)
test_set["weekday"] = test_set["date"].apply(lambda x: x.isoweekday())
test_set['hour_of_week'] = (test_set["weekday"] * 24 - 24) + test_set["hour"]
test_set['hour_of_month'] = (test_set["month"] * 24 - 24) + test_set["hour"]
test_set["quarter"] = test_set["date"].apply(lambda x: x.quarter)
test_set["season"] = test_set["month"].apply(lambda x: (x%12 + 3)//3)
test_set["holiday"] = test_set["date"].apply(lambda x: x.strftime('%Y-%m-%d') in hk_2017_holidays or x.strftime('%Y-%m-%d') in hk_2018_holidays)*1
test_set = test_set.drop(["id", "date"], axis=1)

y_pred = model.predict(test_set)
pd.DataFrame(y_pred, columns=['speed']).reset_index().rename(columns={'index': 'id'}).to_csv("submission.csv", index=False)