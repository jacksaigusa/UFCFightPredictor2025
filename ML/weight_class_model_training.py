import polars as pl
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.datasets import make_multilabel_classification
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.model_selection import RandomizedSearchCV

data = pl.read_csv("/Users/jacksaigusa/Downloads/UFCPredictor2025/Data/elofightstats5122025.csv")

#removing old fights to see if performance improves: it doesnt, slightly worse

reference_date = pl.date(1970, 1, 1)
data = data.with_columns(
    pl.col("date").str.to_datetime("%b. %d, %Y").alias("date")
)

data = data.with_columns(
    (pl.col("date")-reference_date).dt.total_days().alias("date")
)
#print(f"dataset size before filtering out old fights: {data.shape}")

#data = data.filter(pl.col("date") >= 14600)

#print(f"dataset size after filtering out old fights: {data.shape}")
#removing missing weight values
'''data = data.filter(
    pl.col("fighter_weight") != "--", 
    pl.col("opponent_weight") != "--"
)'''

data = data.with_columns(
    pl.when(pl.col("fighter_weight") != "--").then(pl.col("fighter_weight").str.slice(0, 3).cast(pl.Int32, strict=False)).otherwise(None).alias("fighter_weight")
)
data = data.with_columns(
    pl.when(pl.col("opponent_weight") != "--").then(pl.col("opponent_weight").str.slice(0, 3).cast(pl.Int32, strict=False)).otherwise(None).alias("opponent_weight")
)


data = data.with_columns(
    pl.col("fighter_weight").fill_null(-1)
)

data = data.with_columns(
    pl.col("opponent_weight").fill_null(-1)
)

print(data["fighter_weight"].head())
# Creating new column for weight class. 
#   115-135: lightest, encoded as 0
#   145-155: light, encoded as 1
#   170-185: mid, encoded as 2
#   205-265+: heavy, encoded as 3
data = data.with_columns(
    pl.when(
        (pl.col("fighter_weight") != -1) & (pl.col("fighter_weight") >= 115) & (pl.col("fighter_weight") <= 136)
    ).then(0)  # lightest
    .when(
        (pl.col("fighter_weight") != -1) & (pl.col("fighter_weight") > 136) & (pl.col("fighter_weight") <= 156)
    ).then(1)  # light
    .when(
        (pl.col("fighter_weight") != -1) & (pl.col("fighter_weight") > 156) & (pl.col("fighter_weight") <= 186)
    ).then(2)  # mid
    .when(
        (pl.col("fighter_weight") != -1) & (pl.col("fighter_weight") > 186) & (pl.col("fighter_weight") <= 266)
    ).then(3)  # heavy
    .otherwise(-1)
    .alias("weight_class")
)
print(data["weight_class"].head())



'''print(f"average fighter weight: {data["fighter_weight"].mean()}")
print(f"average opponent weight: {data["opponent_weight"].mean()}")'''

data = data.drop_nans(subset=[
        "fighter_kd_differential",
        "fighter_str_differential",
        "fighter_td_differential",
        "fighter_sub_differential",
        "fighter_winstreak",
        "fighter_losestreak",
        "fighter_age_deviation",
        "fighter_titlefights",
        "fighter_titlewins",
        "fighter_elo",
        "fighter_opp_avg_elo",
        "opponent_kd_differential",
        "opponent_str_differential",
        "opponent_td_differential",
        "opponent_sub_differential",
        "opponent_winstreak",
        "opponent_losestreak",
        "opponent_age_deviation",
        "opponent_titlefights",
        "opponent_titlewins" 
    ]
)
data = data.drop_nulls(subset=[
    "date",
    "fighter_kd_differential",
    "fighter_str_differential",
    "fighter_td_differential",
    "fighter_sub_differential",
    "fighter_winstreak",
    "fighter_losestreak",
    "fighter_age_deviation",
    "fighter_titlefights",
    "fighter_titlewins",
    "fighter_elo",
    "fighter_opp_avg_elo",
    "opponent_kd_differential",
    "opponent_str_differential",
    "opponent_td_differential",
    "opponent_sub_differential",
    "opponent_winstreak",
    "opponent_losestreak",
    "opponent_age_deviation",
    "opponent_titlefights",
    "opponent_titlewins",
    "opponent_elo",
    "opponent_opp_avg_elo"
])

selected_columns = [
    "fighter_kd_differential",
    "fighter_str_differential",
    "fighter_td_differential",
    "fighter_sub_differential",
    "fighter_winstreak",
    "fighter_losestreak",
    "fighter_age_deviation",
    "fighter_titlefights",
    "fighter_titlewins",
    "fighter_elo",
    "fighter_opp_avg_elo",
    "opponent_kd_differential",
    "opponent_str_differential",
    "opponent_td_differential",
    "opponent_sub_differential",
    "opponent_winstreak",
    "opponent_losestreak",
    "opponent_age_deviation",
    "opponent_titlefights",
    "opponent_titlewins",
    "opponent_elo",
    "opponent_opp_avg_elo",
    "result",
    "fighter_age",
    "opponent_age", 
    "weight_class"
]







#fixing dates

reference_date = pl.date(1970, 1, 1)
data = data.with_columns(
    pl.col("fighter_dob").str.to_datetime("%b %d, %Y", strict=False))

data = data.with_columns(
    (pl.col("fighter_dob")-reference_date).dt.total_days().alias("fighter_age")
)


data = data.with_columns(
    pl.col("opponent_dob").str.to_datetime("%b %d, %Y", strict=False))

data = data.with_columns(
    (pl.col("opponent_dob")-reference_date).dt.total_days().alias("opponent_age")
)


data = data[selected_columns]

# save the data for later use
#data.write_csv("/Users/jacksaigusa/Downloads/UFCPredictor2025/Data/elo_training_data.csv")
enc = LabelEncoder()
data = data.with_columns(
    pl.col("result").map_batches(enc.fit_transform).alias("result")
)
#encode other variables if you want to predict method and round as well

lightest = data.filter(
    pl.col("weight_class") == 0
)
light = data.filter(
    pl.col("weight_class") == 1
)
mid = data.filter(
    pl.col("weight_class") == 2
)
heavy = data.filter(
    (pl.col("weight_class") == 3) | (pl.col("weight_class") == -1)
) #205+ group contains the smallest number of rows, so I added all rows with unknown weight classes to heavy

weight_classes = [lightest, light, mid, heavy]
for weight_class in weight_classes:
    print(weight_class.shape)



param_dist = {
    'n_estimators': [100, 200, 300, 500],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth': [10, 20, 30, 40, 50, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}

# lightest weight model
X = lightest.drop("result")
y = lightest["result"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



lightest_model = RandomForestClassifier(
    random_state=42, n_estimators=200, min_samples_split=10, min_samples_leaf=4, max_features='sqrt', max_depth=None, bootstrap=True)
lightest_model.fit(X_train, y_train)





y_pred = lightest_model.predict(X_test)
lightest_accuracy = accuracy_score(y_test, y_pred)

lightest_test_size = len(y_pred)
print("LIGHTEST MODEL: 115-135LBS")
print(f"training set size: {pl.DataFrame(X_train).shape}")
print(f"testing set size: {pl.DataFrame(X_test).shape}")
print(f"Accuracy: {round(lightest_accuracy, 2)}\n")







# light weight model
X = light.drop("result")
y = light["result"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



light_model = RandomForestClassifier(random_state=42, n_estimators=200, min_samples_split=10, min_samples_leaf=4, max_features='sqrt', max_depth=None, bootstrap=True)
light_model.fit(X_train, y_train)

y_pred = light_model.predict(X_test)
light_accuracy = accuracy_score(y_test, y_pred)

light_test_size = len(y_pred)
print("LIGHT MODEL: 145-155LBS")




print(f"training set size: {pl.DataFrame(X_train).shape}")
print(f"testing set size: {pl.DataFrame(X_test).shape}")
print(f"Accuracy: {round(light_accuracy, 2)}\n")




# mid model 
X = mid.drop("result")
y = mid["result"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



mid_model = RandomForestClassifier(random_state=42)
mid_model.fit(X_train, y_train)

y_pred = mid_model.predict(X_test)
mid_accuracy = accuracy_score(y_test, y_pred)
mid_test_size = len(y_pred)
print("MID MODEL: 170-185LBS")
print(f"training set size: {pl.DataFrame(X_train).shape}")
print(f"testing set size: {pl.DataFrame(X_test).shape}")
print(f"Accuracy: {round(mid_accuracy, 2)}\n")




# heavy model 



X = heavy.drop("result")
y = heavy["result"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



heavy_model = RandomForestClassifier(random_state=42)

heavy_model.fit(X_train, y_train)


y_pred = heavy_model.predict(X_test)

heavy_accuracy = accuracy_score(y_test, y_pred)

heavy_test_size = len(y_pred)
print("HEAVY MODEL: 205+LBS")
print(f"training set size: {pl.DataFrame(X_train).shape}")
print(f"testing set size: {pl.DataFrame(X_test).shape}")
print(f"Accuracy: {round(heavy_accuracy, 2)}\n")



# For each model, examine feature importance
for name, model in [("Lightest", lightest_model), ("Light", light_model), 
                    ("Mid", mid_model), ("Heavy", heavy_model)]:
    importances = model.feature_importances_
    feature_names = X.columns
    
    # Sort features by importance
    indices = np.argsort(importances)[::-1]
    
    print(f"\n{name} Weight Class - Top 10 Features:")
    for i in range(10):
        print(f"{feature_names[indices[i]]}: {importances[indices[i]]:.4f}")
all_weight_classes = [lightest_accuracy, light_accuracy, mid_accuracy, heavy_accuracy]
average_accuracy = np.mean(all_weight_classes)
print(f"\n\nMEAN ACCURACY OF ALL WEIGHT CLASS-SPECIFIC MODELS: {average_accuracy}")
num_right = 0
test_sizes = [lightest_test_size, light_test_size, mid_test_size, heavy_test_size]
for i in range(len(all_weight_classes)):
    num_right += (all_weight_classes[i] * test_sizes[i])
print(f"\n\nNUMBER OF CORRECT PREDICTIONS COMBINED FOR ALL WEIGHT CLASS-SPECIFIC MODELS: {num_right}")







