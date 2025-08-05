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
from termcolor import colored
data = pl.read_csv("/Users/jacksaigusa/Downloads/UFCPredictor2025/Data/elofightstats5122025.csv")
data_test = pl.read_csv("/Users/jacksaigusa/Downloads/UFCPredictor2025/Data/dolidze_vs_hernandez.csv")

reference_date = pl.date(1970, 1, 1)
data = data.with_columns(
    pl.col("date").str.to_datetime("%b. %d, %Y").alias("date")
)

data = data.with_columns(
    (pl.col("date")-reference_date).dt.total_days().alias("date")
)
#creating weight class column
data = data.with_columns(
    pl.when(pl.col("fighter_weight") != "--").then(pl.col("fighter_weight").str.slice(0, 3).cast(pl.Int32, strict=False)).otherwise(None).alias("fighter_weight")
)
data = data.with_columns(
    pl.when(pl.col("opponent_weight") != "--").then(pl.col("opponent_weight").str.slice(0, 3).cast(pl.Int32, strict=False)).otherwise(None).alias("opponent_weight")
)

data_test = data_test.with_columns(
    pl.col("fighter_weight").cast(pl.Int32).alias("fighter_weight")
)

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

data_test = data_test.with_columns(
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
        "opponent_titlewins",
        "opponent_elo",
        "opponent_opp_avg_elo"
    ]
)
data_test = data_test.drop_nans(
    subset=["fighter_kd_differential",
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
        "opponent_opp_avg_elo"]
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

data_test = data_test.drop_nulls(
    subset=["date",
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
    "opponent_opp_avg_elo"]
)

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

data_test = data_test.with_columns(
    pl.col("fighter_dob").str.to_datetime("%b %d, %Y", strict=False))

data_test = data_test.with_columns(
    (pl.col("fighter_dob")-reference_date).dt.total_days().alias("fighter_age")
)

data_test = data_test.with_columns(
    pl.col("opponent_dob").str.to_datetime("%b %d, %Y", strict=False))

data_test = data_test.with_columns(
    (pl.col("opponent_dob")-reference_date).dt.total_days().alias("opponent_age")
)




# iterate over rows of y_test, determining which model to use to make prediction at each iteration by checking the weight class


data = data[selected_columns]
data_test = data_test[selected_columns]
# save the data for later use
#data.write_csv("/Users/jacksaigusa/Downloads/UFCPredictor2025/Data/elo_training_data.csv")

# create seperate weight class datasets

enc = LabelEncoder()
data = data.with_columns(
    pl.col("result").map_batches(enc.fit_transform).alias("result")
)
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

# Train each weight class-specific model

# lightest weight model
X = lightest.drop("result")
y = lightest["result"]

lightest_model = RandomForestClassifier(
    random_state=42, n_estimators=200, min_samples_split=10, min_samples_leaf=4, max_features='sqrt', max_depth=None, bootstrap=True)
lightest_model.fit(X, y)

# light weight model 

X = light.drop("result")
y = light["result"]

light_model = RandomForestClassifier(random_state=42, n_estimators=200, min_samples_split=10, min_samples_leaf=4, max_features='sqrt', max_depth=None, bootstrap=True)
light_model.fit(X, y)


# mid model 
X = mid.drop("result")
y = mid["result"]

mid_model = RandomForestClassifier(random_state=42)
mid_model.fit(X, y)

# heavy model

X = heavy.drop("result")
y = heavy["result"]

heavy_model = RandomForestClassifier(random_state=42)

heavy_model.fit(X, y)


# make predictions with correct model based on weight class 

data_test = data_test.drop("result")
y_pred = []
y_pred_probs = []
for i in range(len(data_test)):
    
    test_sample = data_test.slice(i, 1)
    weight_class = test_sample["weight_class"][0]
    
    if weight_class == 0: #lightest
        y_pred.append(lightest_model.predict(test_sample)[0])
        y_pred_probs.append(lightest_model.predict_proba(test_sample)[0])
    elif weight_class == 1: #light
        y_pred.append(light_model.predict(test_sample)[0])
        y_pred_probs.append(light_model.predict_proba(test_sample)[0])
    elif weight_class == 2: #mid
        y_pred.append(mid_model.predict(test_sample)[0])
        y_pred_probs.append(mid_model.predict_proba(test_sample)[0])
    else: #heavy or no value 
        y_pred.append(heavy_model.predict(test_sample)[0])
        y_pred_probs.append(heavy_model.predict_proba(test_sample)[0])



new_predictions = y_pred
#create probabilities array 
probabilities = [max(probs) for probs in y_pred_probs]

new_predictions = enc.inverse_transform(new_predictions)
new_fights_with_names = data_test = pl.read_csv("/Users/jacksaigusa/Downloads/UFCPredictor2025/Data/dolidze_vs_hernandez.csv")
i = 0
seen = set()
fights_with_predictions = []



event_name = "UFC Fight Night: Dolidze vs. Hernandez" ## IMPORTANT: CHANGE EVENT NAME BEFORE SAVING ##

import time
## IMPORTANT: Change the date to correct date if storing new predictions in db ##
fight_date = datetime(2025, 6, 7)
print(colored(f"\n\n\tPREDICTIONS FOR {event_name}\n", 'yellow', 'on_black', ['bold', 'blink']))
for row in new_fights_with_names.iter_rows():
    fighter_name = row[3]
    opponent_name = row[33]
    if fighter_name in seen or opponent_name in seen:
        i += 1
        continue
    if new_predictions[i] == "loss":
        print(f"   {fighter_name} vs. {opponent_name}\n   " + colored(f"   Prediction: {opponent_name}", "green") + colored(f"\n      Confidence: {round(probabilities[i], 2)}", "cyan"))
    elif new_predictions[i] == "draw":
        print("   The fight will result in a draw." + colored(f"\n   Confidence: {round(probabilities[i], 2)}", "cyan"))
    else:
        print(f"   {fighter_name} vs. {opponent_name}\n   " + colored(f"   Prediction: {fighter_name}", "green") + colored(f"\n      Confidence: {round(probabilities[i], 2)}", "cyan"))
    fights_with_predictions.append({"event": event_name, "date": fight_date, "fighter_name":fighter_name, "opponent_name":opponent_name, "prediction": str(new_predictions[i]), "confidence": float(round(probabilities[i], 2))})
    seen.add(fighter_name)
    seen.add(opponent_name)
    i += 1
    print('\n')
    time.sleep(0.5)
    
print(f"last index: {i}")
print(f"length: {len(new_predictions)}\n\n")


# save predictions to update after event

prediction_file = pl.DataFrame(fights_with_predictions)

prediction_file = prediction_file.with_columns(
    pl.lit(event_name).alias("event"), 
    pl.lit(True).alias("correct")
)


# Make sure to change output file name for new predictions
prediction_file.write_csv("/Users/jacksaigusa/Downloads/UFCPredictor2025/Data/Predictions/dolidze_vs_hernandez_complete.csv")
print("predictions saved to csv!")

#save predictions to sqlalchemy database 
from flask import Flask
from flask_sqlalchemy import SQLAlchemy

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///predictions.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db3 = SQLAlchemy(app)

class Prediction(db3.Model):
    id = db3.Column(db3.Integer, primary_key = True, index = True)
    Event = db3.Column(db3.String(100))
    Date = db3.Column(db3.DateTime)
    FighterName = db3.Column(db3.String(100))
    OpponentName = db3.Column(db3.String(100))
    Prediction = db3.Column(db3.String(20))
    Confidence = db3.Column(db3.Float)


with app.app_context():
    db3.drop_all() #<- uncomment for clearing and creating fresh database
    db3.create_all()
    


with app.app_context():
    try:
        for pred in fights_with_predictions:
            new_prediction = Prediction(
                Date = pred['date'],
                Event = pred['event'],
                FighterName = pred['fighter_name'],
                OpponentName = pred['opponent_name'],
                Prediction = pred['prediction'],
                Confidence = pred['confidence']
            )
            db3.session.add(new_prediction)
        db3.session.commit()
    except Exception as e:
        db3.session.rollback()
        print(colored(f"Error storing predictions: {e}", "red"))
    finally:
        db3.session.close()

# run command after successful test db update 
#cp instance/{predictions.db,past_predictions.db} WebApp/backend/instance/

















