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
) #205+ group contains the smallest number of rows, so fights with unknown weight classes were added all to heavy



data = data.drop_nans()
data = data.drop_nulls()

X = data.drop(["result", "weight_class"])
y = data["result", "weight_class"]


# making pca 
from sklearn.preprocessing import StandardScaler

X = StandardScaler().fit_transform(X)
from sklearn.decomposition import PCA

pca = PCA(n_components=2)

principal_components = pl.DataFrame(pca.fit_transform(X))
print(principal_components.head())

principleDf = principal_components.rename({"column_0": "PC1",
                                           "column_1": "PC2"})

# Define markers for different weight classes
weight_class_markers = {
    0: 'o',   # lightest: circle
    1: '^',   # light: triangle
    2: 's',   # mid: square
    3: '*',   # heavy: star
    -1: 'x'   # unknown: x
}


y = pl.DataFrame(y)
finalDf = pl.concat([principleDf, y], how="horizontal")

print(finalDf.head())

targets = ["win", "loss", "draw"]
colors = ["g", "r", "b"]

plt.figure(figsize=(12, 10))

# First, iterate through result categories for colors
for target, color in zip(targets, colors):
    target_data = finalDf.filter(pl.col("result") == target)
    
    # Then, iterate through weight classes for shapes
    for weight_class, marker in weight_class_markers.items():
        indices = target_data.filter(pl.col("weight_class") == weight_class).select(pl.all()).to_pandas()
        
        if len(indices) > 0:  # Only plot if there are points for this combination
            # For the label, only include it once per unique combination to avoid duplicate legend entries
            if target_data.filter(pl.col("weight_class") == weight_class).shape[0] > 0:
                label = f"{target} (weight class: {weight_class})"
            else:
                label = None
                
            plt.scatter(indices["PC1"], indices["PC2"], 
                        c=color, marker=marker, s=70, 
                        label=label, alpha=0.7, edgecolors='black', linewidths=0.5)

# Customize plot
plt.xlabel('Principal Component 1', fontsize=14)
plt.ylabel('Principal Component 2', fontsize=14)
plt.title('PCA of UFC Fight Data by Result and Weight Class', fontsize=16)

# Create a custom legend for better readability
from matplotlib.lines import Line2D

# Create legend elements for outcomes (colors)
outcome_legend_elements = [Line2D([0], [0], marker='o', color='w', 
                                 markerfacecolor=color, markersize=10, 
                                 label=target) 
                          for target, color in zip(targets, colors)]

# Create legend elements for weight classes (shapes)
weight_class_names = {
    0: "Lightest (115-135)",
    1: "Light (145-155)",
    2: "Mid (170-185)",
    3: "Heavy (205+)",
    -1: "Unknown"
}

weight_class_legend_elements = [Line2D([0], [0], marker=marker, color='black', 
                                      markersize=10, linestyle='None',
                                      label=weight_class_names[weight_class]) 
                               for weight_class, marker in weight_class_markers.items()]

# Add two separate legends
first_legend = plt.legend(handles=outcome_legend_elements, title="Outcome", loc='upper left')
plt.gca().add_artist(first_legend)  

plt.legend(handles=weight_class_legend_elements, title="Weight Class", loc='upper right')

plt.grid(True, alpha=0.3)

# Add variance explained as text
explained_variance_ratio = pca.explained_variance_ratio_
plt.figtext(0.02, 0.02, f'Explained variance: PC1={explained_variance_ratio[0]:.2f}, PC2={explained_variance_ratio[1]:.2f}', 
           fontsize=10)

plt.tight_layout()
#plt.savefig('ufc_fights_pca_with_weights.png', dpi=300)
plt.show()


# add pc1 as column to dataset and create train_test_split
#train model and assess if PC1 is a useful feature


X = data.drop("result")
# add PC1 as a feature 
# importance of PC1 is not in the top 10; about 0.03
# accuracy of model with PC1 as an additional feature: 0.68
# normal accuracy without PC1: 0.68

X = pl.concat([X, pl.DataFrame({"PC1":principleDf["PC1"]})], how="horizontal")
y = data["result"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


model = RandomForestClassifier(n_estimators=300, min_samples_split=5, min_samples_leaf=1, max_features='log2', max_depth=10, bootstrap=True)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print(f"Accuracy of Model Using PC1 as an Additional Featrure: {round(accuracy, 2)}\n\n")


importances = pl.DataFrame(
    {"Feature": X.columns, "Importance": model.feature_importances_}
)
importances = importances.sort("Importance", descending=True)
print(f"\tTop 10 Features of Model with PC1 as an Additional Feature:\n {importances.head(10)}")


