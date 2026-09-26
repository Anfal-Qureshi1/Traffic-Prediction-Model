
# 1. Import Libraries

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    balanced_accuracy_score
)

import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")


# 2. Load Dataset

df = pd.read_csv("traffic_data.csv")

print("\n FIRST 5 ROWS:\n")
print(df.head())


# 3. Check Missing Values

print("\nMissing values:\n", df.isnull().sum())

df = df.dropna()


# 4. Time Feature Engineering

df['Hour'] = pd.to_datetime(df['Time'], errors='coerce').dt.hour
df = df.dropna(subset=['Hour'])

df['Is_Peak_Hour'] = (
    df['Hour'].between(14, 16) |
    df['Hour'].between(17, 19)
).astype(int)

# Encode day
day_le = LabelEncoder()
df['Day_Label'] = day_le.fit_transform(df['Day of the week'])


# 5. CREATE LABEL 

df['Total_Vehicles'] = (
    df['BikeCount'] +
    df['CarCount'] +
    df['BusCount'] +
    df['TruckCount']
)


def traffic_label(total):
    if total < 30:
        return "Low"
    elif total < 70:
        return "Medium"
    else:
        return "High"

df['Traffic_Situation'] = df['Total_Vehicles'].apply(traffic_label)

# Encode target
traffic_le = LabelEncoder()
df['Traffic_Situation_Label'] = traffic_le.fit_transform(df['Traffic_Situation'])


# 6. FEATURE: WEIGHTED TRAFFIC

df['Weighted_Traffic'] = (
    df['BikeCount'] * 0.5 +
    df['CarCount'] * 1.0 +
    df['BusCount'] * 2.5 +
    df['TruckCount'] * 3.0
)

# 7. Feature Selection

X = df[
    [
        'Hour',
        'Is_Peak_Hour',
        'Day_Label',
        'BikeCount',
        'CarCount',
        'BusCount',
        'TruckCount',
        'Weighted_Traffic'
    ]
]

y = df['Traffic_Situation_Label']


# 8. Train-Test Split

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    random_state=50,
    stratify=y
)


# 9. Train RANDOM FOREST (MODEL CHANGE ONLY)

rf_model = RandomForestClassifier(
    n_estimators=1000,
    class_weight='balanced',
    random_state=42
)

rf_model.fit(X_train, y_train)


# 10. Evaluation

y_pred = rf_model.predict(X_test)


print("Accuracy:", balanced_accuracy_score(y_test, y_pred))

labels = sorted(y.unique())

print("\n Confusion Matrix:")
print(confusion_matrix(y_test, y_pred, labels=labels))

print("\n Classification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        labels=labels,
        target_names=traffic_le.inverse_transform(labels)
    )
)


# 11. New Prediction Example

bike, car, bus, truck = 2, 13, 2, 24
weighted = bike*0.5 + car*1 + bus*2.5 + truck*3

new_data = [[
    0, 0, 1,
    bike, car, bus, truck,
    weighted
]]

pred = rf_model.predict(new_data)[0]

print("\n🚦 Predicted Traffic Situation:",
      traffic_le.inverse_transform([pred])[0])


# 12. Visualization

plt.bar(df['Hour'], df['Weighted_Traffic'], alpha=0.5)
plt.xlabel("Hour of Day")
plt.ylabel("Average Weighted Traffic")
plt.title("Average Weighted Traffic per Hour")
plt.show()
