# Traffic Level Classification

A compact machine-learning learning project that uses historical vehicle-count data to classify traffic into **Low**, **Medium**, or **High** levels.

> **Project status:** Completed learning experiment. The model is implemented in a single Python script and uses the included CSV dataset.

## What the project does

The script:

1. Loads traffic data from `traffic_data.csv`.
2. Removes missing values.
3. Extracts the hour from the time column.
4. Creates a peak-hour indicator.
5. Encodes the day of the week.
6. Calculates total vehicle count.
7. Creates Low/Medium/High traffic labels from defined count thresholds.
8. Builds a weighted-traffic feature.
9. Splits the data into training and test sets.
10. Trains a `RandomForestClassifier`.
11. Reports balanced accuracy, a confusion matrix, and a classification report.
12. Runs one example prediction and displays a traffic visualization.

## Model

The current implementation uses:

```text
RandomForestClassifier
n_estimators = 1000
class_weight = "balanced"
random_state = 42
```

The train/test split uses 70% of the data for training and 30% for testing with stratification.

## Features used

- Hour
- Peak-hour indicator
- Day of the week
- Bike count
- Car count
- Bus count
- Truck count
- Weighted traffic

The weighted feature gives different relative weights to vehicle types before classification.

## Tech stack

- Python
- pandas
- NumPy
- scikit-learn
- Matplotlib

## Repository contents

```text
Traffic-Prediction-Model/
├── Traffic_Prediction_Model.py
├── traffic_data.csv
└── README.md
```

## Run locally

Install the required libraries:

```bash
pip install pandas numpy scikit-learn matplotlib
```

Run the script from the repository directory:

```bash
python Traffic_Prediction_Model.py
```

The script prints the evaluation metrics in the terminal and opens the visualization using Matplotlib.

## Evaluation note

The traffic classes in this project are **created inside the script from total vehicle-count thresholds**:

- Low: fewer than 30 total vehicles
- Medium: 30–69 total vehicles
- High: 70 or more total vehicles

Because the target labels are derived from the same underlying vehicle counts used as model inputs, this should be viewed as a machine-learning practice project rather than evidence of a real-world congestion forecasting system. A stronger future version would use independently observed congestion labels and time-based validation.

## Possible next steps

- Compare multiple classifiers
- Use independently labelled traffic conditions
- Add cross-validation
- Separate training and inference code
- Save the trained model
- Build a small prediction interface

## Author

**Anfal Qureshi**  
Computer Science student exploring machine learning, data preparation, evaluation, and practical modelling workflows.
