import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from logging import getLogger, Formatter, StreamHandler, INFO
import dtreeviz
import json

# Setup logging
logger = getLogger(__name__)
logFormatter = Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger.setLevel(INFO)
consoleHandler = StreamHandler()
consoleHandler.setFormatter(logFormatter)
logger.addHandler(consoleHandler)


def generate_tree(
    X: pd.DataFrame,
    y: pd.Series,
    target_column: str,
    threshold_value: int,
    architecture: str,
    output_folder: Path,
) -> None:
    """
    Generates a decision tree classifier, fits it to the provided data, evaluates its accuracy,
    visualizes the tree, and saves the visualization to the specified output folder.

    Parameters:
    X (pd.DataFrame): The input features for training the decision tree.
    y (pd.Series): The target labels for training the decision tree.
    target_column (str): The name of the target column.
    threshold_value (int): The threshold value to create a binary target column.
    architecture (str): The architecture name for logging purposes.
    output_folder (Path): The directory where the decision tree visualization will be saved.

    Returns:
    None
    """
    logger.info(f"Generating Decision Tree for {target_column} and {architecture}")
    classifier = DecisionTreeClassifier(random_state=1, max_depth=4)
    classifier.fit(X.values, y.values)
    y_pred = classifier.predict(X.values)
    score = accuracy_score(y, y_pred)
    tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
    # Calculate the true positive rate and false positive rate
    tpr = tp / (tp + fn)
    fpr = fp / (fp + tn)
    logger.info(f"Accuracy score: {score}")
    logger.info(f"True Positive Rate: {tpr}")
    logger.info(f"False Positive Rate: {fpr}")

    # Create a dtreeviz object
    viz_model = dtreeviz.model(
        classifier,
        X_train=X,
        y_train=y,
        feature_names=X.columns,
        target_name="Risk",
        class_names=["Low Risk", "High Risk"],
    )
    v = viz_model.view(
        title=f"Decision Tree for {architecture}\n"
        f"High Risk is {threshold_value} or less correct predictions\n"
        f"Low Risk is {threshold_value+1} or more correct predictions\n"
        f"Accuracy: {score:.2f} | TPR: {tpr:.2f} | FPR: {fpr:.2f}",
    )
    # Save the tree
    save_path = output_folder / f"tree_{architecture}_n_{threshold_value}.svg"
    v.save(str(save_path))
    logger.info(f"Decision Tree saved at {save_path}")


def prepare_features(
    df: pd.DataFrame, categorical_columns: list, numerical_columns: list
) -> pd.DataFrame:
    """
    Prepares a DataFrame for decision tree by encoding categorical features and
    selecting numerical features.

    Args:
        df (pd.DataFrame): The input DataFrame containing the data.
        categorical_columns (list): A list of column names in the DataFrame that are categorical.
        numerical_columns (list): A list of column names in the DataFrame that are numerical.

    Returns:
        pd.DataFrame: A DataFrame with numerical features and one-hot encoded categorical features.
    """
    X = df[numerical_columns]
    for column in categorical_columns:
        X = pd.concat([X, pd.get_dummies(df[column], prefix=column)], axis=1)

    return X


# Read JSON config
with open("config.json") as f:
    config = json.load(f)

# Read data
data_path = Path(config["data_path"])
categorical_columns = config["categorical_columns"]
numerical_columns = config["numerical_columns"]
output_folder = Path(config["output_folder"])
target_column = config["target_column"]
thrshold_values = config["total_correct_threshold_values"]

architectures = config["architectures"]
for architecture in architectures:
    csv_path = data_path / architectures[architecture]
    logger.info(f"Reading data from {csv_path}")
    df = pd.read_csv(csv_path)
    # Create a subset of the dataframe with only the columns of interest
    df = df[categorical_columns + numerical_columns + [target_column]]
    logger.info(f"Total number of rows: {df.shape[0]}")
    logger.info(
        f"Number of missing values per column:\n{df[categorical_columns+numerical_columns].isnull().sum()}"
    )
    # Drop rows with any missing values
    df.dropna(inplace=True)
    X = prepare_features(df, categorical_columns, numerical_columns)
    logger.info(f"Number of rows for fitting the tree: {X.shape[0]}")
    # Display distribution of target values
    logger.info(
        f"Distribution of {target_column}:\n{df[target_column].value_counts(normalize=True).sort_index()}"
    )
    for threshold_value in thrshold_values:
        # Create a binary target column based on the threshold value
        logger.info(f"Threshold value: {threshold_value}")
        y = df[target_column] <= threshold_value
        class_proportions = y.value_counts(normalize=True).reindex([True, False])
        logger.info(f"Class proportions\n{class_proportions}")
        generate_tree(X, y, target_column, threshold_value, architecture, output_folder)
