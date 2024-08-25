#%%
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
def plot_column_vs_response(df, column_name):
    """
    This function takes a DataFrame and a column name, then plots a scatter plot with the specified column on the X-axis
    and 'Response' on the Y-axis, correcting the original function to match the DataFrame's actual 'Response' column.
    
    Parameters:
    - df: pandas DataFrame containing the 'Response' column and the specified column.
    - column_name: String name of the column to be plotted on the X-axis.
    
    Returns:
    - None. Displays a scatter plot.
    """
    # Check if the DataFrame contains the 'Response' column and the specified column
    if 'Response' not in df.columns or column_name not in df.columns:
        print(f"DataFrame does not contain the 'Response' column or the specified column '{column_name}'.")
        return
    
    # Drop rows with NaN values in 'Response' or the specified column to avoid errors in plotting
    df_cleaned = df.dropna(subset=['Response', column_name])
    
    plt.figure(figsize=(10, 6))
    plt.scatter(df_cleaned[column_name], df_cleaned['Response'], alpha=0.5)
    plt.title(f'{column_name} vs. Response')
    plt.xlabel(column_name)
    plt.ylabel('Response')
    plt.grid(True)
    plt.show()

def filter_missing_values(df):
    """
    This function takes a DataFrame and returns a new DataFrame with rows containing NaN values removed.
    
    Parameters:
    - df: pandas DataFrame to be filtered.
    
    Returns:
    - A new DataFrame with rows containing any NaN values removed.
    """
    rows_before = df.shape[0]
    df_cleaned = df.dropna()
    rows_after = df_cleaned.shape[0]
    rows_dropped = rows_before - rows_after

    print(f"Number of rows dropped due to NaN values: {rows_dropped}")
    
    return df.dropna()

def prepare_and_train_model(df):
    """
    Encodes categorical variables, splits data into features and target, divides the dataset into training and testing sets,
    creates and trains a logistic regression model, and evaluates the model's accuracy.
    
    Parameters:
    - df: pandas DataFrame to be processed and used for model training.
    
    Returns:
    - The trained model and its accuracy on the test set.
    """
    
    # Encode categorical variables
    label_encoders = {}
    for column in df.select_dtypes(include=['object']).columns:
        if column != 'Response':  # Exclude the target column from encoding
            le = LabelEncoder()
            df[column] = le.fit_transform(df[column])
            label_encoders[column] = le

    # Split data into features and target
    X = df.drop('Response', axis=1)
    y = df['Response']

    # Split Data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create and Train Model
    model = LogisticRegression(max_iter=1000)  # Increase max_iter if convergence warning appears
    model.fit(X_train, y_train)

    # Evaluate Model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy}")
    
    return None

def plot_correlation_matrix(df):
    """
    Generates a heatmap for the correlation matrix of the DataFrame's numerical features.
    
    Parameters:
    - df: pandas DataFrame whose correlation matrix will be plotted.
    """
    # Calculate the correlation matrix
    corr = df.corr(method='kendall')

    # Generate a heatmap
    plt.figure(figsize=(20, 20))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', square=True, linewidths=.5)
    plt.title('Correlation Matrix of DataFrame Variables')
    plt.show()

#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("../data/raw/train.csv")

# Display DataFrame information

df.info(verbose=True)
# %%

# Step 1: Preprocess Data
# Handle missing values (optional based on your DataFrame's current state)
df = filter_missing_values(df)

prepare_and_train_model(df)

# %%
plot_correlation_matrix(df)
# %%
