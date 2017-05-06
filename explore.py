# Import libraries necessary for this project
import numpy as np
import pandas as pd
from IPython.display import display # Allows the use of display() for DataFrames

# Import supplementary visualizations code visuals.py
import visuals as vs

# Pretty display for notebooks
%matplotlib inline

# Load the dataset
in_file = 'titanic_data.csv'
full_data = pd.read_csv(in_file)

# Print the first few entries of the RMS Titanic data
display(full_data.head())

# Store the 'Survived' feature in a new variable and remove it from the dataset
outcomes = full_data['Survived']
data = full_data.drop('Survived', axis = 1)

# Show the new dataset with 'Survived' removed
display(data.head())

def accuracy_score(truth, pred):
    """ Returns accuracy score for input truth and predictions. """
    
    # Ensure that the number of predictions matches number of outcomes
    if len(truth) == len(pred): 
        
        # Calculate and return the accuracy as a percent
        return "Predictions have an accuracy of {:.2f}%.".format((truth == pred).mean()*100)
    
    else:
        return "Number of predictions does not match number of outcomes!"
    
# Test the 'accuracy_score' function
predictions = pd.Series(np.ones(5, dtype = int))
print(predictions)
print (accuracy_score(outcomes[:5], predictions))

# Accuracy assuming none of the passengers survived
print (accuracy_score(outcomes, predictions)) # 61.62%.


# Explore by Sex
vs.survival_stats(data, outcomes, 'Sex')

def predictions_1(data):
    """ Model with one feature: 
            - Predict a passenger survived if they are female. """
    
    predictions = []
    for _, passenger in data.iterrows():
        
        # Remove the 'pass' statement below 
        # and write your prediction conditions here
        if (passenger['Sex'] == 'female'):
            predictions.append(1)
        else:
            predictions.append(0) 
            
    
    # Return our predictions
    return pd.Series(predictions)

# Make the predictions
predictions = predictions_1(data)

# Accuracy assuming all female passengers survived and the remaining passengers did not survive
print (accuracy_score(outcomes, predictions)) # 78.68%.


# Explore again by Age and Sex
vs.survival_stats(data, outcomes, 'Age', ["Sex == 'male'"])

def predictions_2(data):
    """ Model with two features: 
            - Predict a passenger survived if they are female.
            - Predict a passenger survived if they are male and younger than 10. """
    
    predictions = []
    for _, passenger in data.iterrows():
        
        # Remove the 'pass' statement below 
        # and write your prediction conditions here
        if (passenger['Sex'] == 'female' or ( passenger['Sex'] == 'male' and passenger['Age'] < 10) ):
            predictions.append(1)
        else:
            predictions.append(0) 
    
    # Return our predictions
    return pd.Series(predictions)

# Make the predictions
predictions = predictions_2(data)

# Accuracy assuming all female passengers and all male passengers younger than 10 survived
print (accuracy_score(outcomes, predictions)) # 79.35%.


#  Explore again by Pclass and Sex
vs.survival_stats(data, outcomes, 'Pclass') ## La clase social  3 tuvo mas muertos
vs.survival_stats(data, outcomes, 'Pclass', ["Sex == 'female'"])


def predictions_3(data):
    """ Model with multiple features. Makes a prediction with an accuracy of at least 80%. """
    
    predictions = []
    for _, passenger in data.iterrows():
        
        # Remove the 'pass' statement below 
        # and write your prediction conditions here
        if passenger["Sex"] == "female":
            if passenger["Pclass"] == 1 or passenger["Pclass"] == 2:
                predictions.append(1)
            else:
                if passenger["Embarked"] == "C" or passenger["Embarked"] == "Q":
                    predictions.append(1)
                else:
                    predictions.append(0)
        else:
            if passenger["Age"] < 10.0:
                predictions.append(1)
            else:
                predictions.append(0)
    
    # Return our predictions
    return pd.Series(predictions)

# Make the predictions
predictions = predictions_3(data)

# Accuracy assuming all female passengers with Pclass 1 or 2 survived
print (accuracy_score(outcomes, predictions)) # 81.82%.

