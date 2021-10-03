#Importing Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys

#Function for outlier removal
def outlier_removal(input_file):
    dataset=pd.read_csv(input_file)
    data=dataset.iloc[:,1:]
    
    for n,row in data.iterrows():
        #Defining threshold value
        threshold_value=2.5
        mean=np.mean(row)
        standard_deviation=np.std(row)
        
        for value in row:
            #Calculating z score
            z_score=(value-mean)/standard_deviation
            
            #Removing rows whose z_score> threshold value
            if np.abs(z_score)>threshold_value:
                dataset = dataset.drop(data.index[n])
    
          
    rows_removed=len(data) -len(dataset)
    return rows_removed

# evaluate model on the raw dataset
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
# load the dataset
url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/housing.csv'
df = read_csv(url, header=None)
# retrieve the array
data = df.values
# split into input and output elements
X, y = data[:, :-1], data[:, -1]
# split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)
# fit the model
model = LinearRegression()
model.fit(X_train, y_train)
# evaluate the model
yhat = model.predict(X_test)
# evaluate predictions
mae = mean_absolute_error(y_test, yhat)
print('MAE: %.3f' % mae)

def main():
    import sys
    total = len(sys.argv)
    if (total!=2):
        print("ERROR! WRONG NUMBER OF PARAMETERS")
        print("USAGES: $python <programName> <dataset>")
        print('EXAMPLE: $python programName.py data.csv')
        sys.exit(1)
  #  dataset=pd.read_csv(sys.argv[1]).values
    rr=outlier_removal(sys.argv[1])
    print("Number of rows removed are: ",rr)

if __name__=="__main__":
     main()
