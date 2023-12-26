Importing necessary libraries
   import pandas as pd
   import numpy as np
   from sklearn.model_selection import train_test_split
   from sklearn.linear_model import LinearRegression
   from sklearn import metrics
   import matplotlib.pyplot as plt
   
Reading the data
  print("Dataset Preview:")
  print(data.head())
  
Plotting the distribution of scores
  data.plot(x='Hours', y='Scores', style='o')
  plt.title('Hours vs Percentage')
  plt.xlabel('Hours Studied')
  plt.ylabel('Percentage Score')
  plt.show()

Preparing the data
  X = data.iloc[:, :-1].values
  y = data.iloc[:, 1].values

Splitting the data into training and testing sets
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

Training the model
  model = LinearRegression()
  model.fit(X_train, y_train)

Making predictions on the test set
  y_pred = model.predict(X_test)

Comparing actual vs predicted
  df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
  print("\nActual vs Predicted Scores:")
  print(df)

Predicting the percentage for 9.25 hours/day
  hours = 9.25
  predicted_score = model.predict([[hours]])
  print(f'\nPredicted Score for {hours} hours/day: {predicted_score[0]}')

Evaluating the model
  mae = metrics.mean_absolute_error(y_test, y_pred)
  print('\nModel Evaluation:')
  print('Mean Absolute Error:', mae)

Plotting the regression line
  line = model.coef_ * X + model.intercept_
  plt.scatter(X, y)
  plt.plot(X, line, color='red')
  plt.title('Regression Line')
  plt.xlabel('Hours Studied')
  plt.ylabel('Percentage Score')
  plt.show()
