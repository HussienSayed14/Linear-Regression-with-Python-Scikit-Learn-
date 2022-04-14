import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics



url = "http://bit.ly/w-data"
S_Data = pd.read_csv(url)
print("Data imported succeessfully")
S_Data.head(10)
S_Data.plot(x='Hours', y='Scores', style='o')
plt.title('Hours vs Percentage')
plt.xlabel('Hours Studied')
plt.ylabel('Percentage Score')
#plt.show()
X = S_Data.iloc[:, :-1].values
y = S_Data.iloc[:, 1].values

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=0)
regressor = LinearRegression()
regressor.fit(X_train, y_train)
print("Training complete")
line = regressor.coef_*X+regressor.intercept_

plt.scatter(X,y)
plt.plot(X, line);
plt.show()
print(X_test)
y_pred = regressor.predict(X_test)
df = pd.DataFrame({'Actual': y_test, 'Predicted':y_pred})
print(df)
print('Mean Absolute Error:',metrics.mean_absolute_error(y_test, y_pred))






