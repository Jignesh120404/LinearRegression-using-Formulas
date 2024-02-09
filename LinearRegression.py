from ast import arg
import pandas as pd #For Data Handling
import numpy as np
df = pd.read_csv("boston_house_prices.csv")
#print(df.shape)#(rows,Columns)    14 columns so 1 is target so we have 13 features and 502 datapoints. MEDV--Target Data  (ex:- 24 == $24,000)
print(df.head())
def linear_regression_GD(features,target,n_iter=20,learning_rate =0.001):
    B = np.zeros(features.shape[1])#Initialize the BetaValues
    cost =[]
    for i in range(n_iter):
        out_ = np.dot(features,B)#This is XB
        Error_ = out_ - target #This is (XB-Y)
        del_B =np.dot(features.T,Error_)
        B = B - learning_rate*del_B #(Gradient Desecnt formula for updation)
        cost_val =(Error_**2).sum()/2.0 #Error Value 
        cost.append(cost_val)
    return B,cost
X =df[['RM']].values #Input
Y =df[['MEDV']].values#Target Values
from sklearn.preprocessing import StandardScaler #PreProcessing of Data
sc_x = StandardScaler()
sc_y = StandardScaler()
Y_std =sc_x.fit_transform(Y)
X_std= sc_x.fit_transform(X)
Y_std = Y_std.flatten()
n_itr = 20    
B_opt , cost_ = linear_regression_GD(X_std,Y_std)
import matplotlib.pyplot as plt #Plotting the  Graph for Error vs No.of Iterations
#plt.plot(range(1, n_itr + 1), cost_)
#plt.ylabel('SSE')
#plt.xlabel('iterations')
#plt.show()

def predict(features,weights):
    out_ = np.dot(features,weights)
    return out_
predicted_values = predict(X_std,B_opt)
def lin_reg_plot(features ,targets,pred): #Defining Function for predicition!
    plt.scatter(features,targets,c='steelblue',s=20)
    plt.plot(features,pred,color ='black',lw =2)
    return None
lin_reg_plot(X_std,Y_std,predicted_values)
plt.xlabel("Average Number of Rooms (Standardized)")
plt.ylabel("Price in $1000s(standardized)")
plt.show()

    