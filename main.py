import pandas as pd
#read dataset
data=pd.read_csv("dataset/Crop_recommendation.csv")
print("Crop Recommendation Dataset")
print(data.head())
print("\nColumns:")
print(data.columns)
print("\nDataset Shape:")
print(data.shape)
print("\nMissing values:")
print(data.isnull().sum())
print("\nDataset Information:")
print(data.info())
print("\nStatistical Summary:")
print(data.describe())
print("\nCrop Labels:")
print(data['label'].unique())
print("\nNumber of Crop Types:")
print(data['label'].nunique())
import matplotlib.pyplot as plt
plt.figure(figsize=(10,5))
data['label'].value_counts().plot(kind='bar')
plt.title("Crop distribution")
plt.xlabel("Crop")
plt.ylabel("Count")
plt.show()
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data['label']=le.fit_transform(data['label'])
X=data.drop('label',axis=1)
Y=data['label']
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=42)
print("Training Data:",X_train.shape)
print("Testing Data:",X_test.shape)
from sklearn.tree import DecisionTreeClassifier
model=DecisionTreeClassifier(random_state=42)
model.fit(X_train,Y_train)
print("model training completed")
from sklearn.metrics import accuracy_score
Y_pred=model.predict(X_test)
accuracy=accuracy_score(Y_test,Y_pred)
print("Model accuracy:",accuracy)
#user input
N=float(input("Enter nitrogen(N):"))
P=float(input("Enter Phosphorus(P):"))
K=float(input("Enter Potassium(K):"))
temperature=float(input("Enter temerature:"))
humidity=float(input("Enter humidity:"))
ph=float(input("Enter pH:"))
rainfall=float(input("Enter Rainfall:"))
user_data=[[N,P,K,temperature,humidity,ph,rainfall]]
prediction=model.predict(user_data)
crop=le.inverse_transform(prediction)
print("\nRecommended Crop:",crop[0])
import joblib
joblib.dump(model,"crop_model.pkl")
joblib.dump(le,"label_encoder.pkl")
print("Model saved successfully!")
