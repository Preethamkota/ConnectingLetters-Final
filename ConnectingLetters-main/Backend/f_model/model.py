import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib
from sklearn.metrics import accuracy_score,classification_report
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
df=pd.read_csv(r"C:\Users\preet\Downloads\ConnectingLettersFull\ConnectingLetters-main\Backend\preprocessed\landmarks.csv")
min_count = df['label'].value_counts().min()
df = df.groupby('label').apply(
    lambda x: x.sample(n=min_count, random_state=42)
).reset_index(drop=True)
x=df.drop("label",axis=1)
y=df["label"]

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42,stratify=y)

model  =RandomForestClassifier(
    n_estimators=300,
    max_depth=10,
    max_features="sqrt",
    random_state=42,
    min_samples_split=5,
    min_samples_leaf=3, 
    # class_weight="balanced",
    n_jobs=-1
)

model.fit(x_train,y_train)

y_pred=model.predict(x_test)
print("Accuracy: ",accuracy_score(y_test,y_pred))

print(confusion_matrix(y_test,y_pred))

print("classification report: ",classification_report(y_test,y_pred))

print("Train:", model.score(x_train, y_train))
print("Test :", model.score(x_test, y_test))

print(y.value_counts())

joblib.dump(model,r"C:\Users\preet\Downloads\ConnectingLettersFull\ConnectingLetters-main\Backend\f_model\model.pkl")
print("model trained and saved")