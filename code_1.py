import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import confusion_matrix, classification_report
import pyodbc

SERVER = 'DESKTOP-US0HB6H'
DATABASE = 'analpreditivo'
connectionString = f'DRIVER={{ODBC Driver 17 for SQL Server}};SERVER={SERVER};DATABASE={DATABASE};Trusted_Connection=yes;'
conn = pyodbc.connect(connectionString)
query = "Select * from Obesity_data"
df = pd.read_sql(query, conn)

correlation_matrix = df.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, cmap='coolwarm', annot=True, fmt=".2f")
plt.title("Matriz de Correlação")
plt.tight_layout()
plt.savefig("matriz_correlacao.png",dpi=300)
plt.show()

print(df.describe())
print(df.isnull().sum())
print(df.duplicated().sum())

sns.countplot(data=df, x='Class')
plt.title("Distribuição de Classes")
plt.tight_layout()
plt.savefig("distribuicao_classes.png",dpi=300)
plt.show()


x = df.drop(['Class'], axis=1)
y = df['Class']
rf_model = RandomForestClassifier(class_weight='balanced', random_state=42)
scores = cross_val_score(rf_model, x, y, cv=5)
print(f'{rf_model.__class__.__name__}: {scores.mean()}')
f1_scores = cross_val_score(rf_model, x, y, cv=5, scoring='f1_weighted')
print(f'{rf_model.__class__.__name__}: F1-Score (weighted) = {f1_scores.mean():.4f}')

selector = SelectKBest(score_func=f_classif, k=10)
x_new = selector.fit_transform(x, y)
selected_features = x.columns[selector.get_support()]
print(selected_features)

RFmodel = RandomForestClassifier(class_weight='balanced', random_state=42)
scores = cross_val_score(RFmodel, x_new, y, cv=5)
print(f'{RFmodel.__class__.__name__}: {scores.mean()}')
f1_scores = cross_val_score(RFmodel, x_new, y, cv=5, scoring='f1_weighted')
print(f'{RFmodel.__class__.__name__}: F1-Score (weighted) = {f1_scores.mean():.4f}')

y_pred = cross_val_predict(rf_model, x_new, y, cv=5)

conf_matrix = confusion_matrix(y, y_pred)

class_names = ['Abaixo do Peso', 'Normal', 'Sobrepeso', 'Obeso']
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predito')
plt.ylabel('Real')
plt.title('Matriz de Confusão com Cross-Validation')
plt.tight_layout()

plt.savefig("matriz_confusao.png",dpi=300)
plt.show()

conf_matrix_df = pd.DataFrame(conf_matrix, index=class_names, columns=class_names)
conf_matrix_df.to_csv("matriz_confusao.csv")
