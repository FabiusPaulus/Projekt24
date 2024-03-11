import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Daten einlesen
df = pd.read_csv('data\kidney_disease.csv')

# "?" durch NaN ersetzen
df.replace("?", np.nan, inplace=True)

# Spaltenüberschriften ändern
df.rename(columns={'sg': 'specific_gravity', 'hemo': 'haemoglobin', 'pcv': 'packed_cell_volume'}, inplace=True)

# Datentypen überprüfen und korrigieren
df['class'] = df['class'].astype('category')
df['dm'] = df['dm'].replace(['\tyes','\tno',' yes',' no','\tno','\tyes'],'yes')
df['dm'] = df['dm'].replace([' yes',' no'], ['yes','no'])
df['dm'] = df['dm'].map({'yes': 1, 'no': 0})
df['dm'] = pd.to_numeric(df['dm'], errors='coerce')

# Kategorische und numerische Spalten identifizieren
categorical_columns = []
binary_columns = []
numerical_columns = []

for col in df.columns:
    if df[col].dtype == 'object':
        unique_values = df[col].dropna().unique()
        if len(unique_values) == 2:
            binary_columns.append(col)
        else:
            categorical_columns.append(col)
    elif df[col].dtype == 'float64' or df[col].dtype == 'int64':
        numerical_columns.append(col)

# Kategorische Spalten mit nur zwei Optionen in 0 und 1 umwandeln
for col in binary_columns:
    df[col] = pd.get_dummies(df[col], drop_first=True)

# Alle Spalten außer "class", "specific_gravity", "haemoglobin" und "packed_cell_volume" löschen
columns_to_keep = ['class', 'specific_gravity', 'haemoglobin', 'packed_cell_volume']
df = df[columns_to_keep]

# Feature- und Zielvariablen definieren
X = df.drop('class', axis=1)  # Feature-Variablen
y = df['class']  # Zielvariable

# Label-Encoding für die Zielvariable (um ckd/notckd in numerische Werte umzuwandeln)
le = LabelEncoder()
y = le.fit_transform(y)

# Datenaufteilung in Trainings- und Testsets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Modellbildung (hier als Beispiel mit Random Forest)
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

# Vorhersagen treffen
y_pred = rf_model.predict(X_test)

# Modellbewertung
accuracy = accuracy_score(y_test, y_pred)
print("Genauigkeit:", accuracy)
print("Klassifikationsbericht:")
print(classification_report(y_test, y_pred))
print(df)