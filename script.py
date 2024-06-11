import pandas as pd

# Création du DataFrame
data = {
    "Hours": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "Score": [52, 55, 60, 62, 65, 70, 75, 78, 85, 90]
}
df = pd.DataFrame(data)

# Sauvegarde en fichier CSV
df.to_csv("data.csv", index=False)
print("Fichier data.csv créé avec succès.")
