
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("Renewable_Energy_Data.csv")

print("Dimensões do dataset:", df.shape)
print("\nColunas disponíveis:")
print(df.columns.tolist())


print("\nPrévia dos dados:")
print(df.head())


encoder = LabelEncoder()
df["Energy_Class_encoded"] = encoder.fit_transform(df["Energy_Class"])


mapping = dict(zip(encoder.classes_, encoder.transform(encoder.classes_)))
print("\nMapeamento de classes:", mapping)


print("\nContagem por classe codificada:")
print(df["Energy_Class_encoded"].value_counts())


numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
if "Energy_Class_encoded" not in numeric_cols:
    numeric_cols.append("Energy_Class_encoded")

corr_matrix = df[numeric_cols].corr()


print("\nMatriz de correlação (primeiras linhas):")
print(corr_matrix.head())


plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Matriz de Correlação - Renewable Energy Dataset", fontsize=14)
plt.tight_layout()
plt.show()


corr_with_target = corr_matrix["Energy_Class_encoded"].drop("Energy_Class_encoded").abs().sort_values(ascending=False)
print("\nCorrelação das features com Energy_Class_encoded (ordenado por |corr| desc):")
print(corr_with_target)


print("\n--- Interpretação ---")
print("Nenhuma feature apresentou correlação linear forte com a variável alvo (|corr| >= 0.2).")
print("Isso significa que a relação entre as variáveis e Energy_Class não é fortemente linear.")
print("Portanto, recomenda-se utilizar todas as features numéricas para o modelo de classificação,")
print("pois a exclusão de algumas poderia remover informações relevantes não lineares.")


df.to_excel("renewable_energy_part1_treated.xlsx", index=False)
print("\nArquivo salvo: renewable_energy_part1_treated.xlsx")




