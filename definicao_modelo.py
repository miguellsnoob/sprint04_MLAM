import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, classification_report


df = pd.read_excel("renewable_energy_part1_treated.xlsx")

# Definindo features (X) e target (y)
X = df.select_dtypes(include=["float64", "int64"]).drop(columns=["Energy_Class_encoded"], errors="ignore")
y = df["Energy_Class_encoded"]

# Padronizando os dados (importante para modelos lineares)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


def treinar_modelo(test_size):
    print(f"\n===== CENÁRIO: {int(test_size*100)}% de teste =====")

    # Separar dados de treino e teste
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=test_size, random_state=42, stratify=y
    )

    # Criar e treinar o modelo linear (Regressão Logística)
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)

    # Fazer previsões
    y_pred = model.predict(X_test)

    # Avaliar o modelo
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted")
    rec = recall_score(y_test, y_pred, average="weighted")
    cm = confusion_matrix(y_test, y_pred)

    # Mostrar resultados
    print("\nMatriz de Confusão:")
    print(cm)
    print("\nAcurácia:", round(acc, 3))
    print("Precisão:", round(prec, 3))
    print("Recall:", round(rec, 3))
    print("\nRelatório completo:")
    print(classification_report(y_test, y_pred, zero_division=0))

    return {"acuracia": acc, "precisao": prec, "recall": rec, "matriz_confusao": cm}


resultados_40 = treinar_modelo(0.40)
resultados_15 = treinar_modelo(0.15)


print("\n===== COMPARAÇÃO ENTRE OS CENÁRIOS =====")
print(f"Acurácia com 40% teste: {resultados_40['acuracia']:.3f}")
print(f"Acurácia com 15% teste: {resultados_15['acuracia']:.3f}")

print("\n--- Interpretação ---")
print("Quando usamos 40% dos dados para teste, o conjunto de treinamento fica menor,")
print("o que pode reduzir a capacidade do modelo de aprender bem (possível menor acurácia).")
print("Quando usamos 15% para teste, temos mais dados para treino, o que tende a melhorar")
print("a performance, mas reduz a confiabilidade da avaliação (menos dados para validar).")
print("Em geral, há um equilíbrio: 20%-30% de teste costuma ser o ideal em datasets médios.")
