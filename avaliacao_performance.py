import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    classification_report
)


df = pd.read_excel("renewable_energy_part1_treated.xlsx")


print("Colunas no DataFrame:\n", df.columns.tolist())



target_col = "Energy_Class_encoded"
if target_col not in df.columns:
    raise KeyError(f"A coluna '{target_col}' não foi encontrada no DataFrame.")


X = df.select_dtypes(include=[np.number]).copy()
if target_col in X.columns:
    X = X.drop(columns=[target_col])

y = df[target_col]

print("\nShape X:", X.shape)
print("Distribuição das classes (y):\n", y.value_counts())


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


def treinar_e_avaliar(X, y, test_size, random_state=42):
    """
    Treina uma Regressão Logística (multinomial) e retorna métricas e matriz de confusão.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    model = LogisticRegression(max_iter=1000, multi_class="multinomial", solver="lbfgs", random_state=random_state)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    cm = confusion_matrix(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)
    # Usamos média 'macro' para tratar classes possivelmente desbalanceadas (cada classe tem igual peso)
    prec_macro = precision_score(y_test, y_pred, average="macro", zero_division=0)
    rec_macro = recall_score(y_test, y_pred, average="macro", zero_division=0)
    report = classification_report(y_test, y_pred, zero_division=0)

    results = {
        "model": model,
        "cm": cm,
        "accuracy": acc,
        "precision_macro": prec_macro,
        "recall_macro": rec_macro,
        "report": report,
        "y_test": y_test,
        "y_pred": y_pred
    }
    return results


print("\n--- Rodando cenário: 40% para teste ---")
res_40 = treinar_e_avaliar(X_scaled, y, test_size=0.40)

print("\n--- Rodando cenário: 15% para teste ---")
res_15 = treinar_e_avaliar(X_scaled, y, test_size=0.15)


def mostrar_resultados(res, label):
    print(f"\n===== RESULTADOS: {label} =====")
    print("Matriz de Confusão:\n", res["cm"])
    plt.figure(figsize=(5,4))
    sns.heatmap(res["cm"], annot=True, fmt="d", cmap="Blues")
    plt.title(f"Matriz de Confusão — {label}")
    plt.xlabel("Previsto")
    plt.ylabel("Real")
    plt.show()

    print(f"Acurácia: {res['accuracy']:.4f}")
    print(f"Precisão (macro): {res['precision_macro']:.4f}")
    print(f"Recall (macro): {res['recall_macro']:.4f}")
    print("\nRelatório de Classificação:\n", res["report"])

mostrar_resultados(res_40, "Test size = 40%")
mostrar_resultados(res_15, "Test size = 15%")


print("\n=== COMPARAÇÃO RESUMIDA ===")
print(f"Acurácia (40% test): {res_40['accuracy']:.4f}")
print(f"Acurácia (15% test): {res_15['accuracy']:.4f}")
print(f"Precisão macro (40%): {res_40['precision_macro']:.4f}")
print(f"Precisão macro (15%): {res_15['precision_macro']:.4f}")
print(f"Recall macro (40%): {res_40['recall_macro']:.4f}")
print(f"Recall macro (15%): {res_15['recall_macro']:.4f}")


print("\n--- Observações para relatório ---")
print("1) Avalie se a acurácia é suficientemente alta para o objetivo do trabalho.")
print("2) A precisão e recall macro mostram comportamento médio entre as classes (útil quando há desbalanceamento).")
print("3) Se uma classe for mais crítica (ex.: custo alto por erro), foque em sua precisão/recall específicos no classification_report.")
print("\nVantagens/desvantagens - 40% vs 15% test:")
print("- 40% test: avaliação mais robusta (mais exemplos no conjunto de teste), porém menos dados para treinar.")
print("- 15% test: mais dados para treinar => potencialmente melhor performance do modelo, porém avaliação menos estável (menor conjunto de teste).")
print("\nRecomendação geral: entre 20% e 30% costuma ser um bom equilíbrio; usar validação cruzada (k-fold) traz avaliações mais confiáveis.")
