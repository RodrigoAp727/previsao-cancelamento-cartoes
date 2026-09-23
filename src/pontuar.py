"""Gera a fila de retenção: clientes ativos ordenados por risco de cancelamento.

Uso:
    python -m src.pontuar                    # clientes ativos com risco a partir de "Moderado" (15%)
    python -m src.pontuar --prob-minima 0.4  # só risco "Alto" e "Crítico"

As probabilidades são calculadas fora da amostra (cross_val_predict): cada cliente é pontuado
por um modelo que não o viu no treino, o que evita probabilidades otimistas demais.
"""

import argparse

import joblib
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from src import dados, modelagem as m

CAMINHO_FILA = dados.RAIZ / "reports" / "fila_retencao.csv"
CAMINHO_MODELO = dados.RAIZ / "models" / "modelo_cancelamento.joblib"
FAIXAS_RISCO = [(0.7, "Crítico"), (0.4, "Alto"), (0.15, "Moderado"), (0.0, "Baixo")]


def faixa_de_risco(probabilidade: float) -> str:
    return next(nome for limite, nome in FAIXAS_RISCO if probabilidade >= limite)


def gerar_fila(prob_minima: float = 0.15) -> pd.DataFrame:
    df, _ = dados.tratar(dados.carregar_bruto())
    x, y = m.separar_x_y(df)
    modelo = m.modelo_boosting(x)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=m.SEMENTE)
    df["prob_cancelamento"] = cross_val_predict(modelo, x, y, cv=cv, method="predict_proba")[:, 1]

    # Modelo final treinado com toda a base, salvo para pontuar clientes novos
    CAMINHO_MODELO.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(modelo.fit(x, y), CAMINHO_MODELO)

    sinais = m.sinais_de_risco(df)
    df["sinais_de_risco"] = sinais.apply(lambda linha: "; ".join(linha.index[linha]), axis=1)
    df["receita_anual_estimada"] = m.receita_anual_estimada(df, m.PremissasCampanha()).round(2)
    df["faixa_risco"] = df["prob_cancelamento"].map(faixa_de_risco)

    em_risco = (df["categoria"] == "Cliente") & (df["prob_cancelamento"] >= prob_minima)
    fila = df[em_risco].sort_values("prob_cancelamento", ascending=False)
    colunas = ["id_cliente", "prob_cancelamento", "faixa_risco", "sinais_de_risco", "receita_anual_estimada",
               "qtd_transacoes_12m", "var_qtd_q4_q1", "contatos_12m", "qtd_produtos", "saldo_rotativo"]
    return fila[colunas].round({"prob_cancelamento": 4, "var_qtd_q4_q1": 3})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--prob-minima", type=float, default=0.15,
                        help="probabilidade mínima de cancelamento para entrar na fila (padrão: 0.15)")
    args = parser.parse_args()

    fila = gerar_fila(args.prob_minima)
    CAMINHO_FILA.parent.mkdir(parents=True, exist_ok=True)
    fila.to_csv(CAMINHO_FILA, index=False, encoding="utf-8-sig")
    print(f"Fila com {len(fila):,} clientes salva em {CAMINHO_FILA.relative_to(dados.RAIZ)}")
    print(fila["faixa_risco"].value_counts().to_string())
    print(f"Receita anual estimada em jogo: ${fila['receita_anual_estimada'].sum():,.0f}")


if __name__ == "__main__":
    main()
