"""Modelo de propensão a cancelamento e simulação de retorno de campanhas de retenção."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import make_column_transformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.dados import COLUNAS_CATEGORICAS

ALVO = "churn"
# limite_disponivel = limite - saldo_rotativo (identidade exata): fica de fora por redundância
COLUNAS_EXCLUIDAS = ["id_cliente", "categoria", ALVO, "limite_disponivel"]
SEMENTE = 42


def separar_x_y(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    return df.drop(columns=COLUNAS_EXCLUIDAS), df[ALVO]


def sinais_de_risco(df: pd.DataFrame) -> pd.DataFrame:
    """Regra simples, traduzível para SQL, com os 5 sinais encontrados na análise exploratória."""
    return pd.DataFrame({
        "Poucas transações (≤ 60)": df["qtd_transacoes_12m"] <= 60,
        "Queda de uso no trimestre (Q4/Q1 < 0,6)": df["var_qtd_q4_q1"] < 0.6,
        "3+ contatos com o banco": df["contatos_12m"] >= 3,
        "Até 2 produtos": df["qtd_produtos"] <= 2,
        "Rotativo zerado": df["saldo_rotativo"] == 0,
    }, index=df.index)


def _colunas_numericas(x: pd.DataFrame) -> list[str]:
    return [c for c in x.columns if c not in COLUNAS_CATEGORICAS]


def modelo_baseline(x: pd.DataFrame) -> Pipeline:
    """Regressão logística: referência simples e interpretável."""
    preprocessamento = make_column_transformer(
        (OneHotEncoder(handle_unknown="ignore"), COLUNAS_CATEGORICAS),
        (StandardScaler(), _colunas_numericas(x)),
    )
    return make_pipeline(preprocessamento, LogisticRegression(max_iter=2000, class_weight="balanced"))


def modelo_boosting(x: pd.DataFrame) -> Pipeline:
    """Gradient boosting: captura as relações não lineares vistas na análise exploratória."""
    preprocessamento = make_column_transformer(
        (OneHotEncoder(handle_unknown="ignore", sparse_output=False), COLUNAS_CATEGORICAS),
        remainder="passthrough",
        verbose_feature_names_out=False,
    )
    classificador = HistGradientBoostingClassifier(
        learning_rate=0.05, max_iter=500, class_weight="balanced", random_state=SEMENTE,
    )
    return make_pipeline(preprocessamento, classificador)


@dataclass
class PremissasCampanha:
    """Premissas ilustrativas: a base não traz receita, então o valor é estimado a partir do uso.

    Todas são parâmetros; a área de negócio deve substituí-las pelos números reais.
    """
    taxa_intercambio: float = 0.02        # receita sobre o valor transacionado no ano
    juros_rotativo_anual: float = 0.18    # receita sobre o saldo rotativo médio
    custo_por_contato: float = 30.0       # custo de uma ação de retenção (oferta + operação)
    taxa_sucesso_retencao: float = 0.25   # fração de clientes em risco que a ação consegue reter
    horizonte_anos: float = 3.0           # anos de receita preservados por cliente retido (CLV simplificado)


def receita_anual_estimada(df: pd.DataFrame, p: PremissasCampanha) -> pd.Series:
    return p.taxa_intercambio * df["valor_transacoes_12m"] + p.juros_rotativo_anual * df["saldo_rotativo"]


def curva_campanha(y: pd.Series, score: np.ndarray, receita: pd.Series, p: PremissasCampanha) -> pd.DataFrame:
    """Lucro esperado ao contatar os N clientes de maior risco, para todo N."""
    ordem = np.argsort(-score)
    y_ord = y.to_numpy()[ordem]
    receita_ord = receita.to_numpy()[ordem]
    n = np.arange(1, len(y_ord) + 1)
    receita_salva = np.cumsum(y_ord * receita_ord) * p.taxa_sucesso_retencao * p.horizonte_anos
    custo = n * p.custo_por_contato
    return pd.DataFrame({
        "clientes_contatados": n,
        "pct_base": n / len(n),
        "cancelamentos_capturados": np.cumsum(y_ord),
        "pct_cancelamentos_capturados": np.cumsum(y_ord) / y_ord.sum(),
        "receita_salva": receita_salva,
        "custo": custo,
        "lucro": receita_salva - custo,
    })


def tabela_decis(y: pd.Series, score: np.ndarray) -> pd.DataFrame:
    """Taxa de cancelamento e lift por decil de risco (decil 1 = maior risco)."""
    tabela = pd.DataFrame({"y": y.to_numpy(), "score": score})
    tabela["decil"] = pd.qcut(tabela["score"].rank(method="first", ascending=False), 10, labels=range(1, 11))
    resumo = tabela.groupby("decil", observed=True).agg(
        clientes=("y", "size"), cancelados=("y", "sum"), score_medio=("score", "mean"),
    )
    resumo["taxa_cancelamento"] = resumo["cancelados"] / resumo["clientes"]
    resumo["lift"] = resumo["taxa_cancelamento"] / tabela["y"].mean()
    resumo["pct_acumulado_capturado"] = resumo["cancelados"].cumsum() / tabela["y"].sum()
    return resumo
