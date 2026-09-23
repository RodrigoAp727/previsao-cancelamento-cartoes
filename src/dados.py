"""Carga, limpeza e validação da base de clientes de cartão de crédito."""

from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
CAMINHO_BRUTO = RAIZ / "data" / "raw" / "ClientesBanco.csv"
CAMINHO_TRATADO = RAIZ / "data" / "processed" / "clientes_tratado.csv"

# Nomes originais (em português, com acentos) -> snake_case sem acentos
MAPA_COLUNAS = {
    "CLIENTNUM": "id_cliente",
    "Categoria": "categoria",
    "Idade": "idade",
    "Sexo": "sexo",
    "Dependentes": "dependentes",
    "Educação": "escolaridade",
    "Estado Civil": "estado_civil",
    "Faixa Salarial Anual": "faixa_salarial",
    "Categoria Cartão": "categoria_cartao",
    "Meses como Cliente": "meses_cliente",
    "Produtos Contratados": "qtd_produtos",
    "Inatividade 12m": "meses_inativo_12m",
    "Contatos 12m": "contatos_12m",
    "Limite": "limite",
    "Limite Consumido": "saldo_rotativo",
    "Limite Disponível": "limite_disponivel",
    "Mudanças Transacoes_Q4_Q1": "var_valor_q4_q1",
    "Valor Transacoes 12m": "valor_transacoes_12m",
    "Qtde Transacoes 12m": "qtd_transacoes_12m",
    "Mudança Qtde Transações_Q4_Q1": "var_qtd_q4_q1",
    "Taxa de Utilização Cartão": "taxa_utilizacao",
}

COLUNAS_CATEGORICAS = [
    "sexo", "escolaridade", "estado_civil", "faixa_salarial", "categoria_cartao",
]

ORDEM_FAIXA_SALARIAL = [
    "Less than $40K", "$40K - $60K", "$60K - $80K", "$80K - $120K", "$120K +", "Não informado",
]

# Razões Q4/Q1 acima deste valor só ocorrem quando o separador decimal foi perdido
# (ex.: 1.335 gravado como 1335). O máximo legítimo na fonte original é ~3,7.
LIMIAR_ERRO_ESCALA = 10


def carregar_bruto(caminho: Path = CAMINHO_BRUTO) -> pd.DataFrame:
    return pd.read_csv(caminho, encoding="latin1")


def corrigir_escala_razoes(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Divide por 1000 as razões Q4/Q1 que perderam o separador decimal."""
    df = df.copy()
    corrigidas = {}
    for coluna in ["var_valor_q4_q1", "var_qtd_q4_q1"]:
        mascara = df[coluna] >= LIMIAR_ERRO_ESCALA
        df.loc[mascara, coluna] = df.loc[mascara, coluna] / 1000
        corrigidas[coluna] = int(mascara.sum())
    return df, corrigidas


def tratar(df_bruto: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Aplica todas as regras de limpeza e devolve a base tratada + um log do que mudou."""
    df = df_bruto.rename(columns=MAPA_COLUNAS)
    log = {"linhas_brutas": len(df), "ids_duplicados": int(df["id_cliente"].duplicated().sum())}

    # Único nulo da base está em categoria_cartao; imputar como "Não informado"
    # preserva o cliente em vez de descartá-lo (como fazia o dropna original).
    log["nulos"] = {k: int(v) for k, v in df.isna().sum().items() if v}
    df["categoria_cartao"] = df["categoria_cartao"].fillna("Não informado")

    df, log["razoes_corrigidas"] = corrigir_escala_razoes(df)

    df["churn"] = (df["categoria"] == "Cancelado").astype(int)
    df["ticket_medio"] = df["valor_transacoes_12m"] / df["qtd_transacoes_12m"]

    for coluna in COLUNAS_CATEGORICAS:
        df[coluna] = df[coluna].astype("category")
    df["faixa_salarial"] = df["faixa_salarial"].cat.reorder_categories(ORDEM_FAIXA_SALARIAL)

    validar(df)
    return df, log


def validar(df: pd.DataFrame) -> None:
    """Regras de consistência que a base precisa respeitar depois da limpeza."""
    assert df["id_cliente"].is_unique, "id_cliente duplicado"
    assert df.drop(columns=COLUNAS_CATEGORICAS).notna().all().all(), "nulos restantes"
    assert df[["var_valor_q4_q1", "var_qtd_q4_q1"]].max().max() < LIMIAR_ERRO_ESCALA
    assert df["taxa_utilizacao"].between(0, 1).all()
    # Identidade contábil: limite = saldo rotativo + limite disponível
    diferenca = (df["limite"] - df["saldo_rotativo"] - df["limite_disponivel"]).abs()
    assert (diferenca < 1e-6).all(), "limite != saldo_rotativo + limite_disponivel"
