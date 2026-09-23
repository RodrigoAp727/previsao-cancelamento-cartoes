import numpy as np
import pandas as pd
import pytest

from src import modelagem as m
from src.pontuar import faixa_de_risco


def test_curva_campanha_contabiliza_receita_salva_e_custo():
    y = pd.Series([1, 0, 1, 0])
    score = np.array([0.9, 0.8, 0.7, 0.1])
    receita = pd.Series([100.0, 50.0, 200.0, 80.0])
    p = m.PremissasCampanha(custo_por_contato=10, taxa_sucesso_retencao=0.5, horizonte_anos=2)

    curva = m.curva_campanha(y, score, receita, p)

    # Contatando os 3 de maior risco: salva 50% x 2 anos x (100 + 200) = 300, custa 3 x 10 = 30
    linha = curva.iloc[2]
    assert linha["cancelamentos_capturados"] == 2
    assert linha["receita_salva"] == pytest.approx(300)
    assert linha["lucro"] == pytest.approx(270)
    assert curva["pct_cancelamentos_capturados"].iloc[-1] == 1


def test_tabela_decis_ordena_do_maior_para_o_menor_risco():
    rng = np.random.default_rng(0)
    score = rng.random(1000)
    y = pd.Series((score > 0.8).astype(int))

    decis = m.tabela_decis(y, score)

    assert len(decis) == 10
    assert decis["taxa_cancelamento"].iloc[0] == 1
    assert decis["taxa_cancelamento"].iloc[-1] == 0
    assert decis["pct_acumulado_capturado"].iloc[-1] == pytest.approx(1)


def test_sinais_de_risco_aplicam_os_limites_da_regra():
    cliente_em_risco = {"qtd_transacoes_12m": 60, "var_qtd_q4_q1": 0.59, "contatos_12m": 3,
                        "qtd_produtos": 2, "saldo_rotativo": 0}
    cliente_saudavel = {"qtd_transacoes_12m": 61, "var_qtd_q4_q1": 0.6, "contatos_12m": 2,
                        "qtd_produtos": 3, "saldo_rotativo": 1}

    sinais = m.sinais_de_risco(pd.DataFrame([cliente_em_risco, cliente_saudavel]))

    assert sinais.iloc[0].all()
    assert not sinais.iloc[1].any()


@pytest.mark.parametrize("probabilidade, esperado", [
    (0.95, "Crítico"), (0.7, "Crítico"), (0.5, "Alto"), (0.15, "Moderado"), (0.01, "Baixo"),
])
def test_faixa_de_risco(probabilidade, esperado):
    assert faixa_de_risco(probabilidade) == esperado


def test_modelo_supera_o_baseline_na_base_real():
    from sklearn.metrics import average_precision_score
    from sklearn.model_selection import train_test_split

    from src import dados

    df, _ = dados.tratar(dados.carregar_bruto())
    x, y = m.separar_x_y(df)
    x_tr, x_te, y_tr, y_te = train_test_split(x, y, test_size=0.2, stratify=y, random_state=m.SEMENTE)

    ap_boosting = average_precision_score(y_te, m.modelo_boosting(x).fit(x_tr, y_tr).predict_proba(x_te)[:, 1])
    ap_baseline = average_precision_score(y_te, m.modelo_baseline(x).fit(x_tr, y_tr).predict_proba(x_te)[:, 1])

    assert ap_boosting > 0.9
    assert ap_boosting > ap_baseline
