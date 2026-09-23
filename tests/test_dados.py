import pandas as pd
import pytest

from src import dados


@pytest.fixture(scope="module")
def base_tratada():
    df, log = dados.tratar(dados.carregar_bruto())
    return df, log


def test_corrige_razoes_que_perderam_o_separador_decimal():
    df = pd.DataFrame({"var_valor_q4_q1": [1335.0, 0.7, 2.5], "var_qtd_q4_q1": [3714.0, 1.0, 0.0]})

    corrigido, contagem = dados.corrigir_escala_razoes(df)

    assert corrigido["var_valor_q4_q1"].tolist() == pytest.approx([1.335, 0.7, 2.5])
    assert corrigido["var_qtd_q4_q1"].tolist() == pytest.approx([3.714, 1.0, 0.0])
    assert contagem == {"var_valor_q4_q1": 1, "var_qtd_q4_q1": 1}


def test_correcao_nao_altera_o_dataframe_original():
    df = pd.DataFrame({"var_valor_q4_q1": [1335.0], "var_qtd_q4_q1": [0.5]})
    dados.corrigir_escala_razoes(df)
    assert df.loc[0, "var_valor_q4_q1"] == 1335.0


def test_base_tratada_preserva_todos_os_clientes(base_tratada):
    df, log = base_tratada
    assert len(df) == log["linhas_brutas"] == 10_127
    assert log["ids_duplicados"] == 0


def test_razoes_ficam_no_intervalo_da_fonte_original(base_tratada):
    df, _ = base_tratada
    assert df["var_valor_q4_q1"].max() == pytest.approx(3.397)
    assert df["var_qtd_q4_q1"].max() == pytest.approx(3.714)


def test_alvo_e_taxa_de_cancelamento(base_tratada):
    df, _ = base_tratada
    assert set(df["churn"].unique()) == {0, 1}
    assert df["churn"].sum() == 1_627


def test_validar_rejeita_identidade_contabil_quebrada(base_tratada):
    df, _ = base_tratada
    quebrada = df.copy()
    quebrada.loc[quebrada.index[0], "limite_disponivel"] += 100
    with pytest.raises(AssertionError, match="limite"):
        dados.validar(quebrada)
