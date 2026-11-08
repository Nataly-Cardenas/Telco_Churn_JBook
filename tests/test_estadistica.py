"""Pruebas de las funciones estadísticas del experimento confirmatorio."""
import sys
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments" / "confirmatorio"))
import estadistica as e  # noqa: E402


def test_nadeau_bengio_es_mas_conservadora_que_t_pareada():
    d = np.random.default_rng(0).normal(0.01, 0.01, 25)
    _, p_nb = e.nadeau_bengio(d)
    assert p_nb > stats.ttest_1samp(d, 0).pvalue


def test_permutacion_detecta_efecto_y_respeta_h0():
    rng = np.random.default_rng(0)
    assert e.permutacion_signos(rng.normal(1, 1, 200), 5000)[1] < 0.01
    ps = [e.permutacion_signos(rng.normal(0, 1, 100), 500, s)[1] for s in range(200)]
    assert 0.02 < np.mean(np.array(ps) < 0.05) < 0.09       # tasa de error tipo I cercana a 0.05


def test_holm_caso_conocido():
    np.testing.assert_allclose(e.holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])


def test_cliff_extremos():
    assert e.cliff([3, 4], [1, 2]) == 1 and e.cliff([1, 2], [3, 4]) == -1 and e.cliff([1, 2], [1, 2]) == 0


def test_friedman_coincide_con_scipy():
    M = np.random.default_rng(0).normal(size=(25, 5))
    _, _, _, chi2 = e.friedman_iman_davenport(M)
    assert abs(chi2 - stats.friedmanchisquare(*M.T).statistic) < 1e-9


def test_delong_auc_y_simetria():
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 500)
    a, b = y + rng.normal(0, 1, 500), y + rng.normal(0, 1.5, 500)
    z, p, a1, a2 = e.delong(y, a, b)
    assert abs(a1 - roc_auc_score(y, a)) < 1e-12 and abs(a2 - roc_auc_score(y, b)) < 1e-12
    assert abs(e.delong(y, b, a)[0] + z) < 1e-12 and 0 <= p <= 1
    assert e.delong(y, a, a)[1] == 1.0


def test_mcnemar_exacto():
    a = np.array([True] * 10 + [False] * 2 + [True] * 5)
    b = np.array([False] * 10 + [True] * 2 + [True] * 5)
    assert abs(e.mcnemar_exacto(a, b)[1] - stats.binomtest(10, 12, 0.5).pvalue) < 1e-12


def test_bayes_suma_uno_y_detecta_equivalencia():
    g, i, p = e.bayes_rangos_signo(np.random.default_rng(0).normal(0, 1e-5, 25), rope=0.001, n=2000)
    assert abs(g + i + p - 1) < 1e-9 and i > 0.9
