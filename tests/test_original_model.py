"""Pruebas unitarias del modelo original (sección 5.4 de la guía)."""
import numpy as np
import pytest
from sklearn.base import clone
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

from src.original_model import RegionalMAPEM, log_posterior, map_em

EXPERTOS = [("lr", LogisticRegression(max_iter=2000)), ("nb", GaussianNB()),
            ("arbol", DecisionTreeClassifier(max_depth=4, random_state=0))]


@pytest.fixture(scope="module")
def datos():
    return make_classification(n_samples=600, n_features=6, n_informative=4, random_state=0)


@pytest.fixture(scope="module")
def modelo(datos):
    X, y = datos
    return RegionalMAPEM(experts=EXPERTOS, n_regions=3, random_state=0).fit(X[:450], y[:450])


def test_aprende_mejor_que_azar(datos, modelo):            # (i) aprende en un caso con solución conocida
    X, y = datos
    assert (modelo.predict(X[450:]) == y[450:]).mean() > 0.75


def test_reproducible_con_misma_semilla(datos):              # (ii) reproducibilidad
    X, y = datos
    p1 = RegionalMAPEM(experts=EXPERTOS, n_regions=3, random_state=1).fit(X, y).predict_proba(X)
    p2 = RegionalMAPEM(experts=EXPERTOS, n_regions=3, random_state=1).fit(X, y).predict_proba(X)
    np.testing.assert_allclose(p1, p2)


def test_entradas_invalidas(datos):                          # (iii) entradas inválidas
    X, y = datos
    X_malo = X.copy(); X_malo[0, 0] = np.nan
    with pytest.raises(ValueError):
        RegionalMAPEM(experts=EXPERTOS).fit(X_malo, y)
    with pytest.raises(ValueError):                          # solo clasificación binaria
        RegionalMAPEM(experts=EXPERTOS).fit(X, np.arange(len(y)) % 3)
    with pytest.raises(ValueError):
        RegionalMAPEM(experts=EXPERTOS, prior_strength=-1).fit(X, y)


def test_clone_conserva_hiperparametros():                    # (iv) compatibilidad con clone
    m = RegionalMAPEM(n_regions=6, prior_strength=5.0, random_state=3)
    assert clone(m).get_params() == m.get_params()


def test_log_posterior_nunca_baja(modelo):                   # (v) propiedad teórica: ascenso monótono
    assert np.all(np.diff(modelo.history_) >= -1e-9)


def test_optimo_global_por_concavidad():                     # (v) propiedad teórica: óptimo global
    rng = np.random.default_rng(0)
    n, K, M = 400, 3, 4
    Lik = rng.uniform(0.05, 0.95, (n, M))
    R = rng.dirichlet(np.ones(K), n)
    alpha = 1 + 5 * np.tile(rng.dirichlet(np.ones(M)), (K, 1))
    pi, hist, ok = map_em(Lik, R, alpha, max_iter=20000, tol=1e-12)
    F_opt = log_posterior(pi, Lik, R, alpha)
    for _ in range(300):                                     # ningún punto del dominio supera al EM
        otro = rng.dirichlet(np.ones(M), K)
        assert log_posterior(otro, Lik, R, alpha) <= F_opt + 1e-6
    for t in np.linspace(0.05, 0.95, 10):                    # concavidad: F(t a + (1-t) b) >= t F(a) + (1-t) F(b)
        a, b = rng.dirichlet(np.ones(M), K), rng.dirichlet(np.ones(M), K)
        assert log_posterior(t * a + (1 - t) * b, Lik, R, alpha) >= \
            t * log_posterior(a, Lik, R, alpha) + (1 - t) * log_posterior(b, Lik, R, alpha) - 1e-9


def test_salidas_validas(datos, modelo):                     # probabilidades y pesos bien formados
    X, _ = datos
    P = modelo.predict_proba(X)
    np.testing.assert_allclose(P.sum(1), 1)
    assert (P >= 0).all() and (P <= 1).all()
    np.testing.assert_allclose(modelo.weights_.sum(1), 1)


def test_un_region_equivale_a_pesos_globales(datos):         # K = 1 y c = 0: mezcla global
    X, y = datos
    m = RegionalMAPEM(experts=EXPERTOS, n_regions=1, prior_strength=0.0, random_state=0).fit(X, y)
    np.testing.assert_allclose(m.weights_[0], m.global_weights_, atol=1e-4)


def test_variantes_del_protocolo(datos):                      # covarianza diagonal y regiones duras (ablación)
    X, y = datos
    for kw in [dict(covariance="diag"), dict(hard_regions=True)]:
        m = RegionalMAPEM(experts=EXPERTOS, n_regions=3, random_state=0, **kw).fit(X, y)
        P = m.predict_proba(X)
        np.testing.assert_allclose(P.sum(1), 1)
        assert np.all(np.diff(m.history_) >= -1e-9)
    with pytest.raises(ValueError):
        RegionalMAPEM(experts=EXPERTOS, covariance="otra").fit(X, y)
