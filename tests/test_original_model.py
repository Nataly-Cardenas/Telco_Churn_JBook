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


def test_fit_from_predictions_equivale_a_fit(datos, modelo):  # misma ruta de regiones y pesos
    X, y = datos
    Xa, ya = X[:450], y[:450]
    m2 = RegionalMAPEM(n_regions=3, random_state=0).fit_from_predictions(Xa, ya, modelo.oof_proba_)
    np.testing.assert_allclose(m2.weights_, modelo.weights_)
    P_test = modelo.expert_proba(X[450:])
    np.testing.assert_allclose(m2.predict_proba_from_predictions(X[450:], P_test), modelo.predict_proba(X[450:]))
    with pytest.raises(ValueError):                          # sin expertos entrenados no hay predict_proba
        m2.predict_proba(X[450:])
    with pytest.raises(ValueError):                          # filas que no coinciden
        RegionalMAPEM().fit_from_predictions(Xa, ya, modelo.oof_proba_[:10])


def test_gmm_varias_inicializaciones_no_empeora(datos):      # gmm_n_init conserva la mejor verosimilitud
    from src.original_model import _gmm_fit_best, _gmm_loglik
    X, _ = datos
    Z = (X[:, :2] - X[:, :2].mean(0)) / X[:, :2].std(0)
    una = _gmm_loglik(Z, *_gmm_fit_best(Z, 4, np.random.default_rng(0), n_init=1))
    cinco = _gmm_loglik(Z, *_gmm_fit_best(Z, 4, np.random.default_rng(0), n_init=5))
    assert cinco >= una - 1e-12


def test_optimo_global_en_el_borde():                         # (v) supuesto S4: alpha = 1, óptimo con pesos nulos
    from scipy.optimize import minimize
    rng = np.random.default_rng(1)
    n, M = 500, 3
    y = rng.integers(0, 2, n)
    p_bueno = np.clip(np.where(y == 1, 0.8, 0.2) + rng.normal(0, 0.05, n), 0.01, 0.99)
    P = np.column_stack([p_bueno, np.full(n, 0.5), 1 - p_bueno])   # 2.º y 3.º expertos inútiles
    Lik = np.where(y[:, None] == 1, P, 1 - P)
    R, alpha = np.ones((n, 1)), np.ones((1, M))
    pi, _, _ = map_em(Lik, R, alpha, max_iter=20000, tol=1e-14)
    res = minimize(lambda v: -log_posterior(np.abs(v).reshape(1, M) / np.abs(v).sum(), Lik, R, alpha),
                   np.full(M, 1 / M), method="Nelder-Mead", options=dict(xatol=1e-10, fatol=1e-12))
    F_ref = -res.fun
    assert log_posterior(pi, Lik, R, alpha) >= F_ref - 1e-6  # el EM alcanza el máximo global
    assert pi[0, 2] < 1e-3                                   # y el experto dañino queda en el borde (peso ~0)


def test_variantes_del_protocolo(datos):                      # covarianza diagonal y regiones duras (ablación)
    X, y = datos
    for kw in [dict(covariance="diag"), dict(hard_regions=True)]:
        m = RegionalMAPEM(experts=EXPERTOS, n_regions=3, random_state=0, **kw).fit(X, y)
        P = m.predict_proba(X)
        np.testing.assert_allclose(P.sum(1), 1)
        assert np.all(np.diff(m.history_) >= -1e-9)
    with pytest.raises(ValueError):
        RegionalMAPEM(experts=EXPERTOS, covariance="otra").fit(X, y)
