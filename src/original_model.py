"""Modelo original: ponderación por regiones del espacio de entrada aprendida con MAP-EM.

Idea
----
Se combinan M expertos calibrados (los benchmarks) con pesos distintos en cada una de K regiones
del espacio de entrada:

    p(y=1 | x) = sum_k r_k(x) sum_m pi_km p_m(x)

- r_k(x): pertenencia suave de x a la región k, de un GMM ajustado con EM propio (no supervisado).
- p_m(x): probabilidad calibrada del experto m (predicciones fuera de pliegue al entrenar).
- pi_k: pesos de la región k en el símplex, con prior Dirichlet(1 + c * g), donde g son los
  pesos globales (EM con K = 1). El prior encoge las regiones con pocos datos hacia lo global.

Propiedad (se verifica en tests/test_original_model.py)
--------------------------------------------------------
Con r fijo y alpha = 1 + c*g >= 1, la log-posterior

    F(pi) = sum_i log( sum_k sum_m r_ik pi_km l_im ) + sum_k sum_m (alpha_km - 1) log pi_km

es cóncava en pi sobre un conjunto convexo (producto de símplex): el logaritmo de una función
afín es cóncavo y (alpha - 1) log pi es cóncavo para alpha >= 1. El EM no la disminuye nunca
(Dempster, Laird y Rubin, 1977), así que converge al óptimo global.

Herramientas numéricas del núcleo: solo numpy y scipy. scikit-learn se usa únicamente para la
interfaz (BaseEstimator, validación de entradas, clone) y para producir las predicciones de los
expertos, que son los benchmarks.
"""
import numpy as np
from scipy.special import logsumexp
from scipy.linalg import cholesky, solve_triangular
from scipy.stats import norm
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

EPS = 1e-6


# --------------------------------------------------------------------------- piezas propias
def _pca_fit(X, n_components):
    """PCA por SVD: devuelve (media, componentes) con n_components direcciones principales."""
    mu = X.mean(axis=0)
    _, _, Vt = np.linalg.svd(X - mu, full_matrices=False)
    return mu, Vt[:n_components]


def _normal_scores_fit(X):
    """Guarda los valores ordenados de cada columna (para puntajes normales por rangos)."""
    return [np.sort(X[:, j]) for j in range(X.shape[1])]


def _normal_scores(X, sorted_cols):
    """Lleva cada columna a cuantiles normales: Phi^-1(rango medio / (n + 1)). Ajustado solo en fit."""
    out = np.empty_like(X, dtype=float)
    for j, s in enumerate(sorted_cols):
        lo = np.searchsorted(s, X[:, j], side="left"); hi = np.searchsorted(s, X[:, j], side="right")
        u = ((lo + hi) / 2 + 0.5) / (len(s) + 1)
        out[:, j] = norm.ppf(np.clip(u, 1e-6, 1 - 1e-6))
    return out


def _gauss_logpdf(Z, mean, cov):
    """log N(Z | mean, cov) por factorización de Cholesky."""
    L = cholesky(cov, lower=True)
    sol = solve_triangular(L, (Z - mean).T, lower=True)
    d = Z.shape[1]
    return -0.5 * (np.sum(sol ** 2, axis=0) + d * np.log(2 * np.pi)) - np.sum(np.log(np.diag(L)))


def _gmm_fit(Z, K, rng, max_iter=200, tol=1e-6, reg_covar=1e-3, covariance="full"):
    """GMM por EM (covarianza 'full' o 'diag'); inicialización k-means++. Devuelve (pesos, medias, covs)."""
    n, d = Z.shape
    centers = [Z[rng.integers(n)]]
    for _ in range(1, K):                                # k-means++
        d2 = np.min([np.sum((Z - c) ** 2, axis=1) for c in centers], axis=0)
        centers.append(Z[rng.choice(n, p=d2 / d2.sum())])
    means = np.array(centers)
    covs = np.array([np.cov(Z.T) + reg_covar * np.eye(d)] * K)
    w = np.full(K, 1.0 / K)
    prev = -np.inf
    for _ in range(max_iter):
        logp = np.column_stack([np.log(w[k]) + _gauss_logpdf(Z, means[k], covs[k]) for k in range(K)])
        ll = logsumexp(logp, axis=1)
        R = np.exp(logp - ll[:, None])                   # paso E
        Nk = R.sum(axis=0) + 1e-12                       # paso M
        w = Nk / n
        means = (R.T @ Z) / Nk[:, None]
        for k in range(K):
            D = Z - means[k]
            C = (R[:, k, None] * D).T @ D / Nk[k]
            covs[k] = (np.diag(np.diag(C)) if covariance == "diag" else C) + reg_covar * np.eye(d)
        if ll.mean() - prev < tol:
            break
        prev = ll.mean()
    return w, means, covs


def _gmm_resp(Z, w, means, covs):
    logp = np.column_stack([np.log(w[k]) + _gauss_logpdf(Z, means[k], covs[k]) for k in range(len(w))])
    return np.exp(logp - logsumexp(logp, axis=1)[:, None])


def log_posterior(pi, Lik, R, alpha):
    """F(pi): log-verosimilitud de la mezcla + log-prior de Dirichlet (sin constante)."""
    mix = np.einsum("ik,km,im->i", R, pi, Lik)
    return np.log(mix).sum() + np.sum((alpha - 1) * np.log(pi))


def map_em(Lik, R, alpha, max_iter=2000, tol=1e-8):
    """MAP-EM de los pesos por región. Lik: n x M verosimilitudes de los expertos; R: n x K.

    Paso E: q_ikm = r_ik pi_km l_im / sum_{k',m'} r_ik' pi_k'm' l_im'
    Paso M: pi_km = (sum_i q_ikm + alpha_km - 1) / (sum_i sum_m q_ikm + sum_m (alpha_km - 1))
    Devuelve (pi, historial de F, convergió).
    """
    K, M = R.shape[1], Lik.shape[1]
    pi = np.full((K, M), 1.0 / M)
    hist = [log_posterior(pi, Lik, R, alpha)]
    for _ in range(max_iter):
        mix = R[:, :, None] * pi[None] * Lik[:, None, :]
        q = mix / mix.sum(axis=(1, 2))[:, None, None]
        num = q.sum(axis=0) + alpha - 1
        pi = np.clip(num / num.sum(axis=1, keepdims=True), 1e-300, None)
        hist.append(log_posterior(pi, Lik, R, alpha))
        if hist[-1] - hist[-2] <= tol * (1 + abs(hist[-2])):
            return pi, np.array(hist), True
    return pi, np.array(hist), False


def _default_experts(random_state):
    return [("logistica", LogisticRegression(max_iter=3000)),
            ("histgb", HistGradientBoostingClassifier(learning_rate=0.05, max_leaf_nodes=15,
                                                      min_samples_leaf=40, random_state=random_state)),
            ("random_forest", RandomForestClassifier(n_estimators=300, min_samples_leaf=5,
                                                     random_state=random_state, n_jobs=-1))]


# --------------------------------------------------------------------------- clase
class RegionalMAPEM(BaseEstimator, ClassifierMixin):
    """Ensamble con pesos por región del espacio de entrada aprendidos con MAP-EM.

    Parámetros
    ----------
    experts : list of (str, estimator) o None
        Benchmarks a combinar. None usa logística, HistGB y Random Forest.
    n_regions : int, número K de regiones (K = 1 equivale a pesos globales).
    prior_strength : float, c >= 0. Intensidad del prior hacia los pesos globales.
    region_columns : list of int o None
        Columnas de X que definen las regiones. None (regla del protocolo, sección 6): las columnas
        con más de 2 valores distintos en fit (variables continuas).
    n_pca : int, componentes principales (de los puntajes normales) sobre las que se ajusta el GMM.
    covariance : 'full' | 'diag', covarianza del GMM de regiones.
    hard_regions : bool, si True cada cliente pertenece solo a su región más probable (ablación).
    inner_cv : int, pliegues para las predicciones fuera de pliegue de los expertos.
    calibration : 'sigmoid' | 'isotonic' | None, calibración de cada experto.
    max_iter, tol : criterio de parada del MAP-EM (cambio relativo de F <= tol, o max_iter).
    random_state : int, semilla (pliegues, GMM y expertos).
    """

    def __init__(self, experts=None, n_regions=4, prior_strength=20.0, region_columns=None, n_pca=2,
                 covariance="full", hard_regions=False, inner_cv=3, calibration="sigmoid",
                 max_iter=500, tol=1e-9, random_state=42):
        self.experts = experts
        self.n_regions = n_regions
        self.prior_strength = prior_strength
        self.region_columns = region_columns
        self.n_pca = n_pca
        self.covariance = covariance
        self.hard_regions = hard_regions
        self.inner_cv = inner_cv
        self.calibration = calibration
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

    def _make(self, est):
        est = clone(est)
        return CalibratedClassifierCV(est, method=self.calibration, cv=3) if self.calibration else est

    def _region_space(self, X):
        U = _normal_scores(X[:, self.region_columns_], self.scores_)
        Z = (U - self.pca_mean_) @ self.pca_components_.T
        return (Z - self.z_mean_) / self.z_std_

    def _regions(self, X):
        if self.n_regions == 1:
            return np.ones((len(X), 1))
        R = _gmm_resp(self._region_space(X), *self.gmm_)
        if self.hard_regions:
            R = np.eye(self.n_regions)[R.argmax(1)]
        return R

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self.classes_ = np.unique(y)
        if len(self.classes_) != 2:
            raise ValueError("RegionalMAPEM solo admite clasificación binaria.")
        if self.n_regions < 1 or self.prior_strength < 0:
            raise ValueError("n_regions debe ser >= 1 y prior_strength >= 0.")
        if self.covariance not in ("full", "diag"):
            raise ValueError("covariance debe ser 'full' o 'diag'.")
        yb = (y == self.classes_[1]).astype(int)
        rng = np.random.default_rng(self.random_state)
        experts = self.experts if self.experts is not None else _default_experts(self.random_state)
        M = len(experts)

        # 1) predicciones fuera de pliegue de cada experto (sin fuga) y expertos finales
        oof = np.zeros((len(yb), M))
        skf = StratifiedKFold(self.inner_cv, shuffle=True, random_state=self.random_state)
        for a, b in skf.split(X, yb):
            for m, (_, est) in enumerate(experts):
                oof[b, m] = self._make(est).fit(X[a], yb[a]).predict_proba(X[b])[:, 1]
        self.experts_ = [self._make(est).fit(X, yb) for _, est in experts]
        P = np.clip(oof, EPS, 1 - EPS)
        Lik = np.where(yb[:, None] == 1, P, 1 - P)

        # 2) regiones (protocolo, sección 6): columnas continuas -> cuantiles normales -> PCA -> GMM propios
        cols = self.region_columns
        if cols is None:
            cols = [j for j in range(X.shape[1]) if len(np.unique(X[:, j])) > 2]
        if len(cols) == 0:
            raise ValueError("No hay columnas continuas para definir las regiones; indique region_columns.")
        self.region_columns_ = list(cols)
        self.scores_ = _normal_scores_fit(X[:, self.region_columns_])
        U = _normal_scores(X[:, self.region_columns_], self.scores_)
        self.pca_mean_, self.pca_components_ = _pca_fit(U, min(self.n_pca, U.shape[1]))
        Z = (U - self.pca_mean_) @ self.pca_components_.T
        self.z_mean_, self.z_std_ = Z.mean(0), Z.std(0) + 1e-12
        if self.n_regions > 1:
            self.gmm_ = _gmm_fit((Z - self.z_mean_) / self.z_std_, self.n_regions, rng, covariance=self.covariance)
        R = self._regions(X)
        self.oof_proba_, self.train_regions_ = P, R          # diagnóstico: OOF de expertos y regiones

        # 3) pesos globales g (EM, K = 1, sin prior) y pesos por región (MAP-EM)
        g, _, _ = map_em(Lik, np.ones((len(yb), 1)), np.ones((1, M)), self.max_iter, self.tol)
        self.global_weights_ = g[0]
        self.alpha_ = 1 + self.prior_strength * np.tile(self.global_weights_, (R.shape[1], 1))
        self.weights_, self.history_, self.converged_ = map_em(Lik, R, self.alpha_, self.max_iter, self.tol)
        self.n_iter_ = len(self.history_) - 1
        self.expert_names_ = [n for n, _ in experts]
        return self

    def expert_proba(self, X):
        check_is_fitted(self, "weights_")
        X = check_array(X)
        return np.column_stack([e.predict_proba(X)[:, 1] for e in self.experts_])

    def predict_proba(self, X):
        check_is_fitted(self, "weights_")
        X = check_array(X)
        P = np.clip(self.expert_proba(X), EPS, 1 - EPS)
        p1 = np.einsum("ik,km,im->i", self._regions(X), self.weights_, P)
        return np.column_stack([1 - p1, p1])

    def predict(self, X):
        return self.classes_[(self.predict_proba(X)[:, 1] >= 0.5).astype(int)]
