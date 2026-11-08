"""Pruebas de la sección 8 del protocolo. Funciones puras, sin lectura de archivos."""
import numpy as np
from scipy import stats


def nadeau_bengio(d, razon=0.25):
    """t corregida de Nadeau y Bengio (2003) sobre J diferencias. razon = n_test / n_train."""
    d = np.asarray(d, float)
    J = len(d)
    var = d.var(ddof=1)
    if var == 0:
        return 0.0, 1.0
    t = d.mean() / np.sqrt((1 / J + razon) * var)
    return float(t), float(2 * stats.t.sf(abs(t), J - 1))


def permutacion_signos(d, n=100_000, semilla=0):
    """Permutación pareada de signos sobre las pérdidas por cliente. p = (1 + #extremos) / (n + 1)."""
    d = np.asarray(d, float)
    rng = np.random.default_rng(semilla)
    obs = abs(d.mean())
    ext, hechos = 0, 0
    while hechos < n:
        b = min(2000, n - hechos)
        signos = rng.integers(0, 2, (b, len(d)), dtype=np.int8) * 2 - 1
        ext += int(np.sum(np.abs(signos @ d) / len(d) >= obs - 1e-15))
        hechos += b
    return float(d.mean()), float((1 + ext) / (n + 1))


def wilcoxon(d):
    d = np.asarray(d, float)
    if np.allclose(d, 0):
        return 0.0, 1.0
    r = stats.wilcoxon(d, zero_method="wilcox", alternative="two-sided", method="auto")
    return float(r.statistic), float(r.pvalue)


def friedman_iman_davenport(M):
    """M: bloques x modelos con pérdidas (menor es mejor). Devuelve F_F, p, rangos medios, chi2."""
    N, k = M.shape
    R = np.apply_along_axis(stats.rankdata, 1, M).mean(0)
    chi2 = 12 * N / (k * (k + 1)) * (np.sum(R ** 2) - k * (k + 1) ** 2 / 4)
    denom = N * (k - 1) - chi2
    if denom <= 0:
        return np.inf, 0.0, R, chi2
    ff = (N - 1) * chi2 / denom
    return float(ff), float(stats.f.sf(ff, k - 1, (k - 1) * (N - 1))), R, float(chi2)


def nemenyi_p(Ri, Rj, k, N):
    """p del par de Nemenyi con la distribución del rango estudentizado."""
    q = abs(Ri - Rj) / np.sqrt(k * (k + 1) / (6 * N))
    return float(stats.studentized_range.sf(q * np.sqrt(2), k, 1e6))


def diferencia_critica(k, N, alfa=0.05):
    return float(stats.studentized_range.isf(alfa, k, 1e6) / np.sqrt(2) * np.sqrt(k * (k + 1) / (6 * N)))


def cliff(a, b):
    """δ de Cliff entre dos muestras: P(a > b) - P(a < b)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    dif = a[:, None] - b[None, :]
    return float((np.sum(dif > 0) - np.sum(dif < 0)) / (len(a) * len(b)))


def magnitud_cliff(d):
    d = abs(d)
    return "despreciable" if d < 0.147 else "pequeño" if d < 0.33 else "mediano" if d < 0.474 else "grande"


def bca(d, n=10_000, semilla=0):
    """Intervalo BCa del 95 % de la media de las diferencias."""
    d = np.asarray(d, float)
    if np.allclose(d, d[0]):
        return float(d[0]), float(d[0])
    r = stats.bootstrap((d,), np.mean, n_resamples=n, method="BCa", random_state=np.random.default_rng(semilla))
    return float(r.confidence_interval.low), float(r.confidence_interval.high)


def holm(p):
    """p ajustados de Holm-Bonferroni."""
    p = np.asarray(p, float)
    orden = np.argsort(p)
    m = len(p)
    aj = np.empty(m)
    actual = 0.0
    for i, o in enumerate(orden):
        actual = max(actual, min(1.0, (m - i) * p[o]))
        aj[o] = actual
    return aj


def mcnemar_exacto(acierta_a, acierta_b):
    """McNemar exacto (binomial) sobre los pares discordantes."""
    a_si_b_no = int(np.sum(acierta_a & ~acierta_b))
    a_no_b_si = int(np.sum(~acierta_a & acierta_b))
    n = a_si_b_no + a_no_b_si
    if n == 0:
        return 0.0, 1.0
    return float(min(a_si_b_no, a_no_b_si)), float(stats.binomtest(a_si_b_no, n, 0.5).pvalue)


def delong(y, p1, p2):
    """DeLong, DeLong y Clarke-Pearson (1988) para dos AUC correlacionadas (algoritmo por componentes)."""
    y = np.asarray(y)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]

    def comp(p):
        rx = stats.rankdata(p[pos])
        ry = stats.rankdata(p[neg])
        rz = stats.rankdata(np.r_[p[pos], p[neg]])
        v10 = (rz[:len(pos)] - rx) / len(neg)
        v01 = 1 - (rz[len(pos):] - ry) / len(pos)
        return v10.mean(), v10, v01

    a1, v1, w1 = comp(np.asarray(p1))
    a2, v2, w2 = comp(np.asarray(p2))
    s10 = np.cov(np.vstack([v1, v2]))
    s01 = np.cov(np.vstack([w1, w2]))
    S = s10 / len(pos) + s01 / len(neg)
    var = S[0, 0] + S[1, 1] - 2 * S[0, 1]
    if var <= 0:
        return 0.0, 1.0, a1, a2
    z = (a1 - a2) / np.sqrt(var)
    return float(z), float(2 * stats.norm.sf(abs(z))), float(a1), float(a2)


def bayes_rangos_signo(d, rope=0.001, n=20_000, semilla=0):
    """Prueba bayesiana de rangos con signo (Benavoli et al., 2017), con pseudo-observación en 0.

    d = pérdida(original) - pérdida(rival). Devuelve P(original mejor), P(equivalentes), P(rival mejor).
    """
    d = np.r_[0.0, np.asarray(d, float)]
    rng = np.random.default_rng(semilla)
    i, j = np.triu_indices(len(d))
    suma = d[i] + d[j]
    gan, igual, per = suma < -2 * rope, np.abs(suma) <= 2 * rope, suma > 2 * rope
    w = rng.dirichlet(np.ones(len(d)), n)
    pesos = w[:, i] * w[:, j]
    pesos[:, i != j] *= 2
    theta = np.column_stack([pesos[:, gan].sum(1), pesos[:, igual].sum(1), pesos[:, per].sum(1)])
    gana = np.bincount(theta.argmax(1), minlength=3) / n
    return tuple(float(x) for x in gana)
