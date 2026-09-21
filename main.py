import math
import time
import secrets
import random as py_random
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy import stats


# 1. GENERADORES DESDE CERO 

class LCG:
    """Generador de Congruencia Lineal"""
    def __init__(self, seed=12345, a=1664525, c=1013904223, m=2**32):
        self.state = seed
        self.a = a
        self.c = c
        self.m = m

    def random(self):
        self.state = (self.a * self.state + self.c) % self.m
        return self.state / self.m

    def generate(self, n):
        return [self.random() for _ in range(n)]

class MiddleSquare:
    """Método de Cuadrados Medios"""
    def __init__(self, seed=1234, digits=4):
        self.state = seed
        self.digits = digits
        self.m = 10**digits

    def random(self):
        sq_str = str(self.state**2).zfill(2 * self.digits)
        start = (len(sq_str) - self.digits) // 2
        self.state = int(sq_str[start:start + self.digits])
        return self.state / self.m

    def generate(self, n):
        return [self.random() for _ in range(n)]

class MersenneTwister32:
    """Mersenne Twister MT19937 desde cero"""
    def __init__(self, seed=12345):
        self.w, self.n, self.m, self.r = 32, 624, 397, 31
        self.a = 0x9908B0DF
        self.u, self.d = 11, 0xFFFFFFFF
        self.s, self.b = 7, 0x9D2C5680
        self.t, self.c = 15, 0xEFC60000
        self.l = 18
        self.f = 1812433253
        
        self.MT = [0] * self.n
        self.index = self.n
        self.MT[0] = seed
        for i in range(1, self.n):
            self.MT[i] = (self.f * (self.MT[i-1] ^ (self.MT[i-1] >> (self.w - 2))) + i) & 0xFFFFFFFF

    def _twist(self):
        for i in range(self.n):
            x = (self.MT[i] & 0x80000000) + (self.MT[(i + 1) % self.n] & 0x7FFFFFFF)
            xA = x >> 1
            if x % 2 != 0:
                xA ^= self.a
            self.MT[i] = self.MT[(i + self.m) % self.n] ^ xA
        self.index = 0

    def extract_number(self):
        if self.index >= self.n:
            self._twist()
        y = self.MT[self.index]
        y ^= (y >> self.u) & self.d
        y ^= (y << self.s) & self.b
        y ^= (y << self.t) & self.c
        y ^= (y >> self.l)
        self.index += 1
        return (y & 0xFFFFFFFF) / 4294967296.0

    def generate(self, n):
        return [self.extract_number() for _ in range(n)]

class BlumBlumShub:
    """Blum Blum Shub (BBS)"""
    def __init__(self, seed=12345, p=30000000091, q=40000000003):
        self.n = p * q
        self.state = (seed ** 2) % self.n

    def random(self):
        self.state = (self.state ** 2) % self.n
        return self.state / self.n

    def generate(self, n):
        return [self.random() for _ in range(n)]

class RANDU:
    """Generador RANDU"""
    def __init__(self, seed=1):
        self.state = seed
        self.a = 65539
        self.m = 2**31

    def random(self):
        self.state = (self.a * self.state) % self.m
        return self.state / self.m

    def generate(self, n):
        return [self.random() for _ in range(n)]


# 2. PRUEBAS ESTADÍSTICAS Y COMPARACIÓN (INCISOS b, c, d)

def run_hypothesis_tests(data, name):
    """(b) Pruebas de hipótesis: Kolmogorov-Smirnov (Uniformidad) y Autocorrelación Lag-1 (Independencia)"""
    # 1. Prueba KS
    ks_stat, ks_p = stats.kstest(data, 'uniform')
    ks_conclusion = "No se rechaza uniformidad (p > 0.05)." if ks_p > 0.05 else "Se rechaza uniformidad (p <= 0.05)."
    
    # 2. Autocorrelación Lag-1
    n = len(data)
    mean = sum(data) / n
    var = sum((x - mean)**2 for x in data) / n
    cov = sum((data[i] - mean) * (data[i+1] - mean) for i in range(n-1)) / (n - 1)
    autocorr = cov / var if var != 0 else 0
    ac_conclusion = "Muestras independientes sin autocorrelación significativa." if abs(autocorr) < 0.05 else "Existe dependencia serial entre valores consecutivos."

    print(f"\n--- Pruebas de Hipótesis para {name} ---")
    print(f"1. Uniformidad (KS): Estadístico = {ks_stat:.4f}, p-valor = {ks_p:.4e}")
    print(f"   Significado: {ks_conclusion}")
    print(f"2. Independencia (Autocorrelación Lag-1): Coeficiente = {autocorr:.4f}")
    print(f"   Significado: {ac_conclusion}")
    
    return ks_stat, ks_p, autocorr

def compare_with_standard_libraries(n=100000):
    """(d) Comparación de Mersenne Twister propio con random y secrets de Python"""
    print(" (d) COMPARACIÓN: PRNG Propio vs Librerías Estándar (random y secrets)")
    
    # 1. Mersenne Twister propio
    mt_propio = MersenneTwister32(seed=12345)
    t0 = time.time()
    samples_propio = mt_propio.generate(n)
    t_propio = time.time() - t0
    
    # 2. Librería random (Mersenne Twister C-optimizado)
    py_random.seed(12345)
    t0 = time.time()
    samples_random = [py_random.random() for _ in range(n)]
    t_random = time.time() - t0
    
    # 3. Librería secrets (Criptográficamente seguro)
    t0 = time.time()
    samples_secrets = [secrets.randbelow(10**6) / 10**6 for _ in range(n)]
    t_secrets = time.time() - t0
    
    print(f"Tiempo de generación para {n} números:")
    print(f" - MT Propio (Python Puro): {t_propio:.4f} segundos")
    print(f" - random (C-optimizado):    {t_random:.4f} segundos (Speedup: {t_propio/t_random:.2f}x)")
    print(f" - secrets (Seguridad Cripto): {t_secrets:.4f} segundos")
    print("\nConclusión: 'random' es significativamente más rápido por estar compilado en C.")
    print("'secrets' es más lento por requerir entropía del sistema operativo para ser seguro.")

def generate_part1_visuals():
    """Genera (a) Histogramas y (c) Pruebas espectrales 3D"""
    print("\n--> Generando (a) Histogramas de números normalizados a [0, 1]...")
    prngs = {
        'LCG': LCG(),
        'MiddleSquare': MiddleSquare(),
        'MersenneTwister': MersenneTwister32(),
        'BlumBlumShub': BlumBlumShub(),
        'RANDU': RANDU()
    }
    
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    axes = axes.flatten()
    
    for idx, (name, gen) in enumerate(prngs.items()):
        data = gen.generate(10000)
        ks_stat, ks_p, autocorr = run_hypothesis_tests(data, name)
        
        axes[idx].hist(data, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
        axes[idx].set_title(f"{name}\nKS stat={ks_stat:.3f} (p={ks_p:.2f}) | Lag-1 AC={autocorr:.3f}")
        axes[idx].set_xlim(0, 1)
        
    axes[5].axis('off')
    plt.tight_layout()
    plt.savefig("histogramas.png", dpi=300)
    plt.close()

    print("\n--> Generando (c) Pruebas Espectrales 3D (LCG y RANDU)...")
    for name, gen in [('LCG', LCG()), ('RANDU', RANDU())]:
        seq = gen.generate(6000)
        xs, ys, zs = seq[0::3], seq[1::3], seq[2::3]
        
        fig = plt.figure(figsize=(7, 6))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(xs, ys, zs, s=1, c='navy', alpha=0.5)
        ax.set_title(f"Prueba Espectral 3D - {name}")
        ax.set_xlabel("U_i")
        ax.set_ylabel("U_{i+1}")
        ax.set_zlabel("U_{i+2}")
        plt.tight_layout()
        plt.savefig(f"espectral_{name.lower()}.png", dpi=300)
        plt.close()


# 3. MÉTODOS DE MONTE CARLO (PARTE 2 DE LA TAREA)

def monte_carlo_1d(f, a, b, N, prng=None):
    u_samples = prng.generate(N) if prng else [py_random.random() for _ in range(N)]
    samples = [a + (b - a) * u for u in u_samples]
    vals = [f(x) for x in samples]
    
    mean_f = sum(vals) / N
    estimate = (b - a) * mean_f
    var_f = sum((v - mean_f)**2 for v in vals) / (N - 1) if N > 1 else 0
    std_err = (b - a) * math.sqrt(var_f / N) if N > 0 else 0
    
    return estimate, estimate - 1.96 * std_err, estimate + 1.96 * std_err, std_err, var_f

def volume_hyperball_mc(d, N, prng=None):
    count = 0
    if prng is None:
        for _ in range(N):
            point = [py_random.uniform(-1, 1) for _ in range(d)]
            if sum(x**2 for x in point) <= 1.0:
                count += 1
    else:
        u_samples = prng.generate(N * d)
        for i in range(N):
            point = [-1 + 2 * u_samples[i*d + j] for j in range(d)]
            if sum(x**2 for x in point) <= 1.0:
                count += 1
                
    p_hat = count / N
    v_cube = 2**d
    estimate = v_cube * p_hat
    std_err = v_cube * math.sqrt(p_hat * (1 - p_hat) / N) if 0 < p_hat < 1 else 0
    
    return estimate, estimate - 1.96 * std_err, estimate + 1.96 * std_err

if __name__ == "__main__":
    print(" INICIANDO EJECUCIÓN COMPLETA")
    
    # Parte 1: PRNGs, Pruebas y Comparativa
    generate_part1_visuals()
    compare_with_standard_libraries()
    
    # Parte 2: Monte Carlo
    print(" PARTE 2: SIMULACIONES DE MONTE CARLO")
    mt = MersenneTwister32(seed=12345)
    
    # Integrales 1D
    est, low, up, _, _ = monte_carlo_1d(lambda x: math.sin(math.pi * x), 0, 1, 100000, mt)
    print(f"\nIntegral 1D sin(pi*x): {est:.6f} | IC 95%: [{low:.6f}, {up:.6f}]")
    
    # Hiperesferas
    print("\nVolumen de la Hiperesfera V_d:")
    for d in [2, 5, 10, 20]:
        v_est, v_low, v_up = volume_hyperball_mc(d, 100000, mt)
        v_exact = (math.pi**(d/2)) / math.gamma(d/2 + 1)
        print(f" d={d:2d} | Est: {v_est:9.5f} | IC 95%: [{v_low:9.5f}, {v_up:9.5f}] | Real: {v_exact:9.5f}")

    print("\nPipeline ejecutado de principio a fin sin errores.")
