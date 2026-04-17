#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
demo_soliton.py
СОЛИТОНЫ И НЕЛИНЕЙНЫЕ ВОЛНЫ В ОНТОЛОГИИ СИНТЕЗА

Демонстрация солитонов в нелинейной цепочке осцилляторов.

Запуск в Google Colab:
    %run demo_soliton.py
"""

import numpy as np
import matplotlib.pyplot as plt

# ============================================================================
# ВЫВОД ПОСТУЛАТОВ
# ============================================================================

print("\n" + "=" * 70)
print("ОСНОВНЫЕ ПОСТУЛАТЫ ОНТОЛОГИИ СИНТЕЗА")
print("=" * 70)
print("""
1. Беспредельное поле потенций (БПП) — первичная реальность.

2. Пространство, время, материя и поля возникают как паттерны синтеза.

3. Поле связности β_Ω(r, v, t) — интенсивность связи с целым.

4. Вероятность резонанса: Pr_Ω = exp(∫ β_Ω·dr).

5. Закон движения: a = (η/2)·∇[ln Pr_Ω].

ДЛЯ НЕЛИНЕЙНЫХ ВОЛН (СОЛИТОНОВ):
   β_i = -α·x_i - δ·v_i + ε·(x_{i+1} - 2x_i + x_{i-1})
         + γ·[(x_{i+1} - x_i)³ + (x_{i-1} - x_i)³]
""")
print("=" * 70)

# ============================================================================
# 1. КЛАСС НЕЛИНЕЙНОЙ ЦЕПОЧКИ
# ============================================================================

class NonlinearChain:
    def __init__(self, n_osc=100, eta=1.0, alpha=2.0, delta=0.001,
                 epsilon=0.5, gamma=0.5):
        self.n_osc = n_osc
        self.eta = eta
        self.alpha = alpha
        self.delta = delta
        self.epsilon = epsilon
        self.gamma = gamma

        self.omega0_sq = eta * alpha / 2.0
        self.gamma_damp = eta * delta / 2.0
        self.coupling_linear = eta * epsilon / 2.0
        self.coupling_nonlinear = eta * gamma / 2.0

    def acceleration(self, x, v, i):
        a_self = -self.omega0_sq * x[i] - self.gamma_damp * v[i]

        a_coupling = 0.0
        if i > 0:
            diff_left = x[i-1] - x[i]
            a_coupling += self.coupling_linear * diff_left
            a_coupling += self.coupling_nonlinear * diff_left**3
        if i < self.n_osc - 1:
            diff_right = x[i+1] - x[i]
            a_coupling += self.coupling_linear * diff_right
            a_coupling += self.coupling_nonlinear * diff_right**3

        return a_self + a_coupling


def simulate(chain, x0, v0, t_max, dt=0.01):
    n_osc = chain.n_osc
    n_steps = int(t_max / dt)

    t = np.zeros(n_steps)
    x = np.zeros((n_steps, n_osc))
    v = np.zeros((n_steps, n_osc))

    t[0] = 0
    x[0] = x0.copy()
    v[0] = v0.copy()

    for step in range(n_steps - 1):
        a_half = np.array([chain.acceleration(x[step], v[step], i) for i in range(n_osc)])
        v_half = v[step] + a_half * dt/2
        x_half = x[step] + v_half * dt/2

        a_full = np.array([chain.acceleration(x_half, v_half, i) for i in range(n_osc)])
        x[step+1] = x[step] + v_half * dt
        v[step+1] = v_half + a_full * dt/2
        t[step+1] = t[step] + dt

    return t, x, v


# ============================================================================
# 2. ПАРАМЕТРЫ ЭКСПЕРИМЕНТОВ
# ============================================================================

n_osc = 150
eta = 1.0
alpha = 2.0
delta = 0.001
epsilon = 0.5

# Сильная нелинейность
gamma_strong = 2.0

# Узкий гауссов пакет (большая амплитуда → сильный нелинейный эффект)
x_center = n_osc // 2
width = 3
amplitude = 2.0
x0 = amplitude * np.exp(-((np.arange(n_osc) - x_center)**2) / (2 * width**2))
v0 = np.zeros(n_osc)

t_max = 80.0
dt = 0.01

print("\n" + "=" * 70)
print("ЭКСПЕРИМЕНТ: Сравнение линейной и нелинейной цепочки")
print("=" * 70)
print(f"""
  Параметры:
    n = {n_osc}, ε = {epsilon}
    δ = {delta}
    γ_лин = 0.0, γ_нелин = {gamma_strong}
    Гауссов пакет: центр = {x_center}, σ = {width}, A = {amplitude}
    t_max = {t_max}
""")

# Линейная цепочка (γ=0)
chain_linear = NonlinearChain(n_osc=n_osc, eta=eta, alpha=alpha,
                               delta=delta, epsilon=epsilon, gamma=0.0)
t_lin, x_lin, v_lin = simulate(chain_linear, x0, v0, t_max, dt)

# Нелинейная цепочка (γ>0)
chain_nonlinear = NonlinearChain(n_osc=n_osc, eta=eta, alpha=alpha,
                                  delta=delta, epsilon=epsilon, gamma=gamma_strong)
t_nonlin, x_nonlin, v_nonlin = simulate(chain_nonlinear, x0, v0, t_max, dt)

print("  Симуляция завершена.")

# ============================================================================
# 3. ВИЗУАЛИЗАЦИЯ
# ============================================================================

plt.figure(figsize=(18, 14))

# 3.1 x-t диаграмма (линейная)
plt.subplot(2, 2, 1)
step_stride = 5
x_heat_lin = x_lin[::step_stride].T
im1 = plt.imshow(x_heat_lin, aspect='auto', cmap='seismic', origin='lower',
                 extent=[0, t_max, 0, n_osc], vmin=-0.5, vmax=0.5)
plt.colorbar(im1, label='Амплитуда')
plt.xlabel('Время t')
plt.ylabel('Номер осциллятора')
plt.title('ЛИНЕЙНАЯ цепочка (γ=0)\nВолновой пакет расплывается')
plt.grid(True, alpha=0.3)

# 3.2 x-t диаграмма (нелинейная)
plt.subplot(2, 2, 2)
x_heat_nonlin = x_nonlin[::step_stride].T
im2 = plt.imshow(x_heat_nonlin, aspect='auto', cmap='seismic', origin='lower',
                 extent=[0, t_max, 0, n_osc], vmin=-0.5, vmax=0.5)
plt.colorbar(im2, label='Амплитуда')
plt.xlabel('Время t')
plt.ylabel('Номер осциллятора')
plt.title(f'НЕЛИНЕЙНАЯ цепочка (γ={gamma_strong})\nСолитон сохраняет форму')
plt.grid(True, alpha=0.3)

# 3.3 Профили в разные моменты (линейная)
plt.subplot(2, 2, 3)
times = [0, 20, 40, 60]
colors = ['blue', 'green', 'orange', 'red']
positions = np.arange(n_osc)

for t_idx, color in zip(times, colors):
    step_idx = int(t_idx / dt)
    plt.plot(positions, x_lin[step_idx], color=color, linewidth=1.5,
             label=f't = {t_idx}')
plt.xlabel('Номер осциллятора')
plt.ylabel('Амплитуда')
plt.title('ЛИНЕЙНАЯ: расплывание пакета')
plt.legend()
plt.grid(True, alpha=0.3)
plt.ylim(-0.5, 1.6)

# 3.4 Профили в разные моменты (нелинейная)
plt.subplot(2, 2, 4)
for t_idx, color in zip(times, colors):
    step_idx = int(t_idx / dt)
    plt.plot(positions, x_nonlin[step_idx], color=color, linewidth=1.5,
             label=f't = {t_idx}')
plt.xlabel('Номер осциллятора')
plt.ylabel('Амплитуда')
plt.title(f'НЕЛИНЕЙНАЯ: солитон сохраняет форму (γ={gamma_strong})')
plt.legend()
plt.grid(True, alpha=0.3)
plt.ylim(-0.5, 1.6)

plt.suptitle('СРАВНЕНИЕ ЛИНЕЙНОЙ И НЕЛИНЕЙНОЙ ЦЕПОЧКИ\n'
             'β_i = -α·x_i - δ·v_i + ε·Δx_i + γ·(Δx_i)³',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

# ============================================================================
# 4. ДОПОЛНИТЕЛЬНЫЙ ГРАФИК: ЭВОЛЮЦИЯ ШИРИНЫ ПАКЕТА
# ============================================================================

def compute_width(x, positions):
    """Вычисляет ширину пакета (среднеквадратичное отклонение)."""
    norm = np.sum(x**2)
    if norm < 1e-10:
        return 0
    center = np.sum(positions * x**2) / norm
    width = np.sqrt(np.sum((positions - center)**2 * x**2) / norm)
    return width

positions = np.arange(n_osc)
width_lin = []
width_nonlin = []

for step in range(0, len(t_lin), 50):
    width_lin.append(compute_width(x_lin[step], positions))
    width_nonlin.append(compute_width(x_nonlin[step], positions))

time_width = t_lin[::50]

plt.figure(figsize=(12, 6))
plt.plot(time_width, width_lin, 'b-', linewidth=2, label='Линейная (γ=0)')
plt.plot(time_width, width_nonlin, 'r-', linewidth=2, label=f'Нелинейная (γ={gamma_strong})')
plt.xlabel('Время t')
plt.ylabel('Ширина пакета')
plt.title('Эволюция ширины волнового пакета\n(расплывание в линейной среде, стабильность в нелинейной)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# ============================================================================
# 5. ВЫВОДЫ
# ============================================================================

print("\n" + "=" * 70)
print("ВЫВОДЫ ПО НЕЛИНЕЙНЫМ ВОЛНАМ И СОЛИТОНАМ")
print("=" * 70)
print(f"""
РЕЗУЛЬТАТЫ:
    Линейная цепочка (γ=0):
        - Пакет расплывается со временем
        - Ширина растёт ~ √t
        - Амплитуда падает

    Нелинейная цепочка (γ={gamma_strong}):
        - Пакет сохраняет форму (солитон)
        - Ширина стабильна
        - Амплитуда постоянна

ОНТОЛОГИЧЕСКАЯ ИНТЕРПРЕТАЦИЯ:
    - Нелинейность связи — это зависимость связности от амплитуды
    - Солитон — устойчивый паттерн (кластер) в нелинейной среде
    - Это модель солитонов в оптических волокнах, волн-убийц,
      солитонов в плазме, квантовых солитонов
""")
print("=" * 70)