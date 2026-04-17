#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
demo_coupled_oscillators.py
СВЯЗАННЫЕ ОСЦИЛЛЯТОРЫ И ВОЛНЫ В ОНТОЛОГИИ СИНТЕЗА

Демонстрация системы из двух связанных осцилляторов и
распространения волн в цепочке осцилляторов.

Поле связности для каждого осциллятора:
   β_i = -α·x_i - δ·v_i + β_coupling
   β_coupling = ε·(x_{i+1} - x_i) + ε·(x_{i-1} - x_i)

Закон движения: a_i = (η/2)·β_i

Запуск в Google Colab:
    %run demo_coupled_oscillators.py
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

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

ДЛЯ СВЯЗАННЫХ ОСЦИЛЛЯТОРОВ:
   β_i = -α·x_i - δ·v_i + ε·(x_{i+1} - x_i) + ε·(x_{i-1} - x_i)
   a_i = -(η·α/2)·x_i - (η·δ/2)·v_i + (η·ε/2)·(x_{i+1} - 2x_i + x_{i-1})
""")
print("=" * 70)

# ============================================================================
# 1. ПАРАМЕТРЫ
# ============================================================================

print("\n" + "=" * 70)
print("ЗАДАЧА: СВЯЗАННЫЕ ОСЦИЛЛЯТОРЫ И ВОЛНЫ")
print("=" * 70)

class CoupledOscillators:
    def __init__(self, n_osc=50, eta=1.0, alpha=2.0, delta=0.1, epsilon=0.5):
        self.n_osc = n_osc
        self.eta = eta
        self.alpha = alpha
        self.delta = delta
        self.epsilon = epsilon

        self.omega0_sq = eta * alpha / 2.0
        self.gamma = eta * delta / 2.0
        self.coupling = eta * epsilon / 2.0

    def acceleration(self, x, v, i):
        """Ускорение i-го осциллятора."""
        # Собственная сила
        a_self = -self.omega0_sq * x[i] - self.gamma * v[i]

        # Связь с соседями
        a_coupling = 0.0
        if i > 0:
            a_coupling += self.coupling * (x[i-1] - x[i])
        if i < self.n_osc - 1:
            a_coupling += self.coupling * (x[i+1] - x[i])

        return a_self + a_coupling


def simulate_coupled(oscillator, x0, v0, t_max, dt=0.01):
    """Симуляция системы связанных осцилляторов."""
    n_osc = oscillator.n_osc
    n_steps = int(t_max / dt)

    t = np.zeros(n_steps)
    x = np.zeros((n_steps, n_osc))
    v = np.zeros((n_steps, n_osc))

    t[0] = 0
    x[0] = x0.copy()
    v[0] = v0.copy()

    for step in range(n_steps - 1):
        # Полушаг для скорости
        a_half = np.array([oscillator.acceleration(x[step], v[step], i) for i in range(n_osc)])
        v_half = v[step] + a_half * dt/2
        x_half = x[step] + v_half * dt/2

        # Полный шаг
        a_full = np.array([oscillator.acceleration(x_half, v_half, i) for i in range(n_osc)])
        x[step+1] = x[step] + v_half * dt
        v[step+1] = v_half + a_full * dt/2
        t[step+1] = t[step] + dt

    return t, x, v


# ============================================================================
# 2. ЭКСПЕРИМЕНТ 1: ДВА СВЯЗАННЫХ ОСЦИЛЛЯТОРА
# ============================================================================

print("\n" + "=" * 70)
print("ЭКСПЕРИМЕНТ 1: Два связанных осциллятора")
print("=" * 70)

n_osc = 2
eta = 1.0
alpha = 2.0
delta = 0.05
epsilon = 0.3

osc = CoupledOscillators(n_osc=n_osc, eta=eta, alpha=alpha, delta=delta, epsilon=epsilon)

# Начальные условия: первый отклонён, второй в покое
x0 = np.array([1.0, 0.0])
v0 = np.array([0.0, 0.0])

t_max = 40.0
dt = 0.01

t, x, v = simulate_coupled(osc, x0, v0, t_max, dt)

print(f"\n  Параметры:")
print(f"    η = {eta}, α = {alpha}, ω₀ = {np.sqrt(eta*alpha/2):.3f}")
print(f"    δ = {delta}, ε = {epsilon}")
print(f"    Начальные условия: x₁=1.0, x₂=0.0")

# Находим частоты биений
peaks1, _ = find_peaks(x[:, 0], height=0.1)
peaks2, _ = find_peaks(x[:, 1], height=0.1)

if len(peaks1) > 4 and len(peaks2) > 4:
    T_beat = np.mean(np.diff(t[peaks1][-10:]))
    ω_beat = 2 * np.pi / T_beat
    print(f"    Период биений: {T_beat:.3f}, частота: {ω_beat:.3f}")
    print(f"    Теория: ω_beat = √(ε) = {np.sqrt(epsilon):.3f}")

# ============================================================================
# 3. ЭКСПЕРИМЕНТ 2: ЦЕПОЧКА ОСЦИЛЛЯТОРОВ (ВОЛНОВОЙ ПАКЕТ)
# ============================================================================

print("\n" + "=" * 70)
print("ЭКСПЕРИМЕНТ 2: Цепочка осцилляторов (волновой пакет)")
print("=" * 70)

n_osc = 50
eta = 1.0
alpha = 2.0
delta = 0.02
epsilon = 0.5

osc2 = CoupledOscillators(n_osc=n_osc, eta=eta, alpha=alpha, delta=delta, epsilon=epsilon)

# Волновой пакет: гауссов импульс в центре
x_center = n_osc // 2
width = 5
x0_wave = np.exp(-((np.arange(n_osc) - x_center)**2) / (2 * width**2))
v0_wave = np.zeros(n_osc)

t_max_wave = 100.0
dt = 0.02

t_wave, x_wave, v_wave = simulate_coupled(osc2, x0_wave, v0_wave, t_max_wave, dt)

print(f"\n  Параметры:")
print(f"    n = {n_osc}")
print(f"    ε = {epsilon}")
print(f"    Гауссов пакет в центре (σ = {width})")

# Скорость распространения
# Теоретическая групповая скорость для дискретной цепочки
v_phase = np.sqrt(epsilon)  # фазовая скорость в длинноволновом пределе
print(f"    Теоретическая фазовая скорость: {v_phase:.3f}")

# ============================================================================
# 4. ВИЗУАЛИЗАЦИЯ
# ============================================================================

plt.figure(figsize=(16, 14))

# 4.1 Два связанных осциллятора (траектории)
plt.subplot(2, 2, 1)
plt.plot(t, x[:, 0], 'b-', linewidth=1.5, alpha=0.8, label='Осциллятор 1')
plt.plot(t, x[:, 1], 'r-', linewidth=1.5, alpha=0.8, label='Осциллятор 2')
plt.xlabel('Время t')
plt.ylabel('Координата x')
plt.title('Два связанных осциллятора (обмен энергией)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xlim(0, 40)

# 4.2 Два связанных осциллятора (фазовый портрет)
plt.subplot(2, 2, 2)
plt.plot(x[:, 0], x[:, 1], 'purple', linewidth=1, alpha=0.7)
plt.plot(x[0, 0], x[0, 1], 'go', markersize=8, label='Старт')
plt.plot(x[-1, 0], x[-1, 1], 'ro', markersize=8, label='Финиш')
plt.xlabel('x₁')
plt.ylabel('x₂')
plt.title('Фазовый портрет (обмен энергией → фигура Лиссажу)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.axis('equal')

# 4.3 Волновой пакет (начальный профиль)
plt.subplot(2, 2, 3)
positions = np.arange(n_osc)
plt.plot(positions, x0_wave, 'bo-', linewidth=2, markersize=4, label='t=0')
plt.xlabel('Номер осциллятора')
plt.ylabel('Амплитуда')
plt.title('Начальный волновой пакет (гауссов импульс)')
plt.legend()
plt.grid(True, alpha=0.3)

# 4.4 Волновой пакет (распространение — 2D тепловая карта)
plt.subplot(2, 2, 4)
# Берём каждый 5-й шаг для разрежения
step_stride = 10
t_heat = t_wave[::step_stride]
x_heat = x_wave[::step_stride].T
im = plt.imshow(x_heat, aspect='auto', cmap='seismic', origin='lower',
                extent=[0, t_max_wave, 0, n_osc])
plt.colorbar(im, label='Амплитуда')
plt.xlabel('Время t')
plt.ylabel('Номер осциллятора')
plt.title('Распространение волнового пакета (x-t диаграмма)')
plt.grid(True, alpha=0.3)

plt.suptitle('СВЯЗАННЫЕ ОСЦИЛЛЯТОРЫ В ОНТОЛОГИИ СИНТЕЗА\n'
             'β_i = -α·x_i - δ·v_i + ε·(x_{i+1} - 2x_i + x_{i-1})',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

# ============================================================================
# 5. ДОПОЛНИТЕЛЬНЫЙ ГРАФИК: РАСПРОСТРАНЕНИЕ ВОЛНЫ В РАЗНЫЕ МОМЕНТЫ
# ============================================================================

plt.figure(figsize=(15, 8))

times_to_plot = [0, 10, 20, 30, 40, 50, 60, 70, 80]
colors = plt.cm.viridis(np.linspace(0, 1, len(times_to_plot)))

for time_idx, color in zip(times_to_plot, colors):
    # Находим ближайший шаг
    step_idx = int(time_idx / dt)
    if step_idx < len(t_wave):
        plt.plot(positions, x_wave[step_idx], color=color, linewidth=1.5,
                 label=f't = {time_idx}')

plt.xlabel('Номер осциллятора')
plt.ylabel('Амплитуда')
plt.title('Распространение волнового пакета в разные моменты времени')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, alpha=0.3)

plt.suptitle('ВОЛНОВОЙ ПАКЕТ В ЦЕПОЧКЕ ОСЦИЛЛЯТОРОВ', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

# ============================================================================
# 6. ВЫВОДЫ
# ============================================================================

print("\n" + "=" * 70)
print("ВЫВОДЫ ПО СВЯЗАННЫМ ОСЦИЛЛЯТОРАМ И ВОЛНАМ")
print("=" * 70)
print(f"""
1.  ДВА СВЯЗАННЫХ ОСЦИЛЛЯТОРА
    - Энергия периодически переходит от первого ко второму и обратно
    - Частота обмена энергией (биений) зависит от силы связи ε
    - Фазовый портрет — фигура Лиссажу (эллипс/восьмёрка)

2.  ЦЕПОЧКА ОСЦИЛЛЯТОРОВ
    - Волновой пакет распространяется в обе стороны
    - Форма пакета сохраняется (при малом затухании δ)
    - Скорость распространения: v = √(ε) (в длинноволновом пределе)

3.  ВОЛНОВОЕ УРАВНЕНИЕ (непрерывный предел)
    - При n → ∞, dx → 0 получается волновое уравнение
    - ∂²u/∂t² = c²·∂²u/∂x², где c = √(ε·dx²/dt²)

4.  ОНТОЛОГИЧЕСКАЯ ИНТЕРПРЕТАЦИЯ
    - Связь между осцилляторами — это поле связности, зависящее от соседей
    - Волны — это распространение резонанса по цепочке паттернов
    - Это модель упругой среды, фононов в кристалле, звуковых волн
""")
print("=" * 70)