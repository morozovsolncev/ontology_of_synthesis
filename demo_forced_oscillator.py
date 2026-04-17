#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
demo_forced_oscillator.py
ВЫНУЖДЕННЫЕ КОЛЕБАНИЯ И РЕЗОНАНС В ОНТОЛОГИИ СИНТЕЗА

Демонстрация вынужденных колебаний с полем связности
β(x, v, t) = -α·x - δ·v + F0·cos(ω_d·t)

При совпадении частоты вынуждающей силы ω_d с собственной частотой ω₀
возникает резонанс — резкий рост амплитуды колебаний.

Запуск в Google Colab:
    %run demo_forced_oscillator.py
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from ontological_physics import Universe

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

ДЛЯ ВЫНУЖДЕННОГО ОСЦИЛЛЯТОРА:
   β(x, v, t) = -α·x - δ·v + F0·cos(ω_d·t)
   a = -(η·α/2)·x - (η·δ/2)·v + (η·F0/2)·cos(ω_d·t)
""")
print("=" * 70)

# ============================================================================
# 1. ПАРАМЕТРЫ
# ============================================================================

print("\n" + "=" * 70)
print("ЗАДАЧА: ВЫНУЖДЕННЫЙ ОСЦИЛЛЯТОР (РЕЗОНАНС)")
print("=" * 70)

class ForcedOscillator:
    def __init__(self, eta=1.0, alpha=2.0, delta=0.2, F0=0.5, omega_d=1.0):
        self.eta = eta
        self.alpha = alpha
        self.delta = delta
        self.F0 = F0
        self.omega_d = omega_d
        self.omega0_sq = eta * alpha / 2.0
        self.gamma = eta * delta / 2.0

    def acceleration(self, x, v, t):
        """a = -ω₀²·x - γ·v + (η·F0/2)·cos(ω_d·t)"""
        force_term = (self.eta * self.F0 / 2.0) * np.cos(self.omega_d * t)
        return -self.omega0_sq * x - self.gamma * v + force_term

    def stationary_amplitude(self):
        """Стационарная амплитуда вынужденных колебаний (теория)."""
        ω0 = np.sqrt(self.omega0_sq)
        γ = self.gamma
        ω = self.omega_d
        A0 = (self.eta * self.F0 / 2.0) / np.sqrt((ω0**2 - ω**2)**2 + (γ * ω)**2)
        return A0


# ============================================================================
# 2. СИМУЛЯЦИЯ
# ============================================================================

def simulate(oscillator, x0, v0, t_max, dt=0.01):
    """Метод Верле для вынужденного осциллятора."""
    n_steps = int(t_max / dt)
    t = np.zeros(n_steps)
    x = np.zeros(n_steps)
    v = np.zeros(n_steps)

    t[0] = 0
    x[0] = x0
    v[0] = v0

    for i in range(n_steps - 1):
        a = oscillator.acceleration(x[i], v[i], t[i])
        x_half = x[i] + v[i] * dt/2
        v_half = v[i] + a * dt/2
        a_half = oscillator.acceleration(x_half, v_half, t[i] + dt/2)
        x[i+1] = x[i] + v_half * dt
        v[i+1] = v_half + a_half * dt/2
        t[i+1] = t[i] + dt

    return t, x, v


def find_steady_state_amplitude(t, x, t_start=None):
    """Находит амплитуду установившихся колебаний."""
    if t_start is None:
        t_start = t[-1] * 0.7  # последние 30% времени

    mask = t > t_start
    t_steady = t[mask]
    x_steady = x[mask]

    peaks, _ = find_peaks(x_steady, height=0.02, distance=50)
    if len(peaks) < 3:
        return np.nan, np.nan

    peak_values = x_steady[peaks]
    amplitude = np.mean(peak_values[-10:]) if len(peak_values) > 10 else np.mean(peak_values)

    return amplitude, peak_values


# ============================================================================
# 3. ЭКСПЕРИМЕНТ 1: ЗАВИСИМОСТЬ АМПЛИТУДЫ ОТ ЧАСТОТЫ (РЕЗОНАНСНАЯ КРИВАЯ)
# ============================================================================

print("\n" + "=" * 70)
print("ЭКСПЕРИМЕНТ 1: Резонансная кривая")
print("=" * 70)

eta = 1.0
alpha = 2.0
delta = 0.2
F0 = 0.5
x0 = 0.0
v0 = 0.0
t_max = 100.0
dt = 0.01

omega0 = np.sqrt(eta * alpha / 2.0)
print(f"\n  Параметры системы:")
print(f"    η = {eta}, α = {alpha}")
print(f"    ω₀ = {omega0:.3f}")
print(f"    δ = {delta}, λ = {eta*delta/4:.3f}")
print(f"    F0 = {F0}")
print(f"    t_max = {t_max}")

# Сканируем частоты
omega_d_values = np.linspace(0.5, 1.5, 21)
amplitudes_num = []
amplitudes_theor = []

print("\n  Сканирование частот...")
for omega_d in omega_d_values:
    osc = ForcedOscillator(eta=eta, alpha=alpha, delta=delta, F0=F0, omega_d=omega_d)
    t, x, v = simulate(osc, x0, v0, t_max, dt)

    # Находим амплитуду в установившемся режиме
    A_num, _ = find_steady_state_amplitude(t, x, t_start=t_max * 0.7)
    A_theor = osc.stationary_amplitude()

    amplitudes_num.append(A_num)
    amplitudes_theor.append(A_theor)

    if omega_d in [0.8, 0.9, 1.0, 1.1, 1.2]:
        print(f"    ω_d = {omega_d:.2f}: A_числ = {A_num:.4f}, A_теор = {A_theor:.4f}")

# ============================================================================
# 4. ЭКСПЕРИМЕНТ 2: ПЕРЕХОДНЫЕ ПРОЦЕССЫ ПРИ РАЗНЫХ ЧАСТОТАХ
# ============================================================================

print("\n" + "=" * 70)
print("ЭКСПЕРИМЕНТ 2: Переходные процессы")
print("=" * 70)

test_frequencies = [0.8, 0.95, 1.0, 1.05, 1.2]
t_max_transient = 80.0

transient_results = {}
for omega_d in test_frequencies:
    osc = ForcedOscillator(eta=eta, alpha=alpha, delta=delta, F0=F0, omega_d=omega_d)
    t, x, v = simulate(osc, x0, v0, t_max_transient, dt)
    transient_results[omega_d] = {'t': t, 'x': x, 'v': v}

# ============================================================================
# 5. ВИЗУАЛИЗАЦИЯ
# ============================================================================

plt.figure(figsize=(16, 12))

# 5.1 Резонансная кривая
plt.subplot(2, 2, 1)
plt.plot(omega_d_values, amplitudes_theor, 'r-', linewidth=2, label='Теория')
plt.plot(omega_d_values, amplitudes_num, 'bo', markersize=6, label='Численный расчёт')
plt.axvline(x=omega0, color='k', linestyle='--', alpha=0.5, label=f'ω₀ = {omega0:.3f}')
plt.xlabel('Частота вынуждающей силы ω_d')
plt.ylabel('Амплитуда A')
plt.title('Резонансная кривая (δ=0.2, F0=0.5)')
plt.legend()
plt.grid(True, alpha=0.3)

# 5.2 Переходные процессы (наложение)
plt.subplot(2, 2, 2)
for omega_d in test_frequencies[:3]:  # 0.8, 0.95, 1.0
    t = transient_results[omega_d]['t']
    x = transient_results[omega_d]['x']
    plt.plot(t, x, linewidth=1.5, alpha=0.8, label=f'ω_d = {omega_d}')
plt.xlabel('Время t')
plt.ylabel('Координата x')
plt.title('Переходные процессы (биения и выход на режим)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xlim(0, 50)

# 5.3 Установившийся режим (сравнение фаз)
plt.subplot(2, 2, 3)
for omega_d in [0.8, 1.0, 1.2]:
    t = transient_results[omega_d]['t']
    x = transient_results[omega_d]['x']
    mask = t > t_max_transient * 0.7
    t_steady = t[mask]
    x_steady = x[mask]
    plt.plot(t_steady - t_steady[0], x_steady, linewidth=1.5, label=f'ω_d = {omega_d}')
plt.xlabel('Время (установившийся режим)')
plt.ylabel('Координата x')
plt.title('Установившиеся колебания (сравнение фаз)')
plt.legend()
plt.grid(True, alpha=0.3)

# 5.4 Фазовые портреты (установившийся режим)
plt.subplot(2, 2, 4)
for omega_d, color in zip([0.8, 1.0, 1.2], ['blue', 'red', 'green']):
    t = transient_results[omega_d]['t']
    x = transient_results[omega_d]['x']
    v = transient_results[omega_d]['v']
    mask = t > t_max_transient * 0.7
    x_steady = x[mask]
    v_steady = v[mask]
    # Берём каждую 10-ю точку для разреженности
    plt.plot(x_steady[::10], v_steady[::10], '.', color=color, markersize=1, alpha=0.5, label=f'ω_d = {omega_d}')
plt.xlabel('x')
plt.ylabel('v')
plt.title('Фазовые портреты (предельные циклы)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.axis('equal')

plt.suptitle('ВЫНУЖДЕННЫЙ ОСЦИЛЛЯТОР В ОНТОЛОГИИ СИНТЕЗА\n'
             'β(x, v, t) = -α·x - δ·v + F0·cos(ω_d·t)',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

# ============================================================================
# 6. ДОПОЛНИТЕЛЬНЫЙ ГРАФИК: ЭВОЛЮЦИЯ АМПЛИТУДЫ ВО ВРЕМЕНИ (БИЕНИЯ)
# ============================================================================

plt.figure(figsize=(15, 5))

omega_near = 0.95
osc_near = ForcedOscillator(eta=eta, alpha=alpha, delta=delta, F0=F0, omega_d=omega_near)
t, x, v = simulate(osc_near, x0, v0, t_max=100.0, dt=0.01)

# Находим огибающую биений
peaks, _ = find_peaks(np.abs(x), height=0.02, distance=50)
peak_times = t[peaks]
peak_values = np.abs(x[peaks])

plt.subplot(1, 2, 1)
plt.plot(t, x, 'b-', linewidth=1, alpha=0.7)
plt.plot(peak_times, peak_values, 'ro', markersize=3, alpha=0.5)
plt.xlabel('Время t')
plt.ylabel('Координата x')
plt.title(f'Биения при ω_d = {omega_near} (близко к резонансу)')
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.semilogy(peak_times, peak_values, 'ro', markersize=4, alpha=0.7)
plt.xlabel('Время t')
plt.ylabel('Амплитуда биений (лог. шкала)')
plt.title('Нарастание амплитуды при приближении к резонансу')
plt.grid(True, alpha=0.3)

plt.suptitle('БИЕНИЯ И НАРАСТАНИЕ АМПЛИТУДЫ ПРИ РЕЗОНАНСЕ', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

# ============================================================================
# 7. ВЫВОДЫ
# ============================================================================

print("\n" + "=" * 70)
print("ВЫВОДЫ ПО ВЫНУЖДЕННОМУ ОСЦИЛЛЯТОРУ")
print("=" * 70)
print(f"""
1.  РЕЗОНАНСНАЯ КРИВАЯ
    - Максимум амплитуды достигается при ω_d ≈ ω₀ = {omega0:.3f}
    - Форма кривой описывается теорией вынужденных колебаний

2.  ПЕРЕХОДНЫЕ ПРОЦЕССЫ
    - При ω_d ≠ ω₀: биения с частотой |ω_d - ω₀|
    - При ω_d = ω₀: монотонное нарастание амплитуды до стационарного значения

3.  УСТАНОВИВШИЙСЯ РЕЖИМ
    - Частота колебаний равна частоте вынуждающей силы ω_d
    - Фазовый сдвиг зависит от расстройки Δω = ω_d - ω₀

4.  ФАЗОВЫЕ ПОРТРЕТЫ
    - При ω_d = ω₀: эллипс (предельный цикл)
    - При ω_d ≠ ω₀: более сложная форма (фигуры Лиссажу)

5.  ОНТОЛОГИЧЕСКАЯ ИНТЕРПРЕТАЦИЯ
    - Внешнее периодическое поле F0·cos(ω_d·t) — это модуляция связности
    - Резонанс — максимальная передача энергии от внешнего поля к системе
    - Биения — интерференция собственных и вынужденных колебаний
""")
print("=" * 70)