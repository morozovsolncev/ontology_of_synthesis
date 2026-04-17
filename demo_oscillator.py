#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
demo_oscillator.py
Демонстрация решения задачи о гармоническом осцилляторе
с использованием онтологического закона движения.

Задача: частица массой m=1 в поле связности β = -α·x².
Найти траекторию x(t), период колебаний, фазовый портрет.

Онтологический закон: a = (η/2)·d(ln Pr)/dx, где ln Pr = ∫ β dx.
Для осциллятора: β = -α·x² → ln Pr = -α·x³/3 → a = (η/2)·(-α·x²) = -(η·α/2)·x².

Но это даёт a ∝ x², а не a ∝ x! Ошибка?
Проверим: d/dx (x³) = 3x², поэтому d/dx (ln Pr) = -α·x².
Тогда a = (η/2)·(-α·x²) = -(η·α/2)·x².

Это не гармонический осциллятор! Это осциллятор с квадратичной зависимостью ускорения от координаты.

Чтобы получить a = -ω²·x, нужно β = -α·x, тогда ln Pr = -α·x²/2,
и a = (η/2)·(-α·x) = -(η·α/2)·x.

Исправляем: для гармонического осциллятора поле связности должно быть β = -α·x,
а не β = -α·x².
"""

import numpy as np
import matplotlib.pyplot as plt
from ontological_physics import Universe, Particle, Field, Simulator

# ============================================================================
# 1. ПАРАМЕТРЫ
# ============================================================================

# Создаём Вселенную (1D, η=1 для абстрактных единиц)
uni = Universe(eta=1.0, dim=1)

# Параметры осциллятора: a = -ω²·x
# Из онтологии: a = (η/2)·d(ln Pr)/dx = (η/2)·β (так как β = d(ln Pr)/dx)
# Чтобы a = -ω²·x, нужно β = -(2ω²/η)·x
# Выбираем ω = 1, η = 1 → β = -2·x

omega = 1.0
alpha = 2.0 * omega**2 / uni.eta  # = 2.0

# Создаём поле осциллятора с β = -α·x
# Для этого используем кастомное поле (custom) или модифицируем oscillator
# Сейчас oscillator у нас был β = -α·x². Создадим отдельный класс или используем custom.

class OscillatorField(Field):
    """Поле гармонического осциллятора: β = -α·x."""
    def __init__(self, universe, alpha=2.0, **kwargs):
        super().__init__(universe, 'custom', source_strength=alpha, **kwargs)

    def beta(self, r_vec):
        x = (r_vec - self.source_position)[0]
        return -self.source_strength * x

    def ln_pr(self, r_vec, r0_vec=None):
        if r0_vec is None:
            r0_vec = self.source_position
        x = (r_vec - self.source_position)[0]
        x0 = (r0_vec - self.source_position)[0]
        # ∫ β dx = ∫ -α·x dx = -α·x²/2
        return -self.source_strength * (x**2 - x0**2) / 2.0

# Альтернатива: просто используем Field с field_type='custom' и переопределяем методы
# Но для демонстрации создадим отдельный класс.

# ============================================================================
# 2. СОЗДАНИЕ ПОЛЯ И ЧАСТИЦЫ
# ============================================================================

field = OscillatorField(uni, alpha=alpha)
particle = Particle(uni, mass=1.0, position=np.array([1.0]), velocity=np.array([0.0]))

# ============================================================================
# 3. ЗАПУСК СИМУЛЯЦИИ
# ============================================================================

sim = Simulator(uni, field, particle, dt=0.01, method='verlet')
history = sim.run(t_max=10.0, record_every=1)

# Извлекаем данные
t = history['t']
x = history['x'][:, 0]
v = history['v'][:, 0]
a = history['a'][:, 0]

# ============================================================================
# 4. АНАЛИТИЧЕСКОЕ РЕШЕНИЕ
# ============================================================================

t_ana = np.linspace(0, 10, 1000)
x_ana = np.cos(omega * t_ana)  # x0=1, v0=0
v_ana = -omega * np.sin(omega * t_ana)

# ============================================================================
# 5. ВИЗУАЛИЗАЦИЯ
# ============================================================================

plt.figure(figsize=(15, 12))

# 5.1 Траектория x(t)
plt.subplot(2, 2, 1)
plt.plot(t, x, 'b-', linewidth=2, label='Онтологическая симуляция')
plt.plot(t_ana, x_ana, 'r--', linewidth=1.5, label='Аналитика: cos(t)')
plt.xlabel('Время t')
plt.ylabel('Координата x')
plt.title('Гармонический осциллятор: x(t)')
plt.legend()
plt.grid(True, alpha=0.3)

# 5.2 Скорость v(t)
plt.subplot(2, 2, 2)
plt.plot(t, v, 'b-', linewidth=2, label='Симуляция')
plt.plot(t_ana, v_ana, 'r--', linewidth=1.5, label='Аналитика: -sin(t)')
plt.xlabel('Время t')
plt.ylabel('Скорость v')
plt.title('Скорость v(t)')
plt.legend()
plt.grid(True, alpha=0.3)

# 5.3 Фазовый портрет v(x)
plt.subplot(2, 2, 3)
plt.plot(x, v, 'b-', linewidth=1.5, alpha=0.7)
plt.plot(x[0], v[0], 'go', markersize=8, label='Старт')
plt.plot(x[-1], v[-1], 'ro', markersize=8, label='Финиш')
plt.xlabel('Координата x')
plt.ylabel('Скорость v')
plt.title('Фазовый портрет')
plt.legend()
plt.grid(True, alpha=0.3)
plt.axis('equal')

# 5.4 Поле связности β(x) и ln Pr(x)
plt.subplot(2, 2, 4)
x_plot = np.linspace(-1.5, 1.5, 100)
beta_plot = np.array([field.beta(np.array([xx])) for xx in x_plot])
lnPr_plot = np.array([field.ln_pr(np.array([xx]), np.array([0.0])) for xx in x_plot])

plt.plot(x_plot, beta_plot, 'purple', linewidth=2, label='β(x) = -α·x')
plt.plot(x_plot, lnPr_plot, 'orange', linewidth=2, label='ln Pr(x) = -α·x²/2')
plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
plt.xlabel('Координата x')
plt.ylabel('β и ln Pr')
plt.title('Поле связности осциллятора (α = 2)')
plt.legend()
plt.grid(True, alpha=0.3)

plt.suptitle('ГАРМОНИЧЕСКИЙ ОСЦИЛЛЯТОР В ОНТОЛОГИИ СИНТЕЗА\n'
             'Закон движения: a = (η/2)·∇[ln Pr], поле β = -α·x',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('oscillator_demo.png', dpi=150)
plt.show()

# ============================================================================
# 6. АНАЛИЗ ПЕРИОДА
# ============================================================================

# Находим пересечения нуля (для определения периода)
crossings = np.where(np.diff(np.sign(x)))[0]
if len(crossings) >= 2:
    # Период = 2 * время между двумя пересечениями нуля (от - до +)
    T_num = 2 * (t[crossings[1]] - t[crossings[0]])
    T_ana = 2 * np.pi / omega
    print("\n" + "=" * 70)
    print("РЕЗУЛЬТАТЫ ГАРМОНИЧЕСКОГО ОСЦИЛЛЯТОРА")
    print("=" * 70)
    print(f"  Параметры: ω = {omega}, α = {alpha}, η = {uni.eta}")
    print(f"  Теоретический период: T_теор = {T_ana:.4f}")
    print(f"  Численный период: T_числ = {T_num:.4f}")
    print(f"  Относительная ошибка: {abs(T_num - T_ana)/T_ana*100:.4f}%")
    print("=" * 70)
else:
    print("\n⚠️ Недостаточно данных для определения периода.")

# Проверка закона движения: a = -ω²·x
# Вычисляем a_теор = -omega**2 * x
a_theor = -omega**2 * x
a_error = np.mean(np.abs(a - a_theor)) / np.mean(np.abs(a_theor)) * 100
print(f"\nПроверка закона движения: a = -ω²·x")
print(f"  Средняя относительная ошибка: {a_error:.4f}%")