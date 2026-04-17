#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
demo_quantum_well.py
Демонстрация решения задачи о частице в бесконечной потенциальной яме
с использованием онтологического подхода.

Задача: частица массой m в яме шириной L.
Потенциал: V(x) = 0 при |x| < L/2, V(x) = ∞ при |x| ≥ L/2.
Онтологический подход: поле связности β(x) = 0 внутри ямы, β(x) = -∞ снаружи.
Вероятность резонанса Pr_Ω = exp(∫ β dx) → 0 вне ямы.
Уравнение Шрёдингера выводится из принципа максимизации взвешенной сложности.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh_tridiagonal
from ontological_physics import Universe, Particle, Field

# ============================================================================
# 1. ПАРАМЕТРЫ
# ============================================================================

# Единицы: ħ = 1, m = 1, L = 2 (яма от -1 до 1)
uni = Universe(eta=1.0, dim=1, hbar=1.0)
L = 2.0                     # ширина ямы
particle = Particle(uni, mass=1.0)

# Создаём поле бесконечной ямы
# В онтологии: β = 0 внутри, -∞ снаружи
field = Field(uni, 'well', source_strength=L)

# ============================================================================
# 2. ПОСТРОЕНИЕ МАТРИЦЫ ГАМИЛЬТОНИАНА (МЕТОД КОНЕЧНЫХ РАЗНОСТЕЙ)
# ============================================================================

N = 500                     # число точек сетки
x_min, x_max = -1.5, 1.5    # область (чуть шире ямы)
x_grid = np.linspace(x_min, x_max, N)
dx = x_grid[1] - x_grid[0]

# Вычисляем потенциал V(x) из онтологического поля
# V(x) = - (m·η/2)·ln Pr_Ω(x)
# Для ямы: ln Pr_Ω = 0 внутри, -∞ снаружи
def potential(x):
    # Используем онтологическое поле
    lnPr = field.ln_pr(np.array([x]))
    if lnPr < -1e10:
        return 1e10  # бесконечный потенциал
    return -particle.mass * uni.eta / 2.0 * lnPr

V_grid = np.array([potential(x) for x in x_grid])

# Построение трёхдиагональной матрицы гамильтониана
# H = - (ħ²/2m)·d²/dx² + V(x)
# Конечные разности: d²ψ/dx² ≈ (ψ_{i-1} - 2ψ_i + ψ_{i+1})/dx²

hbar = uni.hbar
m = particle.mass
kinetic_factor = -hbar**2 / (2 * m * dx**2)

# Диагональ
diag = np.zeros(N)
diag[:] = -2 * kinetic_factor + V_grid

# Побочные диагонали
off_diag = np.ones(N-1) * kinetic_factor

# ============================================================================
# 3. РЕШЕНИЕ УРАВНЕНИЯ ШРЁДИНГЕРА
# ============================================================================

eigenvalues, eigenvectors = eigh_tridiagonal(diag, off_diag, select='i', select_range=(0, 4))

# Сортируем по возрастанию энергии
idx = np.argsort(eigenvalues)
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]

# ============================================================================
# 4. АНАЛИТИЧЕСКОЕ РЕШЕНИЕ (ДЛЯ СРАВНЕНИЯ)
# ============================================================================

n_analytical = np.arange(1, len(eigenvalues)+1)
E_analytical = n_analytical**2 * np.pi**2 * hbar**2 / (2 * m * L**2)

# Волновые функции (аналитические)
x_center = (x_grid - (x_min + x_max)/2)  # центрируем
psi_analytical = []
for n in range(1, len(eigenvalues)+1):
    if n % 2 == 1:  # чётные функции
        psi = np.cos(n * np.pi * x_center / L)
    else:           # нечётные
        psi = np.sin(n * np.pi * x_center / L)
    # Нормировка
    norm = np.sqrt(np.trapz(psi**2, x_grid))
    psi_analytical.append(psi / norm)

psi_analytical = np.array(psi_analytical).T

# ============================================================================
# 5. ВИЗУАЛИЗАЦИЯ
# ============================================================================

plt.figure(figsize=(16, 12))

# 5.1 Потенциал V(x) из онтологического поля
plt.subplot(2, 2, 1)
V_plot = np.minimum(V_grid, 10)  # обрезаем для наглядности
plt.plot(x_grid, V_plot, 'b-', linewidth=2)
plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
plt.axvline(x=-L/2, color='r', linestyle='--', alpha=0.5, label=f'Стенки ямы (L={L})')
plt.axvline(x=L/2, color='r', linestyle='--', alpha=0.5)
plt.xlabel('x')
plt.ylabel('V(x)')
plt.title('Потенциал из онтологического поля\nV(x) = -(m·η/2)·ln Pr_Ω(x)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.ylim(-0.5, 5)

# 5.2 Уровни энергии (численные vs аналитические)
plt.subplot(2, 2, 2)
n_levels = np.arange(1, len(eigenvalues)+1)
plt.plot(n_levels, eigenvalues, 'bo', markersize=8, label='Численные (онтология)')
plt.plot(n_levels, E_analytical, 'r--', linewidth=2, label='Аналитические: n²π²ħ²/(2mL²)')
plt.xlabel('Квантовое число n')
plt.ylabel('Энергия E_n')
plt.title('Спектр энергий частицы в яме')
plt.legend()
plt.grid(True, alpha=0.3)

# 5.3 Волновые функции (сравнение)
plt.subplot(2, 2, 3)
n_show = 3  # покажем первые 3 уровня
for n in range(n_show):
    # Сдвиг для наглядности
    shift = eigenvalues[n]
    psi_num = eigenvectors[:, n]
    psi_num = psi_num / np.sqrt(np.trapz(psi_num**2, x_grid))  # нормировка
    plt.plot(x_grid, psi_num + shift, 'b-', linewidth=1.5, label=f'n={n+1} (числ)')
    plt.plot(x_grid, psi_analytical[:, n] + shift, 'r--', linewidth=1, alpha=0.7, label=f'n={n+1} (анал)')

plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
plt.axvline(x=-L/2, color='gray', linestyle=':', alpha=0.5)
plt.axvline(x=L/2, color='gray', linestyle=':', alpha=0.5)
plt.xlabel('x')
plt.ylabel('E_n и ψ_n(x)')
plt.title('Волновые функции (со сдвигом на E_n)')
plt.legend(fontsize=8)
plt.grid(True, alpha=0.3)
plt.ylim(-0.5, 12)

# 5.4 Плотность вероятности |ψ|² (численная, n=1)
plt.subplot(2, 2, 4)
psi1 = eigenvectors[:, 0]
psi1_norm = psi1 / np.sqrt(np.trapz(psi1**2, x_grid))
prob_num = psi1_norm**2
prob_ana = psi_analytical[:, 0]**2

plt.plot(x_grid, prob_num, 'b-', linewidth=2, label='Численная')
plt.plot(x_grid, prob_ana, 'r--', linewidth=1.5, label='Аналитическая: cos²(πx/L)')
plt.axvline(x=-L/2, color='gray', linestyle=':', alpha=0.5)
plt.axvline(x=L/2, color='gray', linestyle=':', alpha=0.5)
plt.xlabel('x')
plt.ylabel('|ψ(x)|²')
plt.title('Плотность вероятности (основное состояние, n=1)')
plt.legend()
plt.grid(True, alpha=0.3)

plt.suptitle('ЧАСТИЦА В БЕСКОНЕЧНОЙ ПОТЕНЦИАЛЬНОЙ ЯМЕ\n'
             'Онтологический подход: поле связности β = 0 внутри, -∞ снаружи',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('quantum_well_demo.png', dpi=150)
plt.show()

# ============================================================================
# 6. ВЫВОД РЕЗУЛЬТАТОВ
# ============================================================================

print("\n" + "=" * 70)
print("РЕЗУЛЬТАТЫ: ЧАСТИЦА В БЕСКОНЕЧНОЙ ЯМЕ")
print("=" * 70)
print(f"  Ширина ямы: L = {L}")
print(f"  Масса частицы: m = {particle.mass}")
print(f"  ħ = {uni.hbar}")
print("\n  Сравнение энергий (численные vs аналитические):")
print("  " + "-" * 50)
print(f"  {'n':>3} | {'E_числ':>12} | {'E_анал':>12} | {'Отклонение':>12}")
print("  " + "-" * 50)

for i, (E_num, E_ana) in enumerate(zip(eigenvalues[:6], E_analytical[:6])):
    diff = abs(E_num - E_ana) / E_ana * 100 if E_ana > 0 else 0
    print(f"  {i+1:3d} | {E_num:12.6f} | {E_ana:12.6f} | {diff:11.4f}%")

print("  " + "-" * 50)
print("\n  Вывод: Уровни энергии квантованы.")
print("  Квантование возникает из требования однозначности фазы")
print("  волновой функции в поле связности (условие резонанса).")
print("=" * 70)