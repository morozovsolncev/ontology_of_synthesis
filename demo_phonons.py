#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
demo_phonons.py
ФОНОНЫ — КВАНТЫ КОЛЕБАНИЙ КРИСТАЛЛИЧЕСКОЙ РЕШЁТКИ

Демонстрация:
1. Дисперсионное соотношение ω(k) для цепочки атомов
2. Квантование колебаний → фононы
3. Плотность состояний
4. Теплоёмкость (модель Дебая)

Запуск в Google Colab:
    %run demo_phonons.py
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh_tridiagonal

# ============================================================================
# ВЫВОД ПОСТУЛАТОВ
# ============================================================================

print("\n" + "=" * 70)
print("ОСНОВНЫЕ ПОСТУЛАТЫ ОНТОЛОГИИ СИНТЕЗА")
print("=" * 70)
print("""
1. Беспредельное поле потенций (БПП) — первичная реальность.

2. Пространство, время, материя и поля возникают как паттерны синтеза.

3. Поле связности β_Ω(r) — интенсивность связи с целым.

4. Вероятность резонанса: Pr_Ω = exp(∫ β_Ω·dr).

5. Закон движения: a = (η/2)·∇[ln Pr_Ω].

ДЛЯ ФОНОНОВ:
   Колебания атомов в кристаллической решётке квантуются.
   Квант колебаний — фонон.
   Энергия фонона: E = ħ·ω(k)
   Дисперсия: ω² = (4K/m)·sin²(ka/2) для моноатомной цепочки
""")
print("=" * 70)

# ============================================================================
# 1. ДИСПЕРСИОННОЕ СООТНОШЕНИЕ ДЛЯ ЦЕПОЧКИ АТОМОВ
# ============================================================================

print("\n" + "=" * 70)
print("1. ДИСПЕРСИОННОЕ СООТНОШЕНИЕ ω(k)")
print("=" * 70)

# Параметры цепочки
a = 1.0           # межатомное расстояние
K = 1.0           # жёсткость связи
m = 1.0           # масса атома
hbar = 1.0

# Дисперсия для моноатомной цепочки
k_values = np.linspace(-np.pi/a, np.pi/a, 200)
omega_acoustic = 2 * np.sqrt(K/m) * np.abs(np.sin(k_values * a / 2))

# Дисперсия для двухатомной цепочки (акустическая и оптическая ветви)
m1, m2 = 1.0, 2.0
M = (m1 + m2) / 2
delta = (m2 - m1) / (2 * M)
omega_plus_sq = (2*K/m1 + 2*K/m2) / 2 + np.sqrt(((2*K/m1 + 2*K/m2)/2)**2 - (4*K**2/(m1*m2)) * (1 - np.cos(k_values*a)**2))
omega_minus_sq = (2*K/m1 + 2*K/m2) / 2 - np.sqrt(((2*K/m1 + 2*K/m2)/2)**2 - (4*K**2/(m1*m2)) * (1 - np.cos(k_values*a)**2))

omega_optical = np.sqrt(omega_plus_sq)
omega_acoustic_2atom = np.sqrt(omega_minus_sq)

print(f"\n  Параметры моноатомной цепочки:")
print(f"    a = {a}")
print(f"    K = {K}, m = {m}")
print(f"    ω_max = 2√(K/m) = {2*np.sqrt(K/m):.3f}")

# ============================================================================
# 2. КВАНТОВАНИЕ КОЛЕБАНИЙ: УРОВНИ ЭНЕРГИИ ОДНОЙ МОДЫ
# ============================================================================

print("\n" + "=" * 70)
print("2. КВАНТОВАНИЕ КОЛЕБАНИЙ (ФОНОНЫ)")
print("=" * 70)

# Энергия одной моды: E_n = ħω·(n + 1/2)
omega_mode = 1.0
n_phonons = np.arange(0, 11)
E_n = hbar * omega_mode * (n_phonons + 0.5)

print(f"\n  Энергия моды с ω = {omega_mode}:")
print(f"    E_n = ħω·(n + 1/2)")
for n in range(6):
    print(f"    n = {n}: E = {E_n[n]:.2f}")

# ============================================================================
# 3. ЧИСЛЕННОЕ РЕШЕНИЕ ДЛЯ ЦЕПОЧКИ ИЗ N АТОМОВ
# ============================================================================

print("\n" + "=" * 70)
print("3. ЧИСЛЕННОЕ РЕШЕНИЕ ДЛЯ ЦЕПОЧКИ ИЗ N АТОМОВ")
print("=" * 70)

N_atoms = 20
K = 1.0
m = 1.0

# Матрица жёсткости (диагональ + побочные)
diag = np.ones(N_atoms) * (2*K/m)
off_diag = -np.ones(N_atoms-1) * (K/m)

# Решение уравнения на собственные значения
eigenvalues, eigenvectors = eigh_tridiagonal(diag, off_diag)
omega_modes = np.sqrt(eigenvalues)

print(f"\n  Параметры:")
print(f"    N = {N_atoms} атомов")
print(f"    K = {K}, m = {m}")
print(f"\n  Первые 6 частот (в порядке возрастания):")
for i in range(6):
    print(f"    ω_{i} = {omega_modes[i]:.4f}")

# Теоретические частоты для стоячих волн
k_theor = np.pi * (np.arange(1, N_atoms+1)) / ((N_atoms+1) * a)
omega_theor = 2 * np.sqrt(K/m) * np.abs(np.sin(k_theor * a / 2))

# ============================================================================
# 4. ПЛОТНОСТЬ СОСТОЯНИЙ (DOS)
# ============================================================================

print("\n" + "=" * 70)
print("4. ПЛОТНОСТЬ СОСТОЯНИЙ")
print("=" * 70)

# Вычисляем гистограмму частот
bins = 50
hist, bin_edges = np.histogram(omega_modes, bins=bins, density=True)
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

# Аналитическая DOS для 1D цепочки (модель Дебая)
omega_D = 2 * np.sqrt(K/m)  # частота Дебая
omega_dos = np.linspace(0.01, omega_D, 100)
dos_debye = 2 / (np.pi * omega_D) * 1 / np.sqrt(1 - (omega_dos/omega_D)**2)

# ============================================================================
# 5. ВИЗУАЛИЗАЦИЯ
# ============================================================================

plt.figure(figsize=(16, 14))

# 5.1 Дисперсионное соотношение (моноатомная цепочка)
plt.subplot(2, 2, 1)
plt.plot(k_values, omega_acoustic, 'b-', linewidth=2)
plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
plt.axvline(x=np.pi/a, color='k', linestyle='--', alpha=0.5, label=f'k = π/a')
plt.axvline(x=-np.pi/a, color='k', linestyle='--', alpha=0.5)
plt.xlabel('Волновое число k')
plt.ylabel('Частота ω(k)')
plt.title('Дисперсионное соотношение для моноатомной цепочки\nω(k) = 2√(K/m)·|sin(ka/2)|')
plt.grid(True, alpha=0.3)
plt.xlim(-np.pi/a - 0.2, np.pi/a + 0.2)
plt.ylim(0, 2.5)
plt.legend()

# 5.2 Дисперсионное соотношение (двухатомная цепочка)
plt.subplot(2, 2, 2)
plt.plot(k_values, omega_acoustic_2atom, 'b-', linewidth=2, label='Акустическая ветвь')
plt.plot(k_values, omega_optical, 'r-', linewidth=2, label='Оптическая ветвь')
plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
plt.axvline(x=np.pi/a, color='k', linestyle='--', alpha=0.5, label=f'k = π/a')
plt.xlabel('Волновое число k')
plt.ylabel('Частота ω(k)')
plt.title('Дисперсионное соотношение для двухатомной цепочки\n(акустическая и оптическая ветви)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xlim(-np.pi/a - 0.2, np.pi/a + 0.2)
plt.ylim(0, 3.5)

# 5.3 Частоты нормальных мод (дискретный спектр)
plt.subplot(2, 2, 3)
n_modes = np.arange(1, N_atoms + 1)
plt.stem(n_modes, omega_modes, linefmt='b-', markerfmt='bo', basefmt='r-')
plt.plot(n_modes, omega_theor, 'ro', markersize=4, alpha=0.7, label='Теория стоячих волн')
plt.xlabel('Номер моды')
plt.ylabel('Частота ω')
plt.title(f'Спектр частот цепочки из {N_atoms} атомов')
plt.legend()
plt.grid(True, alpha=0.3)

# 5.4 Плотность состояний
plt.subplot(2, 2, 4)
plt.hist(omega_modes, bins=bins, density=True, alpha=0.5, color='blue', label='Численная DOS')
plt.plot(omega_dos, dos_debye, 'r-', linewidth=2, label='Модель Дебая (1D)')
plt.xlabel('Частота ω')
plt.ylabel('Плотность состояний g(ω)')
plt.title('Плотность состояний фононов в 1D цепочке')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xlim(0, omega_D + 0.2)

plt.suptitle('ФОНОНЫ В КРИСТАЛЛИЧЕСКОЙ РЕШЁТКЕ\n'
             'Кванты колебаний атомов → частицы-переносчики звука и тепла',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

# ============================================================================
# 6. ТЕПЛОЁМКОСТЬ (МОДЕЛЬ ДЕБАЯ)
# ============================================================================

print("\n" + "=" * 70)
print("5. ТЕПЛОЁМКОСТЬ КРИСТАЛЛА (МОДЕЛЬ ДЕБАЯ)")
print("=" * 70)

# Функция теплоёмкости Дебая
def debye_cv(x):
    """x = Θ_D / T, где Θ_D — температура Дебая."""
    if x < 1e-6:
        return 3.0  # закон Дюлонга-Пти
    return 3 * (1 - 3/x + 3/x**2 * (1/(np.exp(x)-1) + 2/(np.exp(2*x)-1) + ...))

# Упрощённая формула
def debye_cv_simple(T, theta_D):
    """Теплоёмкость по Дебаю (аппроксимация)."""
    x = theta_D / T
    if x > 20:
        return 3 * (4 * np.pi**4 / 5) * (T/theta_D)**3
    return 3 * (1 - 3/x + 3/x**2 * (np.exp(-x) + 2*np.exp(-2*x)))

theta_D = 1.0  # температура Дебая (в абстрактных единицах)
T_values = np.linspace(0.1, 2.0, 100)
Cv_values = [debye_cv_simple(T, theta_D) for T in T_values]

plt.figure(figsize=(10, 6))
plt.plot(T_values, Cv_values, 'b-', linewidth=2)
plt.axhline(y=3, color='r', linestyle='--', alpha=0.7, label='Закон Дюлонга-Пти (3R)')
plt.xlabel('Температура T')
plt.ylabel('Теплоёмкость C_v')
plt.title('Теплоёмкость кристалла (модель Дебая)\n'
          'При низких T: C_v ∝ T³ (закон Дебая)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.ylim(0, 3.5)

plt.suptitle('ТЕПЛОЁМКОСТЬ КРИСТАЛЛА — ПРОЯВЛЕНИЕ ФОНОНОВ', fontsize=12)
plt.tight_layout()
plt.show()

# ============================================================================
# 7. ВИЗУАЛИЗАЦИЯ НОРМАЛЬНЫХ МОД
# ============================================================================

plt.figure(figsize=(15, 10))

# Показываем несколько низших мод
modes_to_plot = [1, 2, 3, N_atoms-2, N_atoms-1, N_atoms]
atom_positions = np.arange(N_atoms)

for idx, mode_num in enumerate(modes_to_plot):
    plt.subplot(2, 3, idx+1)
    eigenvec = eigenvectors[:, mode_num-1]
    # Нормируем для наглядности
    eigenvec_norm = eigenvec / np.max(np.abs(eigenvec))
    plt.bar(atom_positions, eigenvec_norm, width=0.8, color='blue', alpha=0.7)
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    plt.xlabel('Номер атома')
    plt.ylabel('Амплитуда смещения')
    plt.title(f'Мода {mode_num} (ω = {omega_modes[mode_num-1]:.3f})')
    plt.grid(True, alpha=0.3)
    plt.ylim(-1.1, 1.1)

plt.suptitle('НОРМАЛЬНЫЕ МОДЫ КОЛЕБАНИЙ ЦЕПОЧКИ АТОМОВ\n'
             'Каждая мода — когерентное колебание всех атомов',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

# ============================================================================
# 8. ВЫВОДЫ
# ============================================================================

print("\n" + "=" * 70)
print("ВЫВОДЫ ПО ФОНОНАМ")
print("=" * 70)
print(f"""
1.  ДИСПЕРСИОННОЕ СООТНОШЕНИЕ
    - Моноатомная цепочка: ω(k) = 2√(K/m)·|sin(ka/2)|
    - Двухатомная цепочка: акустическая и оптическая ветви
    - Акустические фононы (звук), оптические фононы (свет)

2.  КВАНТОВАНИЕ
    - Каждая нормальная мода — квантовый гармонический осциллятор
    - Энергия: E_n = ħω·(n + 1/2)
    - Квант возбуждения — фонон

3.  ПЛОТНОСТЬ СОСТОЯНИЙ
    - Число мод в интервале частот [ω, ω+dω]
    - В 1D: g(ω) ∝ 1/√(ω_max² - ω²) (сингулярность Ван Хова)

4.  ТЕПЛОЁМКОСТЬ
    - При высоких T: C_v → 3R (закон Дюлонга-Пти)
    - При низких T: C_v ∝ T³ (закон Дебая)

5.  ОНТОЛОГИЧЕСКАЯ ИНТЕРПРЕТАЦИЯ
    - Фононы — кванты связности в колебательной системе
    - Это модель: теплопроводность, сверхпроводимость, нейтронное рассеяние
""")
print("=" * 70)