#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
demo_capacitance.py
Демонстрация вычисления ёмкости сферического конденсатора
с использованием онтологического подхода.
"""

import numpy as np
import matplotlib.pyplot as plt
from ontological_physics import Universe, Field

# ============================================================================
# 1. ПАРАМЕТРЫ
# ============================================================================

# Для электромагнетизма η = 2 (чтобы φ = (η/2)·ln Pr = k_EM·Q/r)
uni = Universe(eta=2.0, dim=3)

# Параметры конденсатора
R1 = 1.0
R2 = 2.0
Q = 1.0

# Константа k_EM (в абстрактных единицах k_EM = 1)
k_EM = 1.0
q_test = 1.0

# ============================================================================
# 2. ОНТОЛОГИЧЕСКОЕ ПОЛЕ СФЕРИЧЕСКОГО КОНДЕНСАТОРА
# ============================================================================

class SphericalCapacitorField(Field):
    def __init__(self, universe, R1, R2, Q, k_EM=1.0, q_test=1.0):
        super().__init__(universe, 'custom')
        self.R1 = R1
        self.R2 = R2
        self.Q = Q
        self.k_EM = k_EM
        self.q_test = q_test
        self.source_position = np.zeros(3)

    def beta(self, r_vec):
        r = np.linalg.norm(r_vec - self.source_position)
        if r < self.R1 or r > self.R2:
            return 0.0
        return -self.k_EM * self.Q * self.q_test / r**2

    def ln_pr(self, r_vec, r0_vec=None):
        if r0_vec is None:
            r0_vec = np.array([self.R1, 0.0, 0.0])
        r = np.linalg.norm(r_vec - self.source_position)
        r0 = np.linalg.norm(r0_vec - self.source_position)

        if r < self.R1:
            return 0.0
        if r > self.R2:
            return self.ln_pr(np.array([self.R2, 0.0, 0.0]), r0_vec)

        A = self.k_EM * self.Q * self.q_test
        return A * (1.0/r - 1.0/r0)


# ============================================================================
# 3. ВЫЧИСЛЕНИЕ ПОТЕНЦИАЛА И ЁМКОСТИ
# ============================================================================

field = SphericalCapacitorField(uni, R1, R2, Q, k_EM, q_test)

r1_vec = np.array([R1, 0.0, 0.0])
r2_vec = np.array([R2, 0.0, 0.0])

lnPr1 = field.ln_pr(r1_vec)
lnPr2 = field.ln_pr(r2_vec)

# Потенциал: φ = (η/2)·ln Pr (теперь η/2 = 1)
phi1 = (uni.eta / 2.0) * lnPr1
phi2 = (uni.eta / 2.0) * lnPr2

# Разность потенциалов (модуль, напряжение)
delta_phi = abs(phi2 - phi1)

# Ёмкость C = Q / Δφ
C = Q / delta_phi

# Аналитическая формула: C = 4πε₀·(R1·R2)/(R2-R1)
# ε₀ = 1/(4πk_EM)
epsilon0 = 1.0 / (4.0 * np.pi * k_EM) if k_EM != 0 else 1.0/(4.0*np.pi)
C_analytical = 4.0 * np.pi * epsilon0 * (R1 * R2) / (R2 - R1)

# ============================================================================
# 4. ВИЗУАЛИЗАЦИЯ
# ============================================================================

plt.figure(figsize=(15, 10))

# 4.1 Поле связности β(r)
plt.subplot(2, 2, 1)
r_plot = np.linspace(0.5, 3.0, 200)
beta_plot = [field.beta(np.array([r, 0.0, 0.0])) for r in r_plot]
plt.plot(r_plot, beta_plot, 'b-', linewidth=2)
plt.axvline(x=R1, color='r', linestyle='--', alpha=0.7, label=f'R1 = {R1}')
plt.axvline(x=R2, color='g', linestyle='--', alpha=0.7, label=f'R2 = {R2}')
plt.xlabel('Радиус r')
plt.ylabel('β(r)')
plt.title('Поле связности β(r) = -k_EM·Q·q_test / r²')
plt.legend()
plt.grid(True, alpha=0.3)

# 4.2 Вероятность резонанса Pr_Ω(r)
plt.subplot(2, 2, 2)
lnPr_plot = [field.ln_pr(np.array([r, 0.0, 0.0]), np.array([R1, 0.0, 0.0])) for r in r_plot]
Pr_plot = np.exp(np.minimum(lnPr_plot, 10))
plt.plot(r_plot, Pr_plot, 'purple', linewidth=2)
plt.axvline(x=R1, color='r', linestyle='--', alpha=0.7)
plt.axvline(x=R2, color='g', linestyle='--', alpha=0.7)
plt.xlabel('Радиус r')
plt.ylabel('Pr_Ω(r)')
plt.title('Вероятность резонанса Pr_Ω = exp(∫ β dr)')
plt.grid(True, alpha=0.3)

# 4.3 Потенциал φ(r) = (η/2)·ln Pr_Ω(r)
plt.subplot(2, 2, 3)
phi_plot = (uni.eta / 2.0) * np.array(lnPr_plot)
plt.plot(r_plot, phi_plot, 'orange', linewidth=2)
plt.axvline(x=R1, color='r', linestyle='--', alpha=0.7)
plt.axvline(x=R2, color='g', linestyle='--', alpha=0.7)
plt.axhline(y=phi1, color='r', linestyle=':', alpha=0.5, label=f'φ(R1) = {phi1:.4f}')
plt.axhline(y=phi2, color='g', linestyle=':', alpha=0.5, label=f'φ(R2) = {phi2:.4f}')
plt.xlabel('Радиус r')
plt.ylabel('Потенциал φ(r)')
plt.title('Электростатический потенциал: φ = (η/2)·ln Pr_Ω')
plt.legend()
plt.grid(True, alpha=0.3)

# 4.4 Сравнение ёмкости
plt.subplot(2, 2, 4)
methods = ['Онтология', 'Аналитика']
capacitance = [C, C_analytical]
bars = plt.bar(methods, capacitance, color=['blue', 'red'], alpha=0.7)
plt.ylabel('Ёмкость C')
plt.title(f'Ёмкость сферического конденсатора\n(R1={R1}, R2={R2})')
for bar, val in zip(bars, capacitance):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
             f'{val:.4f}', ha='center', va='bottom')
plt.grid(True, alpha=0.3, axis='y')

plt.suptitle('СФЕРИЧЕСКИЙ КОНДЕНСАТОР В ОНТОЛОГИИ СИНТЕЗА\n'
             'η = 2 для электромагнетизма (φ = (η/2)·ln Pr = k_EM·Q/r)',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('capacitance_demo.png', dpi=150)
plt.show()

# ============================================================================
# 5. ВЫВОД РЕЗУЛЬТАТОВ
# ============================================================================

print("\n" + "=" * 70)
print("РЕЗУЛЬТАТЫ: СФЕРИЧЕСКИЙ КОНДЕНСАТОР")
print("=" * 70)
print(f"\n  Параметры:")
print(f"    R1 = {R1}")
print(f"    R2 = {R2}")
print(f"    Q = {Q}")
print(f"    k_EM = {k_EM}")
print(f"    η = {uni.eta}")
print(f"    ε₀ = 1/(4πk_EM) = {epsilon0:.6f}")

print(f"\n  Онтологические величины:")
print(f"    ln Pr_Ω(R1) = {lnPr1:.6f}")
print(f"    ln Pr_Ω(R2) = {lnPr2:.6f}")
print(f"    Δln Pr = |ln Pr(R2) - ln Pr(R1)| = {abs(lnPr2 - lnPr1):.6f}")

print(f"\n  Потенциалы:")
print(f"    φ(R1) = (η/2)·ln Pr_Ω(R1) = {phi1:.6f}")
print(f"    φ(R2) = (η/2)·ln Pr_Ω(R2) = {phi2:.6f}")
print(f"    Δφ = |φ(R2) - φ(R1)| = {delta_phi:.6f}")

print(f"\n  Ёмкость:")
print(f"    C = Q / Δφ = {C:.6f}")
print(f"    C_аналитическая = 4πε₀·(R1·R2)/(R2-R1) = {C_analytical:.6f}")
print(f"    Относительная ошибка: {abs(C - C_analytical)/C_analytical*100:.4f}%")

print("\n" + "=" * 70)
if abs(C - C_analytical) < 1e-6:
    print("✅ Ёмкость, вычисленная онтологически, совпадает с аналитической формулой.")
else:
    print("⚠️ Проверьте параметры: возможно, требуется калибровка k_EM или η.")
print("=" * 70)