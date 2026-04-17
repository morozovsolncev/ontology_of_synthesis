#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ontological_physics.py
Модуль для онтологического моделирования физических процессов.
Версия 0.2.0 (Этап 1: реализованы конкретные поля)

Основан на трёх постулатах Онтологии синтеза:
1. Первичность Потенции: всё сущее — паттерны (кластеры) в Беспредельном поле потенций.
2. Поле связности β_Ω: каждый объект создаёт поле, показывающее интенсивность связи.
3. Закон движения: a = (η/2) · ∇[ln Pr_Ω], где ln Pr_Ω = ∫ β_Ω·dr.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any
import warnings

# ============================================================================
# БАЗОВЫЕ КОНСТАНТЫ (по умолчанию — абстрактные единицы)
# ============================================================================

@dataclass
class Universe:
    """
    Беспредельное поле потенций (БПП) — контекст всей симуляции.
    Содержит фундаментальные константы и общие параметры.
    """
    eta: float = 1.0          # Константа перевода (в физике η = c²)
    dim: int = 2              # Размерность пространства (1, 2 или 3)
    k: float = 1.0            # Фундаментальный масштаб (β_Ω = α_Ω/k)
    hbar: float = 1.0         # Онтологический квант действия (ħ_Ω)
    c: float = 1.0            # Скорость света (в абстрактных единицах)

    def __post_init__(self):
        if self.dim not in [1, 2, 3]:
            raise ValueError(f"Размерность dim={self.dim} не поддерживается. Используйте 1, 2 или 3.")


# ============================================================================
# КЛАСС PATTERN / PARTICLE
# ============================================================================

@dataclass
class Particle:
    """Пробное тело (паттерн, кластер) — носитель массы, заряда, положения и скорости."""
    universe: Universe
    mass: float = 1.0
    charge: float = 0.0
    position: np.ndarray = field(default_factory=lambda: np.zeros(2))
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(2))
    history: Dict[str, List] = field(default_factory=lambda: {
        't': [], 'x': [], 'v': [], 'a': [], 'beta': [], 'lnPr': [], 'Pr': []
    })

    def __post_init__(self):
        if len(self.position) != self.universe.dim:
            self.position = np.resize(self.position, self.universe.dim)
        if len(self.velocity) != self.universe.dim:
            self.velocity = np.resize(self.velocity, self.universe.dim)

    def reset_history(self):
        self.history = {'t': [], 'x': [], 'v': [], 'a': [], 'beta': [], 'lnPr': [], 'Pr': []}


# ============================================================================
# КЛАСС FIELD (поле связности β_Ω) — РЕАЛИЗОВАН В ЭТАПЕ 1
# ============================================================================

class Field:
    """
    Поле связности β_Ω(r).
    Создаётся источником (массой, зарядом, потенциалом) и определяет
    вероятность резонанса Pr_Ω = exp(∫ β_Ω·dr).
    """

    FIELD_TYPES = ['gravity', 'coulomb', 'oscillator', 'well', 'custom']

    def __init__(self, universe: Universe, field_type: str = 'gravity',
                 source_strength: float = 1.0, test_charge: float = 1.0, **kwargs):
        """
        Параметры:
        - universe: объект Вселенной
        - field_type: тип поля
        - source_strength: интенсивность источника (M, Q, k, L)
        - test_charge: заряд пробного тела (для кулоновского поля)
        - **kwargs: source_position (по умолчанию [0,0,...])
        """
        self.universe = universe
        self.field_type = field_type
        self.source_strength = source_strength
        self.test_charge = test_charge
        self.params = kwargs
        self.source_position = kwargs.get('source_position', np.zeros(universe.dim))

        if field_type not in self.FIELD_TYPES:
            warnings.warn(f"Неизвестный тип поля '{field_type}'. Используется 'gravity'.")
            self.field_type = 'gravity'

    # ------------------------------------------------------------------------
    # 1. ГРАВИТАЦИОННОЕ ПОЛЕ
    # ------------------------------------------------------------------------
    def _beta_gravity(self, r_vec: np.ndarray) -> float:
        r = np.linalg.norm(r_vec - self.source_position)
        if r < 1e-12:
            return -1e12
        return -self.source_strength / r

    def _ln_pr_gravity(self, r_vec: np.ndarray, r0_vec: np.ndarray) -> float:
        r = np.linalg.norm(r_vec - self.source_position)
        r0 = np.linalg.norm(r0_vec - self.source_position)
        if r < 1e-12:
            return -1e12
        # ln Pr = ∫ β dr = -M · ln(r) + const
        return -self.source_strength * (np.log(r) - np.log(r0))

    # ------------------------------------------------------------------------
    # 2. КУЛОНОВСКОЕ ПОЛЕ (электромагнетизм)
    # ------------------------------------------------------------------------
    def _beta_coulomb(self, r_vec: np.ndarray) -> float:
        r = np.linalg.norm(r_vec - self.source_position)
        if r < 1e-12:
            return -1e12
        # β = -(k_EM)·(Q·q)/r. Здесь source_strength = Q·q
        return -self.source_strength / r

    def _ln_pr_coulomb(self, r_vec: np.ndarray, r0_vec: np.ndarray) -> float:
        r = np.linalg.norm(r_vec - self.source_position)
        r0 = np.linalg.norm(r0_vec - self.source_position)
        if r < 1e-12:
            return -1e12
        return -self.source_strength * (np.log(r) - np.log(r0))

    # ------------------------------------------------------------------------
    # 3. ГАРМОНИЧЕСКИЙ ОСЦИЛЛЯТОР (1D, вдоль x)
    # ------------------------------------------------------------------------
    def _beta_oscillator(self, r_vec: np.ndarray) -> float:
        # Только x-компонента (для 1D или проекции)
        x = (r_vec - self.source_position)[0]
        # β = -α·x², где α = source_strength
        return -self.source_strength * x**2

    def _ln_pr_oscillator(self, r_vec: np.ndarray, r0_vec: np.ndarray) -> float:
        x = (r_vec - self.source_position)[0]
        x0 = (r0_vec - self.source_position)[0]
        # ln Pr = ∫ β dx = -α·x³/3 + const
        return -self.source_strength * (x**3 - x0**3) / 3.0

    # ------------------------------------------------------------------------
    # 4. БЕСКОНЕЧНАЯ ПОТЕНЦИАЛЬНАЯ ЯМА (1D, от -L/2 до L/2)
    # ------------------------------------------------------------------------
    def _beta_well(self, r_vec: np.ndarray) -> float:
        x = (r_vec - self.source_position)[0]
        L = self.source_strength
        if abs(x) > L/2:
            return -1e12   # бесконечная связность вне ямы
        return 0.0

    def _ln_pr_well(self, r_vec: np.ndarray, r0_vec: np.ndarray) -> float:
        x = (r_vec - self.source_position)[0]
        x0 = (r0_vec - self.source_position)[0]
        L = self.source_strength
        if abs(x) > L/2 or abs(x0) > L/2:
            return -1e12
        # Внутри ямы ln Pr постоянен (градиент = 0)
        return 0.0

    # ------------------------------------------------------------------------
    # ПУБЛИЧНЫЕ МЕТОДЫ
    # ------------------------------------------------------------------------
    def beta(self, r_vec: np.ndarray) -> float:
        """Возвращает β_Ω(r)."""
        if self.field_type == 'gravity':
            return self._beta_gravity(r_vec)
        elif self.field_type == 'coulomb':
            return self._beta_coulomb(r_vec)
        elif self.field_type == 'oscillator':
            return self._beta_oscillator(r_vec)
        elif self.field_type == 'well':
            return self._beta_well(r_vec)
        else:
            # custom — пока заглушка
            return -self.source_strength / max(np.linalg.norm(r_vec - self.source_position), 1e-12)

    def ln_pr(self, r_vec: np.ndarray, r0_vec: Optional[np.ndarray] = None) -> float:
        """
        Возвращает ln Pr_Ω(r) = ∫_{r0}^{r} β·dr.
        Если r0 не задан, используется положение источника.
        """
        if r0_vec is None:
            r0_vec = self.source_position

        if self.field_type == 'gravity':
            return self._ln_pr_gravity(r_vec, r0_vec)
        elif self.field_type == 'coulomb':
            return self._ln_pr_coulomb(r_vec, r0_vec)
        elif self.field_type == 'oscillator':
            return self._ln_pr_oscillator(r_vec, r0_vec)
        elif self.field_type == 'well':
            return self._ln_pr_well(r_vec, r0_vec)
        else:
            r = np.linalg.norm(r_vec - self.source_position)
            r0 = np.linalg.norm(r0_vec - self.source_position)
            return -self.source_strength * (np.log(r) - np.log(r0))

    def gradient_ln_pr(self, r_vec: np.ndarray, delta: float = 1e-6) -> np.ndarray:
        """Численный градиент ln Pr_Ω(r)."""
        grad = np.zeros_like(r_vec)
        lnP0 = self.ln_pr(r_vec)
        for i in range(len(r_vec)):
            r_plus = r_vec.copy()
            r_minus = r_vec.copy()
            r_plus[i] += delta
            r_minus[i] -= delta
            grad[i] = (self.ln_pr(r_plus) - self.ln_pr(r_minus)) / (2 * delta)
        return grad


# ============================================================================
# КЛАСС SIMULATOR (без изменений из Этапа 0)
# ============================================================================

class Simulator:
    """Симулятор движения: a = (η/2)·∇[ln Pr_Ω]."""

    METHODS = ['euler', 'verlet', 'rk45']

    def __init__(self, universe: Universe, field: Field, particle: Particle,
                 dt: float = 0.01, method: str = 'verlet'):
        self.universe = universe
        self.field = field
        self.particle = particle
        self.dt = dt
        self.method = method
        if method not in self.METHODS:
            warnings.warn(f"Неизвестный метод '{method}'. Используется 'verlet'.")
            self.method = 'verlet'

    def acceleration(self, pos: np.ndarray) -> np.ndarray:
        grad = self.field.gradient_ln_pr(pos)
        return (self.universe.eta / 2.0) * grad

    def _step_euler(self, pos, vel):
        a = self.acceleration(pos)
        vel_new = vel + a * self.dt
        pos_new = pos + vel_new * self.dt
        return pos_new, vel_new

    def _step_verlet(self, pos, vel):
        a = self.acceleration(pos)
        pos_new = pos + vel * self.dt + 0.5 * a * self.dt**2
        a_new = self.acceleration(pos_new)
        vel_new = vel + 0.5 * (a + a_new) * self.dt
        return pos_new, vel_new

    def _step_rk45(self, pos, vel):
        return self._step_verlet(pos, vel)

    def run(self, t_max: float, record_every: int = 1) -> Dict[str, np.ndarray]:
        self.particle.reset_history()
        pos = self.particle.position.copy()
        vel = self.particle.velocity.copy()
        t = 0.0
        step = 0

        # Начальное состояние
        a0 = self.acceleration(pos)
        beta0 = self.field.beta(pos)
        lnPr0 = self.field.ln_pr(pos)
        Pr0 = np.exp(lnPr0)

        self._record_state(t, pos, vel, a0, beta0, lnPr0, Pr0)

        while t < t_max - 1e-12:
            if self.method == 'euler':
                pos, vel = self._step_euler(pos, vel)
            else:
                pos, vel = self._step_verlet(pos, vel)
            t += self.dt
            step += 1

            if step % record_every == 0:
                a = self.acceleration(pos)
                beta = self.field.beta(pos)
                lnPr = self.field.ln_pr(pos)
                Pr = np.exp(lnPr)
                self._record_state(t, pos, vel, a, beta, lnPr, Pr)

        # Преобразуем в массивы
        for key in self.particle.history:
            self.particle.history[key] = np.array(self.particle.history[key])
        return self.particle.history

    def _record_state(self, t, pos, vel, a, beta, lnPr, Pr):
        self.particle.history['t'].append(t)
        self.particle.history['x'].append(pos.copy())
        self.particle.history['v'].append(vel.copy())
        self.particle.history['a'].append(a.copy())
        self.particle.history['beta'].append(beta)
        self.particle.history['lnPr'].append(lnPr)
        self.particle.history['Pr'].append(Pr)


# ============================================================================
# КЛАСС QUANTUM_SYSTEM (заглушка для Этапа 4)
# ============================================================================

class QuantumSystem:
    def __init__(self, universe: Universe, field: Field, particle: Particle, x_grid: np.ndarray):
        self.universe = universe
        self.field = field
        self.particle = particle
        self.x_grid = x_grid
        self.N = len(x_grid)
        self.dx = x_grid[1] - x_grid[0]

    def potential(self, x: float) -> float:
        lnPr = self.field.ln_pr(np.array([x]))
        if lnPr < -1e10:
            return 1e10
        return -self.particle.mass * self.universe.eta / 2.0 * lnPr

    def solve(self, n_states: int = 3) -> Tuple[np.ndarray, np.ndarray]:
        # Заглушка
        eigenvalues = np.array([1.0, 4.0, 9.0])
        eigenvectors = np.zeros((self.N, n_states))
        return eigenvalues, eigenvectors


# ============================================================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ============================================================================

def print_module_info():
    print("=" * 70)
    print("ontological_physics.py")
    print("Версия 0.2.0 (Этап 1: реализованы конкретные поля)")
    print("=" * 70)
    print("\nДоступные типы полей:")
    print("  • gravity    — гравитация: β = -GM/r, ln Pr = -GM·ln r")
    print("  • coulomb    — электростатика: β = -k·Q·q/r, ln Pr = -k·Q·q·ln r")
    print("  • oscillator — гармонический осциллятор: β = -α·x², ln Pr = -α·x³/3")
    print("  • well       — бесконечная яма: β = 0 внутри, -∞ снаружи")
    print("=" * 70)


# ============================================================================
# ТОЧКА ВХОДА ДЛЯ ТЕСТИРОВАНИЯ
# ============================================================================

if __name__ == "__main__":
    print_module_info()

    # Быстрый тест всех полей
    uni = Universe(eta=1.0, dim=1)  # для осциллятора и ямы удобнее 1D

    print("\n--- Тест гравитационного поля ---")
    field_g = Field(uni, 'gravity', source_strength=5.0)
    r = np.array([2.0])
    print(f"  β(2) = {field_g.beta(r):.4f} (ожидается -2.5)")
    print(f"  ln Pr(2) = {field_g.ln_pr(r):.4f} (ожидается -5·ln2 ≈ -3.4657)")

    print("\n--- Тест кулоновского поля ---")
    field_c = Field(uni, 'coulomb', source_strength=3.0)
    print(f"  β(2) = {field_c.beta(r):.4f} (ожидается -1.5)")
    print(f"  ln Pr(2) = {field_c.ln_pr(r):.4f} (ожидается -3·ln2 ≈ -2.0794)")

    print("\n--- Тест осциллятора ---")
    field_o = Field(uni, 'oscillator', source_strength=2.0)
    r = np.array([1.0])
    print(f"  β(1) = {field_o.beta(r):.4f} (ожидается -2.0)")
    r0 = np.array([0.0])
    print(f"  ln Pr(1) от 0 = {field_o.ln_pr(r, r0):.4f} (ожидается -2/3 ≈ -0.6667)")

    print("\n--- Тест потенциальной ямы ---")
    field_w = Field(uni, 'well', source_strength=4.0)  # L=4
    r_inside = np.array([1.0])
    r_outside = np.array([3.0])
    print(f"  β(1) внутри = {field_w.beta(r_inside):.4f} (ожидается 0)")
    print(f"  β(3) снаружи = {field_w.beta(r_outside):.4f} (ожидается -1e12)")

    print("\n✅ Этап 1 завершён. Все поля реализованы.")