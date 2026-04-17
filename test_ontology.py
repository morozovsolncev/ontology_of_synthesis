#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_ontology.py
Тестовая среда для ontological_physics.py (Этап 1)
"""

import sys
import numpy as np
from ontological_physics import Universe, Particle, Field, Simulator, QuantumSystem

def test_gravity_field():
    """Тест гравитационного поля."""
    uni = Universe(eta=1.0, dim=2)
    field = Field(uni, 'gravity', source_strength=5.0)
    r = np.array([2.0, 0.0])
    beta = field.beta(r)
    lnPr = field.ln_pr(r)
    # Ожидания: β = -5/2 = -2.5, ln Pr = -5·ln2 ≈ -3.4657
    assert abs(beta + 2.5) < 1e-6
    assert abs(lnPr + 3.4657359027997265) < 1e-6
    print("  ✓ gravity: β и ln Pr совпадают с аналитикой")

def test_coulomb_field():
    """Тест кулоновского поля."""
    uni = Universe(eta=1.0, dim=2)
    field = Field(uni, 'coulomb', source_strength=3.0)
    r = np.array([2.0, 0.0])
    beta = field.beta(r)
    lnPr = field.ln_pr(r)
    # Ожидания: β = -3/2 = -1.5, ln Pr = -3·ln2 ≈ -2.0794
    assert abs(beta + 1.5) < 1e-6
    assert abs(lnPr + 2.0794415416798357) < 1e-6
    print("  ✓ coulomb: β и ln Pr совпадают с аналитикой")

def test_oscillator_field():
    """Тест поля осциллятора."""
    uni = Universe(eta=1.0, dim=1)
    field = Field(uni, 'oscillator', source_strength=2.0)
    r = np.array([1.0])
    beta = field.beta(r)
    r0 = np.array([0.0])
    lnPr = field.ln_pr(r, r0)
    # Ожидания: β = -2·1² = -2, ln Pr = -2·(1³ - 0³)/3 = -2/3 ≈ -0.6667
    assert abs(beta + 2.0) < 1e-6
    assert abs(lnPr + 0.6666666666666666) < 1e-6
    print("  ✓ oscillator: β и ln Pr совпадают с аналитикой")

def test_well_field():
    """Тест потенциальной ямы."""
    uni = Universe(eta=1.0, dim=1)
    field = Field(uni, 'well', source_strength=4.0)  # L=4, от -2 до 2
    r_inside = np.array([1.0])
    r_outside = np.array([3.0])
    beta_in = field.beta(r_inside)
    beta_out = field.beta(r_outside)
    lnPr_in = field.ln_pr(r_inside)
    lnPr_out = field.ln_pr(r_outside)
    assert beta_in == 0.0
    assert beta_out < -1e10  # очень большое отрицательное
    assert lnPr_in == 0.0
    assert lnPr_out < -1e10
    print("  ✓ well: внутри β=0, ln Pr=0; снаружи β и ln Pr стремятся к -∞")

def test_gradient():
    """Тест численного градиента."""
    uni = Universe(eta=1.0, dim=2)
    field = Field(uni, 'gravity', source_strength=5.0)
    r = np.array([2.0, 0.0])
    grad = field.gradient_ln_pr(r)
    # Аналитический градиент: d/dx (-5·ln r) = -5·x/r²
    expected_grad = np.array([-5.0 * 2.0 / 4.0, 0.0])  # [-2.5, 0]
    assert abs(grad[0] - expected_grad[0]) < 1e-4
    assert abs(grad[1] - expected_grad[1]) < 1e-4
    print("  ✓ численный градиент совпадает с аналитическим")

def run_all_tests():
    print("=" * 70)
    print("ТЕСТИРОВАНИЕ ontological_physics.py (ЭТАП 1)")
    print("=" * 70)

    test_gravity_field()
    test_coulomb_field()
    test_oscillator_field()
    test_well_field()
    test_gradient()

    print("\n" + "=" * 70)
    print("✅ Все тесты Этапа 1 пройдены.")
    print("=" * 70)

if __name__ == "__main__":
    run_all_tests()