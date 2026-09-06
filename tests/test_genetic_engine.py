"""
Testes Unitários para o Motor do Algoritmo Genético
"""

import pytest
import numpy as np
from genetic_engine import Individual, GeneticEngine


def test_individual_initialization():
    length = 10
    ind = Individual(chromosome_length=length)
    assert len(ind.genes) == length
    assert ind.fitness == 0.0
    assert ind.rmse == float("inf")


def test_individual_evaluation():
    # Criação de dados sintéticos simples: y = 2 * x0 - 1.5 * x1 + 0.5
    np.random.seed(42)
    X = np.random.randn(100, 3)
    weights = np.array([2.0, -1.5, 0.5])
    y = np.dot(X, weights)

    ind = Individual(chromosome_length=3, genes=weights)
    fitness = ind.evaluate(X, y)

    assert ind.rmse < 1e-5
    assert ind.mae < 1e-5
    assert fitness > 500.0


def test_crossover_and_mutation():
    parent1 = Individual(chromosome_length=5, genes=np.ones(5))
    parent2 = Individual(chromosome_length=5, genes=-np.ones(5))

    engine = GeneticEngine(population_size=20, generations=5, crossover_rate=1.0, mutation_rate=0.5)
    child1, child2 = engine.arithmetic_crossover(parent1, parent2)

    assert len(child1.genes) == 5
    assert len(child2.genes) == 5
    # Verifica que houve recombinação
    assert not np.array_equal(child1.genes, parent1.genes)

    # Testa mutação
    genes_before = child1.genes.copy()
    engine.mutate(child1, current_gen=1, max_gen=10)
    assert not np.array_equal(child1.genes, genes_before)


def test_genetic_engine_convergence():
    # Otimização em dados lineares
    np.random.seed(42)
    X = np.random.randn(80, 4)
    true_weights = np.array([1.2, -0.8, 0.5, 0.1])
    y = np.dot(X, true_weights) + np.random.normal(0, 0.05, size=80)

    engine = GeneticEngine(
        population_size=40,
        generations=20,
        crossover_rate=0.85,
        mutation_rate=0.15,
        elitism_ratio=0.1,
        random_seed=42
    )

    best_ind = engine.evolve(X, y)

    assert best_ind is not None
    assert len(engine.history["generation"]) == 20
    # Verifica que o fitness melhorou ao longo das gerações
    initial_fitness = engine.history["best_fitness"][0]
    final_fitness = engine.history["best_fitness"][-1]
    assert final_fitness >= initial_fitness
    assert best_ind.rmse < 0.5
