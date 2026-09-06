"""
Motor de Algoritmo Genético (Genetic Algorithm Engine)
Otimização evolutiva de pesos e coeficientes para modelos de previsão temporal.
"""

import copy
import logging
import random
from typing import Callable, Dict, List, Optional, Tuple
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class Individual:
    """
    Representação de um cromossomo/indivíduo na população evolutiva.
    O genoma é um vetor de ponto flutuante contendo os pesos das features e bias.
    """

    def __init__(self, chromosome_length: int, genes: Optional[np.ndarray] = None):
        self.length = chromosome_length
        if genes is not None:
            self.genes = np.array(genes, dtype=np.float64)
        else:
            # Inicialização com valores aleatórios (distribuição normal com escala moderada)
            self.genes = np.random.uniform(-1.0, 1.0, size=chromosome_length).astype(np.float64)
            # Normalização suave inicial para evitar saturação
            norm = np.linalg.norm(self.genes)
            if norm > 0:
                self.genes /= norm

        self.fitness: float = 0.0
        self.rmse: float = float("inf")
        self.mae: float = float("inf")
        self.directional_accuracy: float = 0.0

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Avalia o indivíduo calculando as predições lineares ponderadas: y_pred = X @ genes
        Retorna o score de fitness.
        """
        predictions = np.dot(X, self.genes)
        errors = y - predictions

        # Métricas de erro
        mse = np.mean(errors ** 2)
        self.rmse = float(np.sqrt(mse))
        self.mae = float(np.mean(np.abs(errors)))

        # Acurácia direcional (se acertou o sinal da variação em relação ao passo anterior)
        # O último termo do lag em X[:, -6] ou o sinal de y relativo a 0
        actual_direction = np.sign(y)
        pred_direction = np.sign(predictions)
        matches = np.sum(actual_direction == pred_direction)
        self.directional_accuracy = float(matches / len(y)) if len(y) > 0 else 0.0

        # Função de Aptidão (Fitness): Minimizar erro e maximizar assertividade direcional
        # Quanto menor o RMSE e MAE, maior o fitness.
        base_fitness = 1000.0 / (1.0 + (self.rmse * 10.0) + (self.mae * 5.0))
        directional_bonus = 1.0 + (0.4 * self.directional_accuracy)
        
        # Penalidade por saturação de pesos extremos (regularização L2 embutida no fitness)
        l2_penalty = 0.01 * np.sum(self.genes ** 2)

        self.fitness = float(max(1e-4, (base_fitness * directional_bonus) - l2_penalty))
        return self.fitness

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Executa a inferência nos dados de entrada."""
        return np.dot(X, self.genes)


class GeneticEngine:
    """
    Controlador do Algoritmo Genético:
    Gerencia a evolução da população, seleção, cruzamento, mutação e elitismo.
    """

    def __init__(
        self,
        population_size: int = 60,
        generations: int = 35,
        crossover_rate: float = 0.85,
        mutation_rate: float = 0.15,
        elitism_ratio: float = 0.08,
        tournament_size: int = 4,
        random_seed: Optional[int] = 42
    ):
        self.population_size = max(10, population_size)
        self.generations = max(5, generations)
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.elitism_ratio = elitism_ratio
        self.tournament_size = tournament_size

        if random_seed is not None:
            random.seed(random_seed)
            np.random.seed(random_seed)

        self.population: List[Individual] = []
        self.best_individual: Optional[Individual] = None
        self.history: Dict[str, List[float]] = {
            "generation": [],
            "best_fitness": [],
            "avg_fitness": [],
            "best_rmse": [],
            "best_mae": [],
            "best_directional_acc": []
        }

    def initialize_population(self, chromosome_length: int):
        """Inicializa a população com indivíduos de comprimento especificado."""
        self.population = [Individual(chromosome_length) for _ in range(self.population_size)]
        self.history = {
            "generation": [],
            "best_fitness": [],
            "avg_fitness": [],
            "best_rmse": [],
            "best_mae": [],
            "best_directional_acc": []
        }
        self.best_individual = None

    def tournament_selection(self) -> Individual:
        """Seleciona o melhor indivíduo entre k competidores sorteados."""
        candidates = random.sample(self.population, min(self.tournament_size, len(self.population)))
        return max(candidates, key=lambda ind: ind.fitness)

    def arithmetic_crossover(self, parent1: Individual, parent2: Individual) -> Tuple[Individual, Individual]:
        """
        Cruzamento Aritmético com recombinação linear e dispersão aleatória (BLX-alpha adaptado).
        """
        if random.random() > self.crossover_rate:
            return copy.deepcopy(parent1), copy.deepcopy(parent2)

        alpha = random.uniform(0.1, 0.9)
        child1_genes = (alpha * parent1.genes) + ((1.0 - alpha) * parent2.genes)
        child2_genes = ((1.0 - alpha) * parent1.genes) + (alpha * parent2.genes)

        # Perturbação BLX suave para explorar a vizinhança genética
        blx_noise = np.random.normal(0, 0.05, size=parent1.length)
        child1_genes += blx_noise
        child2_genes -= blx_noise

        return Individual(parent1.length, child1_genes), Individual(parent2.length, child2_genes)

    def mutate(self, individual: Individual, current_gen: int, max_gen: int):
        """
        Mutação gaussiana adaptativa: a magnitude do ruído diminui conforme as gerações avançam.
        """
        adaptive_scale = max(0.02, 0.3 * (1.0 - (current_gen / max_gen)))

        for i in range(individual.length):
            if random.random() < self.mutation_rate:
                # Mutação por perturbação gaussiana
                noise = np.random.normal(0, adaptive_scale)
                individual.genes[i] += noise

            # Mutação de reset ocasional para escapar de mínimos locais
            if random.random() < (self.mutation_rate * 0.2):
                individual.genes[i] = random.uniform(-1.0, 1.0)

    def evolve(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        callback: Optional[Callable[[int, int, float, float, float], None]] = None
    ) -> Individual:
        """
        Executa o ciclo completo de gerações do Algoritmo Genético.
        Recebe opcionalmente uma função callback para atualização da interface gráfica.
        """
        chromosome_len = X_train.shape[1]
        self.initialize_population(chromosome_len)

        # Quantidade de indivíduos mantidos por elitismo
        elite_count = max(1, int(self.population_size * self.elitism_ratio))

        for gen in range(1, self.generations + 1):
            # 1. Avaliação de toda a população
            for ind in self.population:
                ind.evaluate(X_train, y_train)

            # 2. Ordenação da população por aptidão decrescente
            self.population.sort(key=lambda ind: ind.fitness, reverse=True)

            current_best = self.population[0]
            if self.best_individual is None or current_best.fitness > self.best_individual.fitness:
                self.best_individual = copy.deepcopy(current_best)

            # Coleta de métricas e histórico
            fitnesses = [ind.fitness for ind in self.population]
            avg_fit = float(np.mean(fitnesses))
            best_fit = float(self.best_individual.fitness)
            best_rmse = float(self.best_individual.rmse)
            best_mae = float(self.best_individual.mae)
            best_dir = float(self.best_individual.directional_accuracy)

            self.history["generation"].append(gen)
            self.history["best_fitness"].append(best_fit)
            self.history["avg_fitness"].append(avg_fit)
            self.history["best_rmse"].append(best_rmse)
            self.history["best_mae"].append(best_mae)
            self.history["best_directional_acc"].append(best_dir)

            # Callback para GUI / log
            if callback:
                callback(gen, self.generations, best_fit, avg_fit, best_rmse)

            # 3. Criação da nova geração
            new_population: List[Individual] = []

            # Elitismo: os melhores são copiados diretamente
            for i in range(elite_count):
                new_population.append(copy.deepcopy(self.population[i]))

            # Preenchimento do restante da população via Seleção, Crossover e Mutação
            while len(new_population) < self.population_size:
                parent1 = self.tournament_selection()
                parent2 = self.tournament_selection()

                child1, child2 = self.arithmetic_crossover(parent1, parent2)

                self.mutate(child1, gen, self.generations)
                self.mutate(child2, gen, self.generations)

                new_population.append(child1)
                if len(new_population) < self.population_size:
                    new_population.append(child2)

            self.population = new_population

        logger.info(
            f"Evolução concluída. Melhor Fitness: {self.best_individual.fitness:.2f} | "
            f"RMSE: {self.best_individual.rmse:.4f} | MAE: {self.best_individual.mae:.4f}"
        )
        return self.best_individual
