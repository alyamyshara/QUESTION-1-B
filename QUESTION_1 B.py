# app.py
import random
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

# ---- Fixed GA Parameters (LAB TEST requirements) ----
POP_SIZE = 300          # Population = 300
CHROM_LEN = 80          # Chromosome Length = 80
TARGET_ONES = 40        # Fitness peaks at ones = 40
MAX_FITNESS = 80        # Max fitness = 80
N_GENERATIONS = 50      # Generations = 50

# ---- GA Hyperparameters ----
TOURNAMENT_K = 3
CROSSOVER_RATE = 0.9
MUTATION_RATE = 1.0 / CHROM_LEN

# ---- Fitness Function ----
def fitness(individual: np.ndarray) -> float:
    """
    Fitness peaks when the number of 1s equals TARGET_ONES.
    Max fitness is MAX_FITNESS (=80) at ones == 40.
    """
    ones = int(individual.sum())
    return MAX_FITNESS - abs(ones - TARGET_ONES)

# ---- GA Operators ----
def init_population(pop_size: int, chrom_len: int) -> np.ndarray:
    return np.random.randint(0, 2, size=(pop_size, chrom_len), dtype=np.int8)

def tournament_selection(pop: np.ndarray, fits: np.ndarray, k: int) -> np.ndarray:
    idxs = np.random.randint(0, len(pop), size=k)
    best_idx = idxs[np.argmax(fits[idxs])]
    return pop[best_idx].copy()

def single_point_crossover(p1: np.ndarray, p2: np.ndarray):
    if np.random.rand() > CROSSOVER_RATE:
        return p1.copy(), p2.copy()
    point = np.random.randint(1, CHROM_LEN)
    c1 = np.concatenate([p1[:point], p2[point:]])
    c2 = np.concatenate([p2[:point], p1[point:]])
    return c1, c2

def mutate(individual: np.ndarray) -> np.ndarray:
    mask = np.random.rand(CHROM_LEN) < MUTATION_RATE
    individual[mask] = 1 - individual[mask]
    return individual

def evolve(pop: np.ndarray, generations: int):
    best_fitness_per_gen = []
    best_individual = None
    best_f = -np.inf

    for _ in range(generations):
        fits = np.array([fitness(ind) for ind in pop])

        gen_best_idx = np.argmax(fits)
        gen_best = pop[gen_best_idx]
        gen_best_f = fits[gen_best_idx]
        best_fitness_per_gen.append(float(gen_best_f))

        if gen_best_f > best_f:
            best_f = float(gen_best_f)
            best_individual = gen_best.copy()

        new_pop = []
        while len(new_pop) < len(pop):
            p1 = tournament_selection(pop, fits, TOURNAMENT_K)
            p2 = tournament_selection(pop, fits, TOURNAMENT_K)
            c1, c2 = single_point_crossover(p1, p2)
            c1 = mutate(c1)
            c2 = mutate(c2)
            new_pop.extend([c1, c2])

        pop = np.array(new_pop[:len(pop)], dtype=np.int8)

    return best_individual, best_f, best_fitness_per_gen

# ---- Streamlit UI ----
st.set_page_config(page_title="Genetic Algorithm Bit Pattern Generator", page_icon="🧬")

st.title("🧬 Genetic Algorithm Bit Pattern Generator")
st.caption(
    "Population = 300 | Chromosome Length = 80 | Generations = 50\n"
    "Fitness peaks at ones = 40 with maximum fitness = 80."
)

seed = st.number_input("Random seed", min_value=0, value=42)
run_btn = st.button("Run Genetic Algorithm", type="primary")

if run_btn:
    random.seed(seed)
    np.random.seed(seed)

    population = init_population(POP_SIZE, CHROM_LEN)
    best_ind, best_fit, curve = evolve(population, N_GENERATIONS)

    ones_count = int(best_ind.sum())
    bitstring = "".join(map(str, best_ind.tolist()))

    st.subheader("Best Solution Found")
    st.metric("Best Fitness", f"{best_fit:.0f}")
    st.write(f"Number of Ones: {ones_count} / {CHROM_LEN}")
    st.code(bitstring)

    st.subheader("Convergence Plot")
    fig, ax = plt.subplots()
    ax.plot(range(1, len(curve) + 1), curve)
    ax.set_xlabel("Generation")
    ax.set_ylabel("Best Fitness")
    ax.set_title("Fitness Improvement Across Generations")
    ax.grid(True)
    st.pyplot(fig)

    if best_fit == MAX_FITNESS:
        st.success("Optimal solution achieved (ones = 40, fitness = 80).")
    else:
        st.info("Near-optimal solution achieved. Try another seed for variation.")
