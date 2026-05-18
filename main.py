import random
import numpy as np
import matplotlib.pyplot as plt
import os
import json

from environment import PacManEnv
from train_qlearning import train_qlearning
from train_sarsa import train_sarsa
from visualizer import visualize_policy
from visualizer import manual_play
from save_results import save_results_to_csv


# Cartella dove salvare grafici e risultati
results_dir = "results"
os.makedirs(results_dir, exist_ok=True)

# Azioni possibili:
# 0 = su
# 1 = giù
# 2 = sinistra
# 3 = destra
actions = [0, 1, 2, 3]


def safe_filename(title):
    """
    Trasforma il titolo di un grafico in un nome file sicuro.

    Esempio:
    "Q-learning Training" diventa "qlearning_training.png"
    """

    return title.lower().replace(" ", "_").replace("-", "").replace(":", "")


def get_q_values(Q, state):
    """
    Restituisce i valori Q associati a uno stato.

    Se lo stato non è presente nella Q-table,
    viene inizializzato con quattro valori a zero,
    uno per ogni azione possibile.
    """

    if state not in Q:
        Q[state] = np.zeros(len(actions))

    return Q[state]


def plot_training_results(title, rewards_per_episode, q0_values):
    """
    Crea e salva il grafico dell'andamento del training.

    Nel grafico vengono mostrati:
    - reward di ogni episodio
    - media cumulativa delle reward
    - media mobile delle reward
    - andamento smussato del valore Q dello stato iniziale
    """

    # Media cumulativa:
    # a ogni episodio calcola la media di tutte le reward ottenute fino a quel punto
    cumulative_avg = (
        np.cumsum(rewards_per_episode)
        / np.arange(1, len(rewards_per_episode) + 1)
    )

    # Finestra per la media mobile delle reward
    window = 200

    # Media mobile:
    # serve a rendere più leggibile l'andamento delle reward
    moving_avg = np.convolve(
        rewards_per_episode,
        np.ones(window) / window,
        mode="valid"
    )

    # Finestra per rendere più liscio il grafico di Q0
    window_q0 = 200

    # Media mobile dei valori Q dello stato iniziale
    q0_smooth = np.convolve(
        q0_values,
        np.ones(window_q0) / window_q0,
        mode="valid"
    )

    # Crea figura e primo asse y
    fig, ax1 = plt.subplots()

    # Reward grezza di ogni episodio
    ax1.plot(
        rewards_per_episode,
        label="Reward episodio",
        alpha=0.2
    )

    # Media cumulativa della reward
    ax1.plot(
        cumulative_avg,
        label="Media cumulativa",
        linewidth=2
    )

    # Media mobile della reward
    ax1.plot(
        range(window - 1, len(rewards_per_episode)),
        moving_avg,
        label="Media mobile",
        linewidth=2
    )

    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Reward")

    # Secondo asse y per visualizzare i valori Q
    ax2 = ax1.twinx()

    ax2.plot(
        range(window_q0 - 1, len(q0_values)),
        q0_smooth,
        color="red",
        label="Q0 smoothed",
        linewidth=2
    )

    ax2.set_ylabel("Q-value")

    # Unisce le legende dei due assi
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()

    ax1.legend(
        lines_1 + lines_2,
        labels_1 + labels_2,
        loc="lower right"
    )

    plt.title(title)

    # Salva il grafico nella cartella results
    filename = safe_filename(title) + ".png"
    plt.savefig(
        os.path.join(results_dir, filename),
        dpi=300,
        bbox_inches="tight"
    )

    # Mostra il grafico a schermo
    plt.show()


def evaluate_random_agent(episodes=100):
    """
    Valuta un agente casuale.

    L'agente casuale sceglie a ogni passo una delle quattro azioni
    senza usare nessuna Q-table.

    Restituisce:
    - reward media
    - cibo medio mangiato
    - numero medio di step
    - collision rate
    """

    env = PacManEnv()

    rewards = []
    foods = []
    steps_list = []
    collisions = 0

    for _ in range(episodes):

        # Nuovo episodio
        env.reset()
        done = False
        total_reward = 0

        while not done:

            # Azione scelta casualmente
            action = random.choice(actions)

            # Esecuzione azione nell'ambiente
            _, reward, done = env.step(action)

            # Accumulo reward
            total_reward += reward

        # Salvataggio metriche dell'episodio
        rewards.append(total_reward)
        foods.append(env.food_eaten)
        steps_list.append(env.steps)

        # Controllo se l'episodio è terminato per collisione
        if env.pacman == env.ghost:
            collisions += 1

    return (
        np.mean(rewards),
        np.mean(foods),
        np.mean(steps_list),
        collisions / episodes
    )


def evaluate_trained_agent(Q, episodes=100):
    """
    Valuta un agente già addestrato.

    L'agente usa la Q-table e sceglie sempre l'azione con valore Q massimo.
    Quindi in valutazione non esplora più: usa solo la policy greedy.
    """

    env = PacManEnv()

    rewards = []
    foods = []
    steps_list = []
    collisions = 0

    for _ in range(episodes):

        # Reset episodio
        state = env.reset()
        done = False
        total_reward = 0

        while not done:

            # Prende i valori Q dello stato corrente
            q_values = get_q_values(Q, state)

            # Sceglie l'azione migliore
            action = int(np.argmax(q_values))

            # Esegue l'azione e aggiorna lo stato
            state, reward, done = env.step(action)

            # Accumula reward
            total_reward += reward

        # Salva metriche episodio
        rewards.append(total_reward)
        foods.append(env.food_eaten)
        steps_list.append(env.steps)

        # Collisione con il fantasma
        if env.pacman == env.ghost:
            collisions += 1

    return (
        np.mean(rewards),
        np.mean(foods),
        np.mean(steps_list),
        collisions / episodes
    )


def plot_bar(labels, values, ylabel, title, filename):
    """
    Crea un grafico a barre per confrontare gli agenti.
    """

    plt.figure()
    plt.bar(labels, values)
    plt.ylabel(ylabel)
    plt.title(title)

    # Salva il grafico nella cartella results
    plt.savefig(
        os.path.join(results_dir, filename),
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()


def save_all_results(metrics):
    """
    Salva i risultati finali sia in formato JSON sia in formato CSV.

    Il JSON viene creato direttamente qui.
    Il CSV viene salvato usando la funzione save_results_to_csv.
    """

    results_data = {}

    for item in metrics:

        # Crea una chiave leggibile per ogni agente
        agent_key = item["Agent"].lower().replace("-", "_")

        # Salva le metriche convertendole in float
        results_data[agent_key] = {
            "average_reward": float(item["Avg Reward"]),
            "average_food_eaten": float(item["Avg Food"]),
            "average_steps": float(item["Avg Steps"]),
            "collision_rate": float(item["Collision Rate"])
        }

    # Salvataggio JSON
    with open(os.path.join(results_dir, "results.json"), "w") as f:
        json.dump(results_data, f, indent=4)

    # Salvataggio CSV
    save_results_to_csv("comparison_results.csv", metrics)


def main():
    """
    Funzione principale del progetto.

    Esegue in ordine:
    1. training Q-Learning
    2. training SARSA
    3. grafici di training
    4. valutazione Random, Q-Learning e SARSA
    5. grafici di confronto
    6. salvataggio risultati
    7. visualizzazione grafica delle policy
    8. gioco manuale
    """

    print("\n========== TRAINING Q-LEARNING ==========\n")

    # Addestramento Q-Learning
    Q_qlearning, rewards_qlearning, q0_qlearning = train_qlearning()

    print("\n========== TRAINING SARSA ==========\n")

    # Addestramento SARSA
    Q_sarsa, rewards_sarsa, q0_sarsa = train_sarsa()

    # Grafico training Q-Learning
    plot_training_results(
        "Q-learning Training",
        rewards_qlearning,
        q0_qlearning
    )

    # Grafico training SARSA
    plot_training_results(
        "SARSA Training",
        rewards_sarsa,
        q0_sarsa
    )

    print("\n========== EVALUATION ==========\n")

    # Valutazione agente random
    random_score, random_food, random_steps, random_collision = (
        evaluate_random_agent(episodes=100)
    )

    # Valutazione agente Q-Learning
    qlearning_score, qlearning_food, qlearning_steps, qlearning_collision = (
        evaluate_trained_agent(Q_qlearning, episodes=100)
    )

    # Valutazione agente SARSA
    sarsa_score, sarsa_food, sarsa_steps, sarsa_collision = (
        evaluate_trained_agent(Q_sarsa, episodes=100)
    )

    # Stampa risultati numerici
    print("Random agent average reward:", random_score)
    print("Q-learning average reward:", qlearning_score)
    print("SARSA average reward:", sarsa_score)

    print("\nRandom agent average food eaten:", random_food)
    print("Q-learning average food eaten:", qlearning_food)
    print("SARSA average food eaten:", sarsa_food)

    print("\nRandom average steps:", random_steps)
    print("Q-learning average steps:", qlearning_steps)
    print("SARSA average steps:", sarsa_steps)

    print("\nRandom collision rate:", random_collision)
    print("Q-learning collision rate:", qlearning_collision)
    print("SARSA collision rate:", sarsa_collision)

    # Lista unica con tutte le metriche
    metrics = [
        {
            "Agent": "Random",
            "Avg Reward": random_score,
            "Avg Food": random_food,
            "Avg Steps": random_steps,
            "Collision Rate": random_collision
        },
        {
            "Agent": "Q-learning",
            "Avg Reward": qlearning_score,
            "Avg Food": qlearning_food,
            "Avg Steps": qlearning_steps,
            "Collision Rate": qlearning_collision
        },
        {
            "Agent": "SARSA",
            "Avg Reward": sarsa_score,
            "Avg Food": sarsa_food,
            "Avg Steps": sarsa_steps,
            "Collision Rate": sarsa_collision
        }
    ]

    # Etichette comuni per i grafici a barre
    labels = ["Random", "Q-learning", "SARSA"]

    # Confronto reward media
    plot_bar(
        labels,
        [random_score, qlearning_score, sarsa_score],
        "Average Reward",
        "Performance Comparison",
        "performance_comparison.png"
    )

    # Confronto cibo mangiato
    plot_bar(
        labels,
        [random_food, qlearning_food, sarsa_food],
        "Average Food Eaten",
        "Food Eaten Comparison",
        "food_comparison.png"
    )

    # Confronto numero medio di step
    plot_bar(
        labels,
        [random_steps, qlearning_steps, sarsa_steps],
        "Average Steps",
        "Survival Steps Comparison",
        "steps_comparison.png"
    )

    # Confronto collision rate
    plot_bar(
        labels,
        [random_collision, qlearning_collision, sarsa_collision],
        "Collision Rate",
        "Collision Rate Comparison",
        "collision_comparison.png"
    )

    # Salva risultati in JSON e CSV
    save_all_results(metrics)

    # Visualizza graficamente policy Q-Learning
    visualize_policy(
        Q_qlearning,
        "Q-learning Pac-Man",
        max_steps=100,
        delay=400
    )

    # Visualizza graficamente policy SARSA
    visualize_policy(
        Q_sarsa,
        "SARSA Pac-Man",
        max_steps=100,
        delay=400
    )

    # Permette all'utente di giocare manualmente
    manual_play()


# Esegue main solo se questo file viene lanciato direttamente
if __name__ == "__main__":
    main()