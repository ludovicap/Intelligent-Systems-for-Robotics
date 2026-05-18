import random
import numpy as np
from environment import PacManEnv


def train_qlearning(
    episodes=10000,
    alpha=0.1,
    gamma=0.95,
    epsilon=0.1
):
    """
    Algoritmo Q-Learning classico.

    Parametri:
    - episodes: numero di episodi di addestramento
    - alpha: learning rate, cioè quanto aggiorno la Q-table
    - gamma: discount factor, cioè quanto considero importanti le ricompense future
    - epsilon: probabilità di esplorazione nella politica epsilon-greedy
    """

    # Creo l'ambiente Pac-Man
    env = PacManEnv()

    # Azioni possibili:
    # 0 = up
    # 1 = down
    # 2 = left
    # 3 = right
    actions = [0, 1, 2, 3]

    # Q-table:
    # è un dizionario in cui ogni stato ha associato un vettore di 4 valori Q
    Q = {}

    def get_q_values(state):
        """
        Restituisce i valori Q associati a uno stato.

        Se lo stato non è ancora presente nella Q-table,
        viene inizializzato con valori Q tutti pari a zero.
        """
        if state not in Q:
            Q[state] = np.zeros(len(actions))

        return Q[state]

    def choose_action(state):
        """
        Politica epsilon-greedy.

        Con probabilità epsilon sceglie un'azione casuale,
        cioè esplora.

        Con probabilità 1 - epsilon sceglie l'azione migliore
        secondo la Q-table, cioè sfrutta ciò che ha imparato.
        """

        # Esplorazione
        if random.random() < epsilon:
            return random.choice(actions)

        # Sfruttamento
        q_values = get_q_values(state)
        return int(np.argmax(q_values))

    # Lista dove salvo la ricompensa totale di ogni episodio
    rewards_per_episode = []

    # Lista dove salvo il valore Q massimo dello stato iniziale
    q0_values = []

    # Ciclo principale sugli episodi
    for episode in range(episodes):

        # Reset dell'ambiente all'inizio di ogni episodio
        state = env.reset()

        # Salvo lo stato iniziale per monitorare come evolve il suo valore Q
        initial_state = state

        # Ricompensa totale accumulata nell'episodio
        total_reward = 0

        # Flag che indica se l'episodio è finito
        done = False

        # Ciclo interno: continua finché l'episodio non termina
        while not done:

            # Scelgo un'azione con politica epsilon-greedy
            action = choose_action(state)

            # Eseguo l'azione nell'ambiente
            next_state, reward, done = env.step(action)

            # Valori Q dello stato attuale
            q_values = get_q_values(state)

            # Valori Q dello stato successivo
            next_q_values = get_q_values(next_state)

            # Aggiornamento Q-Learning classico:
            #
            # Q(s,a) = Q(s,a) + alpha * [
            #          reward + gamma * max Q(s',a') - Q(s,a)
            # ]
            q_values[action] = q_values[action] + alpha * (
                reward + gamma * np.max(next_q_values) - q_values[action]
            )

            # Passo allo stato successivo
            state = next_state

            # Accumulo la reward dell'episodio
            total_reward += reward

        # Salvo la reward totale dell'episodio
        rewards_per_episode.append(total_reward)

        # Salvo il valore Q massimo dello stato iniziale
        q0_values.append(np.max(get_q_values(initial_state)))


        # Stampa di controllo ogni 500 episodi
        if (episode + 1) % 500 == 0:
            print(
                f"Q-learning Episode {episode + 1}/{episodes}, "
                f"Reward: {total_reward}"
            )

    # Restituisco:
    # - Q-table allenata
    # - ricompense per episodio
    # - andamento del valore dello stato iniziale
    return Q, rewards_per_episode, q0_values