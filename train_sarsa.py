import random
import numpy as np
from environment import PacManEnv


def train_sarsa(
    episodes=10000,
    alpha=0.1,
    gamma=0.95,
    epsilon=0.1
):
    """
    Algoritmo SARSA classico.

    Parametri:
    - episodes: numero di episodi di addestramento
    - alpha: learning rate
    - gamma: discount factor
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

    # Q-table: dizionario stato -> valori Q delle azioni
    Q = {}

    def get_q_values(state):
        """
        Restituisce i valori Q di uno stato.

        Se lo stato non è ancora nella Q-table,
        lo inizializza con quattro valori pari a zero.
        """
        if state not in Q:
            Q[state] = np.zeros(len(actions))

        return Q[state]

    def choose_action(state):
        """
        Politica epsilon-greedy.

        Con probabilità epsilon sceglie un'azione casuale.
        Con probabilità 1 - epsilon sceglie l'azione migliore.
        """

        # Esplorazione
        if random.random() < epsilon:
            return random.choice(actions)

        # Sfruttamento
        q_values = get_q_values(state)
        return int(np.argmax(q_values))

    # Ricompensa totale per ogni episodio
    rewards_per_episode = []

    # Valore Q massimo dello stato iniziale per ogni episodio
    q0_values = []

    # Ciclo sugli episodi
    for episode in range(episodes):

        # Reset ambiente
        state = env.reset()

        # Salvo stato iniziale
        initial_state = state

        # In SARSA l'azione viene scelta subito,
        # prima di entrare nel ciclo dell'episodio
        action = choose_action(state)

        total_reward = 0
        done = False

        while not done:

            # Eseguo l'azione scelta nello stato corrente
            next_state, reward, done = env.step(action)

            # Scelgo la prossima azione usando ancora epsilon-greedy
            next_action = choose_action(next_state)

            # Valori Q dello stato corrente
            q_values = get_q_values(state)

            # Valori Q dello stato successivo
            next_q_values = get_q_values(next_state)

            # Aggiornamento SARSA:
            #
            # Q(s,a) = Q(s,a) + alpha * [
            #          reward + gamma * Q(s',a') - Q(s,a)
            # ]
            #
            # La differenza con Q-Learning è che SARSA usa
            # Q(s', a'), cioè il valore della prossima azione scelta
            # realmente dalla politica epsilon-greedy.
            q_values[action] = q_values[action] + alpha * (
                reward + gamma * next_q_values[next_action] - q_values[action]
            )

            # Aggiorno stato e azione
            state = next_state
            action = next_action

            # Accumulo la reward dell'episodio
            total_reward += reward

        # Salvo la reward totale
        rewards_per_episode.append(total_reward)

        # Salvo il valore massimo dello stato iniziale
        q0_values.append(np.max(get_q_values(initial_state)))

        # Stampa di controllo ogni 500 episodi
        if (episode + 1) % 500 == 0:
            print(
                f"SARSA Episode {episode + 1}/{episodes}, "
                f"Reward: {total_reward}"
            )

    return Q, rewards_per_episode, q0_values