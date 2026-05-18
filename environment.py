import random

class PacManEnv:
    def __init__(self):
        # Dimensioni della griglia (5x5)
        self.rows = 5
        self.cols = 5

        # Numero massimo di mosse per episodio
        self.max_steps = 100

        # Posizioni iniziali (verranno definite nel reset)
        self.pacman = None
        self.ghost = None
        self.food = None

        # Contatori
        self.steps = 0
        self.food_eaten = 0

        # Muri tra celle (non si può passare tra queste coppie)
        self.walls = [
            ((1, 1), (1, 2)),
            ((2, 3), (3, 3)),
            ((3, 0), (3, 1))
        ]

    def reset(self):
        """
        Resetta l'ambiente all'inizio di un nuovo episodio
        """

        # Posizione iniziale Pac-Man (alto sinistra)
        self.pacman = [0, 0]

        # Posizione iniziale fantasma (basso destra)
        self.ghost = [4, 4]

        # Genera il cibo in posizione casuale
        self.food = self.spawn_food()

        # Reset contatori
        self.steps = 0
        self.food_eaten = 0

        # Restituisce lo stato iniziale
        return self.get_state()

    def spawn_food(self):
        """
        Genera il cibo in una posizione casuale valida
        (diversa da Pac-Man e dal fantasma)
        """
        while True:
            r = random.randint(0, self.rows - 1)
            c = random.randint(0, self.cols - 1)

            # Evita di spawnare sopra Pac-Man o fantasma
            if [r, c] != self.pacman and [r, c] != self.ghost:
                return [r, c]

    def get_state(self):
        """
        Restituisce lo stato come tupla:
        (pacman_r, pacman_c, food_r, food_c, ghost_r, ghost_c)
        """
        return (
            self.pacman[0], self.pacman[1],
            self.food[0], self.food[1],
            self.ghost[0], self.ghost[1]
        )

    def is_wall_between(self, pos1, pos2):
        """
        Controlla se c'è un muro tra due celle
        """
        p1 = tuple(pos1)
        p2 = tuple(pos2)

        # Il muro vale in entrambe le direzioni
        return (p1, p2) in self.walls or (p2, p1) in self.walls

    def step(self, action):
        """
        Esegue una mossa:
        0 = up
        1 = down
        2 = left
        3 = right
        """

        # Incrementa numero di passi
        self.steps += 1

        # Reward base (penalità per ogni passo)
        reward = -1
        done = False

        # Distanza dal cibo prima della mossa
        old_distance = self.distance_to_food()

        # Posizione attuale Pac-Man
        pr, pc = self.pacman

        # Calcolo nuova posizione in base all'azione
        if action == 0:      # su
            new_pos = [pr - 1, pc]
        elif action == 1:    # giù
            new_pos = [pr + 1, pc]
        elif action == 2:    # sinistra
            new_pos = [pr, pc - 1]
        elif action == 3:    # destra
            new_pos = [pr, pc + 1]
        else:
            new_pos = [pr, pc]  # azione non valida

        # Controllo validità movimento
        if (
            0 <= new_pos[0] < self.rows and
            0 <= new_pos[1] < self.cols and
            not self.is_wall_between(self.pacman, new_pos)
        ):
            # Movimento valido
            self.pacman = new_pos
        else:
            # Movimento non valido (muro o fuori griglia)
            reward = -5

        # Distanza dopo la mossa
        new_distance = self.distance_to_food()

        # Reward shaping: incentiva ad avvicinarsi al cibo
        if new_distance < old_distance:
            reward += 1
        elif new_distance > old_distance:
            reward -= 1

        # Se Pac-Man mangia il cibo
        if self.pacman == self.food:
            reward += 10
            self.food_eaten += 1

            # Genera nuovo cibo
            self.food = self.spawn_food()

        # Muove il fantasma
        self.move_ghost()

        # Se il fantasma prende Pac-Man
        if self.pacman == self.ghost:
            reward = -50
            done = True

        # Se raggiunto numero massimo di passi
        if self.steps >= self.max_steps:
            done = True

        # Restituisce stato, reward e flag di fine episodio
        return self.get_state(), reward, done

    def distance_to_food(self):
        """
        Distanza Manhattan tra Pac-Man e cibo
        """
        return abs(self.pacman[0] - self.food[0]) + abs(self.pacman[1] - self.food[1])

    def move_ghost(self):
        """
        Movimento del fantasma:
        - 70% insegue Pac-Man
        - 30% movimento casuale
        """
        if random.random() < 0.7:
            self.move_ghost_towards_pacman()
        else:
            self.move_ghost_random()

    def move_ghost_towards_pacman(self):
        """
        Movimento "intelligente" verso Pac-Man
        """
        gr, gc = self.ghost
        pr, pc = self.pacman

        possible_moves = []

        # Movimento verticale verso Pac-Man
        if pr > gr:
            possible_moves.append([gr + 1, gc])
        elif pr < gr:
            possible_moves.append([gr - 1, gc])

        # Movimento orizzontale verso Pac-Man
        if pc > gc:
            possible_moves.append([gr, gc + 1])
        elif pc < gc:
            possible_moves.append([gr, gc - 1])

        # Filtra solo mosse valide
        valid_moves = [
            pos for pos in possible_moves
            if (
                0 <= pos[0] < self.rows and
                0 <= pos[1] < self.cols and
                pos != self.food and
                not self.is_wall_between(self.ghost, pos)
            )
        ]

        # Sceglie una mossa valida
        if valid_moves:
            self.ghost = random.choice(valid_moves)

    def move_ghost_random(self):
        """
        Movimento casuale del fantasma
        """
        gr, gc = self.ghost

        possible_moves = [
            [gr - 1, gc],
            [gr + 1, gc],
            [gr, gc - 1],
            [gr, gc + 1]
        ]

        # Filtra mosse valide
        valid_moves = [
            pos for pos in possible_moves
            if (
                0 <= pos[0] < self.rows and
                0 <= pos[1] < self.cols and
                pos != self.food and
                not self.is_wall_between(self.ghost, pos)
            )
        ]

        # Movimento casuale tra quelli validi
        if valid_moves:
            self.ghost = random.choice(valid_moves)
