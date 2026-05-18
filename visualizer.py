import pygame
import numpy as np
from environment import PacManEnv


# Lista delle azioni possibili:
# 0 = su
# 1 = giù
# 2 = sinistra
# 3 = destra
ACTIONS = [0, 1, 2, 3]


def get_q_values(Q, state):
    """
    Restituisce i valori Q associati a uno stato.

    Se lo stato non esiste ancora nella Q-table,
    viene inizializzato con quattro valori pari a zero,
    uno per ogni azione possibile.
    """
    if state not in Q:
        Q[state] = np.zeros(len(ACTIONS))

    return Q[state]


def load_assets():
    """
    Carica immagini e suoni del gioco.

    Le immagini vengono lette dalla cartella sprites/.
    I suoni vengono letti dalla cartella sounds/.
    """

    # Caricamento immagini con trasparenza
    pac_open = pygame.image.load("sprites/pacman_open.png").convert_alpha()
    pac_closed = pygame.image.load("sprites/pacman_closed.png").convert_alpha()
    ghost = pygame.image.load("sprites/ghost.png").convert_alpha()
    food = pygame.image.load("sprites/food.png").convert_alpha()

    # Ridimensionamento immagini
    pac_open = pygame.transform.scale(pac_open, (58, 58))
    pac_closed = pygame.transform.scale(pac_closed, (58, 58))
    ghost = pygame.transform.scale(ghost, (58, 58))
    food = pygame.transform.scale(food, (18, 18))

    # Dizionario dei suoni
    sounds = {}

    # Prova a caricare i suoni.
    # Se non li trova, il gioco continua comunque senza audio.
    try:
        sounds["waka"] = pygame.mixer.Sound("sounds/waka.wav")
        sounds["eat"] = pygame.mixer.Sound("sounds/eat.wav")
        sounds["game_over"] = pygame.mixer.Sound("sounds/game_over.wav")
    except pygame.error:
        sounds["waka"] = None
        sounds["eat"] = None
        sounds["game_over"] = None

    return pac_open, pac_closed, ghost, food, sounds


def rotate_pacman(img, action):
    """
    Ruota l'immagine di Pac-Man in base alla direzione.

    L'immagine originale è orientata verso destra.
    """

    if action == 0:      # su
        return pygame.transform.rotate(img, 90)

    if action == 1:      # giù
        return pygame.transform.rotate(img, -90)

    if action == 2:      # sinistra
        return pygame.transform.rotate(img, 180)

    # destra: immagine originale
    return img


def draw_maze(screen, env, cell, offset_y, width):
    """
    Disegna la griglia di gioco e i muri.

    screen: finestra pygame
    env: ambiente PacManEnv
    cell: dimensione di una cella in pixel
    offset_y: spazio superiore riservato al titolo/info
    width: larghezza finestra
    """

    rows, cols = env.rows, env.cols

    # Sfondo nero
    screen.fill((0, 0, 0))

    # Rettangolo principale del labirinto
    maze_rect = pygame.Rect(0, offset_y, cols * cell, rows * cell)
    pygame.draw.rect(screen, (0, 0, 10), maze_rect)

    # Disegna tutte le celle della griglia
    for r in range(rows):
        for c in range(cols):
            rect = pygame.Rect(
                c * cell,
                offset_y + r * cell,
                cell,
                cell
            )

            # Colore interno cella
            pygame.draw.rect(screen, (5, 5, 18), rect)

            # Bordino sottile della cella
            pygame.draw.rect(screen, (12, 12, 35), rect, 1)

    # Colori dei muri
    border_color = (0, 80, 255)
    highlight = (80, 180, 255)

    # Bordo esterno del labirinto
    pygame.draw.rect(screen, border_color, maze_rect, 10)

    # Evidenziazione interna del bordo
    pygame.draw.rect(screen, highlight, maze_rect.inflate(-10, -10), 2)

    # Disegna i muri definiti nell'ambiente
    for (r1, c1), (r2, c2) in env.walls:

        # Coordinate pixel della prima cella
        x1, y1 = c1 * cell, offset_y + r1 * cell

        # Coordinate pixel della seconda cella
        x2, y2 = c2 * cell, offset_y + r2 * cell

        # Se le due celle sono sulla stessa riga,
        # allora il muro è verticale tra le due celle.
        if r1 == r2:
            x = max(x1, x2)
            y = y1

            pygame.draw.line(
                screen,
                border_color,
                (x, y + 10),
                (x, y + cell - 10),
                14
            )

            pygame.draw.line(
                screen,
                highlight,
                (x, y + 12),
                (x, y + cell - 12),
                3
            )

        # Se le due celle sono sulla stessa colonna,
        # allora il muro è orizzontale tra le due celle.
        elif c1 == c2:
            x = x1
            y = max(y1, y2)

            pygame.draw.line(
                screen,
                border_color,
                (x + 10, y),
                (x + cell - 10, y),
                14
            )

            pygame.draw.line(
                screen,
                highlight,
                (x + 12, y),
                (x + cell - 12, y),
                3
            )


def draw_game(
    screen, env, cell, width, height,
    steps, max_steps, total_reward,
    font, big_font,
    pac_open, pac_closed, ghost_img, food_img,
    last_action, anim_frame, title
):
    """
    Disegna l'intera schermata di gioco:
    - labirinto
    - titolo
    - reward
    - cibo
    - fantasma
    - Pac-Man
    - eventuale messaggio finale
    """

    # Spazio superiore per titolo e statistiche
    offset_y = 70

    # Disegna il labirinto
    draw_maze(screen, env, cell, offset_y, width)

    # Barra nera superiore
    pygame.draw.rect(screen, (0, 0, 0), (0, 0, width, 70))

    # Titolo
    title_text = big_font.render(title, True, (255, 255, 0))
    screen.blit(title_text, (20, 8))

    # Informazioni dell'episodio
    info = font.render(
        f"TOTAL REWARD: {total_reward}    FOOD: {env.food_eaten}    STEP: {steps}/{max_steps}",
        True,
        (255, 255, 255)
    )
    screen.blit(info, (20, 42))

    # Disegna il cibo
    fr, fc = env.food
    screen.blit(food_img, (fc * cell + 41, offset_y + fr * cell + 41))

    # Disegna il fantasma
    gr, gc = env.ghost
    screen.blit(ghost_img, (gc * cell + 21, offset_y + gr * cell + 21))

    # Alterna immagine di Pac-Man con bocca aperta/chiusa
    # per creare una piccola animazione.
    if (anim_frame // 8) % 2 == 0:
        pac = pac_open
    else:
        pac = pac_closed

    # Ruota Pac-Man nella direzione dell'ultima azione
    pac = rotate_pacman(pac, last_action)

    # Disegna Pac-Man
    pr, pc = env.pacman
    screen.blit(pac, (pc * cell + 21, offset_y + pr * cell + 21))

    # Messaggio finale se:
    # - Pac-Man viene preso dal fantasma
    # - oppure si raggiunge il numero massimo di step
    if env.pacman == env.ghost or steps >= max_steps:

        # Overlay semi-trasparente sopra il gioco
        overlay = pygame.Surface((width, height), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 120))
        screen.blit(overlay, (0, 0))

        if env.pacman == env.ghost:
            message = "GAME OVER"
            color = (255, 0, 0)
        else:
            message = "EPISODE COMPLETE"
            color = (255, 255, 0)

        final_text = big_font.render(message, True, color)

        # Posiziona il messaggio al centro
        screen.blit(
            final_text,
            (
                width // 2 - final_text.get_width() // 2,
                height // 2 - final_text.get_height() // 2
            )
        )


def animate_pause(
    screen, env, cell, width, height,
    steps, max_steps, total_reward,
    font, big_font,
    pac_open, pac_closed, ghost, food,
    last_action, anim_frame, title, delay
):
    """
    Crea una pausa animata tra una mossa e l'altra.

    Invece di bloccare semplicemente il programma,
    ridisegna il gioco più volte, così Pac-Man continua
    ad avere l'animazione della bocca.
    """

    # Numero di frame da mostrare durante la pausa
    frames = max(1, delay // 40)

    for _ in range(frames):

        # Gestione chiusura finestra
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return anim_frame, False

        # Ridisegna il gioco
        draw_game(
            screen, env, cell, width, height,
            steps, max_steps, total_reward,
            font, big_font,
            pac_open, pac_closed, ghost, food,
            last_action, anim_frame, title
        )

        # Aggiorna la finestra
        pygame.display.flip()

        # Pausa di 40 ms tra un frame e l'altro
        pygame.time.delay(40)

        # Avanza il contatore dell'animazione
        anim_frame += 1

    return anim_frame, True


def visualize_policy(Q, title="Pac-Man RL", max_steps=100, delay=400):
    """
    Visualizza graficamente una policy già appresa.

    La policy viene ricavata dalla Q-table:
    per ogni stato si sceglie l'azione con valore Q massimo.
    """

    # Inizializzazione audio e pygame
    pygame.mixer.pre_init(44100, -16, 1, 512)
    pygame.init()
    pygame.mixer.init()

    # Creo ambiente e stato iniziale
    env = PacManEnv()
    state = env.reset()

    # Dimensioni grafiche
    cell = 100
    width = env.cols * cell
    height = env.rows * cell + 70

    # Creo finestra
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption(title)

    # Carico immagini e suoni
    pac_open, pac_closed, ghost, food, sounds = load_assets()

    # Font per testi
    font = pygame.font.SysFont("Arial", 18, bold=True)
    big_font = pygame.font.SysFont("Arial", 28, bold=True)

    # Variabili dell'episodio
    steps = 0
    total_reward = 0
    done = False
    last_action = 3
    anim_frame = 0
    running = True
    food_before = env.food_eaten

    while running:

        # Se episodio non finito, applico la policy
        if not done and steps < max_steps:

            # Prendo i valori Q dello stato attuale
            q = get_q_values(Q, state)

            # Scelgo l'azione migliore secondo la Q-table
            action = int(np.argmax(q))
            last_action = action

            # Eseguo azione nell'ambiente
            state, reward, done = env.step(action)

            # Suono movimento
            if sounds["waka"]:
                sounds["waka"].play()

            # Se Pac-Man ha mangiato cibo, suono eat
            if env.food_eaten > food_before:
                food_before = env.food_eaten
                if sounds["eat"]:
                    sounds["eat"].play()

            # Se Pac-Man è stato preso, suono game over
            if done and env.pacman == env.ghost and sounds["game_over"]:
                sounds["game_over"].play()

            # Aggiorno reward e step
            total_reward += reward
            steps += 1

            # Mostro animazione tra una mossa e l'altra
            anim_frame, running = animate_pause(
                screen, env, cell, width, height,
                steps, max_steps, total_reward,
                font, big_font,
                pac_open, pac_closed, ghost, food,
                last_action, anim_frame, title, delay
            )

        else:
            # Quando l'episodio è finito, mostro schermata finale
            anim_frame, running = animate_pause(
                screen, env, cell, width, height,
                steps, max_steps, total_reward,
                font, big_font,
                pac_open, pac_closed, ghost, food,
                last_action, anim_frame, title, 2000
            )

            # Dopo la pausa finale chiudo la simulazione
            running = False

    pygame.quit()


def manual_play():
    """
    Permette di giocare manualmente con le frecce della tastiera.
    """

    # Inizializzazione audio e pygame
    pygame.mixer.pre_init(44100, -16, 1, 512)
    pygame.init()
    pygame.mixer.init()

    # Creo e resetto ambiente
    env = PacManEnv()
    env.reset()

    # Dimensioni finestra
    cell = 100
    width = env.cols * cell
    height = env.rows * cell + 70

    # Creo finestra pygame
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("Manual Pac-Man")

    # Carico immagini e suoni
    pac_open, pac_closed, ghost, food, sounds = load_assets()

    # Font
    font = pygame.font.SysFont("Arial", 18, bold=True)
    big_font = pygame.font.SysFont("Arial", 28, bold=True)

    # Variabili di gioco
    steps = 0
    total_reward = 0
    done = False
    last_action = 3
    anim_frame = 0
    running = True
    food_before = env.food_eaten

    while running:

        # Nessuna azione finché non viene premuto un tasto
        action = None

        # Lettura eventi pygame
        for event in pygame.event.get():

            # Chiusura finestra
            if event.type == pygame.QUIT:
                running = False

            # Lettura tasti freccia
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    action = 0
                elif event.key == pygame.K_DOWN:
                    action = 1
                elif event.key == pygame.K_LEFT:
                    action = 2
                elif event.key == pygame.K_RIGHT:
                    action = 3

        # Se è stata premuta una freccia e il gioco non è finito
        if action is not None and not done:

            last_action = action

            # Eseguo la mossa manuale
            _, reward, done = env.step(action)

            # Suono movimento
            if sounds["waka"]:
                sounds["waka"].play()

            # Suono cibo mangiato
            if env.food_eaten > food_before:
                food_before = env.food_eaten
                if sounds["eat"]:
                    sounds["eat"].play()

            # Suono game over
            if done and env.pacman == env.ghost and sounds["game_over"]:
                sounds["game_over"].play()

            # Aggiorno reward totale e numero step
            total_reward += reward
            steps += 1

        # Disegno la schermata corrente
        draw_game(
            screen, env, cell, width, height,
            steps, env.max_steps, total_reward,
            font, big_font,
            pac_open, pac_closed, ghost, food,
            last_action, anim_frame, "Manual Pac-Man"
        )

        # Aggiorno finestra
        pygame.display.flip()

        # Piccola pausa per non aggiornare troppo velocemente
        pygame.time.delay(40)

        # Avanzo animazione bocca Pac-Man
        anim_frame += 1

    pygame.quit()