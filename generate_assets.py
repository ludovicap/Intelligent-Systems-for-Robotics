import os
import math
import wave
import struct
import pygame

os.makedirs("sprites", exist_ok=True)
os.makedirs("sounds", exist_ok=True)

pygame.init()

# PACMAN OPEN
surface = pygame.Surface((100, 100), pygame.SRCALPHA)
pygame.draw.circle(surface, (255, 255, 0), (50, 50), 40)
pygame.draw.polygon(surface, (0, 0, 0, 0), [(50, 50), (95, 25), (95, 75)])
pygame.image.save(surface, "sprites/pacman_open.png")

# PACMAN CLOSED
surface = pygame.Surface((100, 100), pygame.SRCALPHA)
pygame.draw.circle(surface, (255, 255, 0), (50, 50), 40)
pygame.image.save(surface, "sprites/pacman_closed.png")

# FOOD
surface = pygame.Surface((30, 30), pygame.SRCALPHA)
pygame.draw.circle(surface, (255, 220, 120), (15, 15), 6)
pygame.image.save(surface, "sprites/food.png")

# GHOST
surface = pygame.Surface((100, 100), pygame.SRCALPHA)
pygame.draw.rect(surface, (255, 60, 60), (20, 40, 60, 35))
pygame.draw.circle(surface, (255, 60, 60), (50, 40), 30)

pygame.draw.circle(surface, (255, 255, 255), (38, 40), 8)
pygame.draw.circle(surface, (255, 255, 255), (62, 40), 8)
pygame.draw.circle(surface, (0, 0, 255), (38, 40), 4)
pygame.draw.circle(surface, (0, 0, 255), (62, 40), 4)

pygame.image.save(surface, "sprites/ghost.png")

pygame.quit()


def create_tone(filename, frequencies, duration=0.15, volume=0.4):
    sample_rate = 44100
    n_samples = int(sample_rate * duration)

    with wave.open(filename, "w") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)

        for i in range(n_samples):
            t = i / sample_rate
            freq = frequencies[min(int(i / n_samples * len(frequencies)), len(frequencies) - 1)]
            value = int(volume * 32767 * math.sin(2 * math.pi * freq * t))
            wav.writeframes(struct.pack("<h", value))


create_tone("sounds/waka.wav", [500, 750, 500, 750], duration=0.12)
create_tone("sounds/eat.wav", [900, 1200, 1500], duration=0.18)
create_tone("sounds/game_over.wav", [400, 250, 120], duration=0.45)

print("Assets creati: sprites/ e sounds/")