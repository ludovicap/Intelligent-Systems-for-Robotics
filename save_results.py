import csv
import os


def save_results_to_csv(filename, results):
    """
    Salva i risultati in un file CSV.

    Parametri:
    - filename: nome del file CSV da creare
    - results: lista di dizionari contenente le metriche degli agenti
    """

    # Crea la cartella "results" se non esiste già
    os.makedirs("results", exist_ok=True)

    # Percorso completo del file da salvare
    filepath = os.path.join("results", filename)

    # Prende i nomi delle colonne dalle chiavi del primo dizionario
    keys = results[0].keys()

    # Apre il file CSV in scrittura
    with open(filepath, mode="w", newline="") as file:

        # Crea un writer che scrive dizionari nel CSV
        writer = csv.DictWriter(file, fieldnames=keys)

        # Scrive la prima riga con i nomi delle colonne
        writer.writeheader()

        # Scrive tutte le righe dei risultati
        writer.writerows(results)

    # Messaggio di conferma
    print(f"Risultati salvati in {filepath}")