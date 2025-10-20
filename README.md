# Soccer Project

Riorganizzazione del progetto universitario: raccolta di script e moduli per analisi statistiche sul calcio (Serie A / Premier, ecc.).

Struttura consigliata
- src/soccer_project/: codice riutilizzabile (data access, utils, analysis)
- scripts/: script eseguibili di esempio
- notebooks/: Jupyter notebook (senza output salvati)
- Data/: file di dati (mantenuti qui o con istruzioni per scaricarli)

Esempio rapido
1. Crea un ambiente virtuale:
   python -m venv .venv
   source .venv/bin/activate  # Windows: .venv\Scripts\activate

2. Installa dipendenze:
   pip install -r requirements.txt

3. Esegui uno script di esempio:
   python scripts/run_analysis.py --example regressioni

Note sui dati
I file in Data/ possono essere grandi. Se preferisci non tenerli nel repo, valuta di spostarli in uno storage esterno e lasciare in Data/ solo file d'esempio e uno script per il download.

Autore: Canisa99 (progetto universitario)