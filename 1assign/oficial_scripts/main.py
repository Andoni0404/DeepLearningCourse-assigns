import warnings
from optuna_optimize import run_optimization  # ¡Directo, sin el src.!

warnings.filterwarnings("ignore", category=UserWarning)

if __name__ == "__main__":
    print("Iniciando el pipeline de Deep Learning...")
    
    study = run_optimization()
    
    print("\n¡Búsqueda completada!")