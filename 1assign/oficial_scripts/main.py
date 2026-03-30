from optuna_optimize import run_optimization

def main():
    print("Iniciando pipeline de Machine Learning para Seguros...")
    # Ejecuta la búsqueda de hiperparámetros de Optuna
    study = run_optimization()
    print("¡Proceso terminado con éxito!")

if __name__ == "__main__":
    main()