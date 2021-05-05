from algoritmos import algoritmos_supervisados
from time import sleep

def importar_dataset():
    pass

def main():
    opcion = ''
    while (opcion != '7'):

        print('ALGORITMOS')
        print('1. Logistic Regression')
        print('2. Nearest Neighbor')
        print('3. Support Vector Machines ')
        print('4. Naive Bayes')
        print('5. Decision Tree Algorithm')
        print('6. Random Forest Classification')
        print('7. Salir')

        opcion = input('Digite una opción:')

        if opcion == '1':
            algoritmos_supervisados.logistic_regression()
        elif opcion == '2':
            algoritmos_supervisados.nearest_neighbor()
        elif opcion == '3':
            algoritmos_supervisados.support_vector_machine()
        elif opcion == '4':
            algoritmos_supervisados.naive_bayes()
        elif opcion == '5':
            algoritmos_supervisados.decision_tree()
        elif opcion == '6':
            algoritmos_supervisados.random_forest_classification()
        elif opcion == '7':
            print('Saliendo...')
            sleep(2)
        else:
            print("Digite una opcion valida")

if __name__ == "__main__":
    main()