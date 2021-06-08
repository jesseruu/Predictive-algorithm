from time import sleep
import logisticRegression
import nearestNeigbor
import supportVectorMachine
import naiveBayes
import randomForest
import decisionTree

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
            logisticRegression.train_logist()

        elif opcion == '2':
            nearestNeigbor.neigh_neigbors()

        elif opcion == '3':
            supportVectorMachine.support_vectoM()

        elif opcion == '4':
            naiveBayes.naive_bayes()

        elif opcion == '5':
            decisionTree.decision_tree()

        elif opcion == '6':
            randomForest.random_forestC()
       
        elif opcion == '7':
            print('Saliendo...')
            sleep(1)
            
        else:
            print("Digite una opcion valida")

if __name__ == "__main__":
    main()