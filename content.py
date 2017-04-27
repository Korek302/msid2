# --------------------------------------------------------------------------
# ------------  Metody Systemowe i Decyzyjne w Informatyce  ----------------
# --------------------------------------------------------------------------
#  Zadanie 2: k-NN i Naive Bayes
#  autorzy: A. Gonczarek, J. Kaczmar, S. Zareba
#  2017
# --------------------------------------------------------------------------

from __future__ import division
import numpy as np


def hamming_distance(X, X_train):
    """
    :param X: zbior porownwanych obiektow N1xD
    :param X_train: zbior obiektow do ktorych porownujemy N2xD
    Funkcja wyznacza odleglosci Hamminga obiektow ze zbioru X od
    obiektow X_train. ODleglosci obiektow z jednego i drugiego
    zbioru zwrocone zostana w postaci macierzy
    :return: macierz odleglosci pomiedzy obiektami z X i X_train N1xN2
    """
    N1 = X.shape[0]
    N2 = X_train.shape[0]
    
    X = X.toarray()
    X_train = X_train.toarray()
    
    def hamming(A, B):
        m = 0
        for i in range(0, len(A)):
            if(A[i] != B[i]):
                m=m+1
                
        return m
    
    m = np.zeros(shape=(N1, N2))
    for i in range(0, N1):
        for j in range(0, N2):
            m[i][j] = hamming(X[i], X_train[j])
    
    return m


def sort_train_labels_knn(Dist, y):
    """
    Funkcja sortujaca etykiety klas danych treningowych y
    wzgledem prawdopodobienstw zawartych w macierzy Dist.
    Funkcja zwraca macierz o wymiarach N1xN2. W kazdym
    wierszu maja byc posortowane etykiety klas z y wzgledem
    wartosci podobienstw odpowiadajacego wiersza macierzy
    Dist
    :param Dist: macierz odleglosci pomiedzy obiektami z X
    i X_train N1xN2
    :param y: wektor etykiet o dlugosci N2
    :return: macierz etykiet klas posortowana wzgledem
    wartosci podobienstw odpowiadajacego wiersza macierzy
    Dist. Uzyc algorytmu mergesort.
    """
    
    N1 = len(Dist[:,0])
    N2 = len(y)

    w = np.zeros(shape=(N1,N2))
    for i in range(0, N1):
        w[i,:] = y[Dist[i,:].argsort(kind='mergesort')]
        
    return w


def p_y_x_knn(y, k):
    """
    Funkcja wyznacza rozklad prawdopodobienstwa p(y|x) dla
    kazdej z klas dla obiektow ze zbioru testowego wykorzystujac
    klasfikator KNN wyuczony na danych trenningowych
    :param y: macierz posortowanych etykiet dla danych treningowych N1xN2
    :param k: liczba najblizszuch sasiadow dla KNN
    :return: macierz prawdopodobienstw dla obiektow z X
    """
    N1 = len(y)
    
    numOf1 = 0
    numOf2 = 0
    numOf3 = 0
    numOf4 = 0
    
    w = np.zeros(shape=(N1, 4))
    for i in range(0, N1):
        for j in range(0, k):
            if(y[i][j] == 1):
                numOf1 = numOf1 + 1    
            if(y[i][j] == 2):
                numOf2 = numOf2 + 1
            if(y[i][j] == 3):
                numOf3 = numOf3 + 1
            if(y[i][j] == 4):
                numOf4 = numOf4 + 1
                
        w[i][0] = numOf1/k
        w[i][1] = numOf2/k
        w[i][2] = numOf3/k
        w[i][3] = numOf4/k
        numOf1 = 0
        numOf2 = 0
        numOf3 = 0
        numOf4 = 0
    return w


def classification_error(p_y_x, y_true):
    """
    Wyznacz blad klasyfikacji.
    :param p_y_x: macierz przewidywanych prawdopodobienstw
    :param y_true: zbior rzeczywistych etykiet klas 1xN.
    Kazdy wiersz macierzy reprezentuje rozklad p(y|x)
    :return: blad klasyfikacji
    """
    
    N1 = len(y_true)
    
    a = np.zeros(shape = (N1))
    for i in range(0, N1):
        for j in range(0, 4):
            if(j == 0 and p_y_x[i][j] >= a[i]):
                a[i] = 1
            if(j == 1 and p_y_x[i][j] >= a[i]):
                a[i] = 2
            if(j == 2 and p_y_x[i][j] >= a[i]):
                a[i] = 3
            if(j == 3 and p_y_x[i][j] >= a[i]):
                a[i] = 4
    
    w = 0            
    for i in range(0, N1):
        if(a[i] != y_true[i]):
            w = w + 1
            
    return w/N1
    
    """
    numOf1 = 0
    numOf2 = 0
    numOf3 = 0
    numOf4 = 0
    
    for i in range(0, N1):
        if(y_true[i] == 1):
            numOf1 = numOf1 + 1    
        if(y_true[i] == 2):
            numOf2 = numOf2 + 1
        if(y_true[i] == 3):
            numOf3 = numOf3 + 1
        if(y_true[i] == 4):
            numOf4 = numOf4 + 1
    
    prob1 = numOf1/N1
    prob2 = numOf2/N1
    prob3 = numOf3/N1
    prob4 = numOf4/N1
    
    w = 0
    for i in range(0, N1):
        for j in range(0, 4):
            if((j == 0 and p_y_x[i][j] == prob1)or
              (j == 1 and p_y_x[i][j] == prob2)or
              (j == 2 and p_y_x[i][j] == prob3)or
              (j == 3 and p_y_x[i][j] == prob4)):
                w = w + 1
                
    
    
    w = 1 - w/N1
    return w
    """
           
    


def model_selection_knn(Xval, Xtrain, yval, ytrain, k_values):
    """
    :param Xval: zbior danych walidacyjnych N1xD
    :param Xtrain: zbior danych treningowych N2xD
    :param yval: etykiety klas dla danych walidacyjnych 1xN1
    :param ytrain: etykiety klas dla danych treningowych 1xN2
    :param k_values: wartosci parametru k, ktore maja zostac sprawdzone
    :return: funkcja wykonuje selekcje modelu knn i zwraca krotke (best_error,best_k,errors),
    gdzie best_error to najnizszy
    osiagniety blad, best_k - k dla ktorego blad byl najnizszy, 
        errors - lista wartosci bledow dla kolejnych k z k_values
    """
    N = len(k_values)
    
    best_error = np.inf
    best_k = np.inf
    errors = []
    
    
    
    distances = hamming_distance(Xval, Xtrain)
    
    sorted_dist = sort_train_labels_knn(distances, ytrain)
    
    for i in range(0, N):
        curr_k = k_values[i]
        curr_err = classification_error(p_y_x_knn(sorted_dist, curr_k), yval)
        
        errors[i] = curr_err
        if(curr_err < best_error):
            best_error = curr_err
            best_k = curr_k
            
    print( (best_error, best_k, errors))
    return (best_error, best_k, errors)



def estimate_a_priori_nb(ytrain):
    """
    :param ytrain: etykiety dla dla danych treningowych 1xN
    :return: funkcja wyznacza rozklad a priori p(y) i zwraca p_y -
    wektor prawdopodobienstw a priori 1xM
    """
    N = len(ytrain)
    
    numOf1 = 0
    numOf2 = 0
    numOf3 = 0
    numOf4 = 0
    
    for i in range(0, N):
        if(ytrain[i] == 1):
            numOf1 = numOf1 + 1
        if(ytrain[i] == 2):
            numOf2 = numOf2 + 1
        if(ytrain[i] == 3):
            numOf3 = numOf3 + 1
        if(ytrain[i] == 4):
            numOf4 = numOf4 + 1
            
    w = np.zeros(shape=(4))
    w[0] = numOf1/N
    w[1] = numOf2/N
    w[2] = numOf3/N
    w[3] = numOf4/N
     
    return w


def estimate_p_x_y_nb(Xtrain, ytrain, a, b):
    """
    :param Xtrain: dane treningowe NxD
    :param ytrain: etykiety klas dla danych treningowych 1xN
    :param a: parametr a rozkladu Beta
    :param b: parametr b rozkladu Beta
    :return: funkcja wyznacza rozklad prawdopodobienstwa p(x|y) zakladajac, 
    ze x przyjmuje wartosci binarne i ze elementy
    x sa niezalezne od siebie. Funkcja zwraca macierz p_x_y o wymiarach MxD.
    """
    Xtrain = Xtrain.toarray()
    D = len(Xtrain[0])
    N = len(ytrain)
    
    
    numOf1 = 0
    numOf2 = 0
    numOf3 = 0
    numOf4 = 0
    
    for i in range(0, N):
        if(ytrain[i] == 1):
            numOf1 = numOf1 + 1
        if(ytrain[i] == 2):
            numOf2 = numOf2 + 1
        if(ytrain[i] == 3):
            numOf3 = numOf3 + 1
        if(ytrain[i] == 4):
            numOf4 = numOf4 + 1
            
    sum1 = np.zeros(shape=(4))
    sum1[0] = numOf1
    sum1[1] = numOf2
    sum1[2] = numOf3
    sum1[3] = numOf4
        
        
    numOf1_2 = 0
    numOf2_2 = 0
    numOf3_2 = 0
    numOf4_2 = 0    
        
    for i in range(0, N):
        for j in range(0, D):
            if(ytrain[i] == 1 and Xtrain[i][j] == 1):
                numOf1_2 = numOf1_2 + 1
            if(ytrain[i] == 2 and Xtrain[i][j] == 1):
                numOf2_2 = numOf2_2 + 1
            if(ytrain[i] == 3 and Xtrain[i][j] == 1):
                numOf3_2 = numOf3_2 + 1
            if(ytrain[i] == 4 and Xtrain[i][j] == 1):
                numOf4_2 = numOf4_2 + 1
                
    sum2 = np.zeros(shape = (4))            
    sum2[0] = numOf1_2
    sum2[1] = numOf2_2
    sum2[2] = numOf3_2
    sum2[3] = numOf4_2      
        
    print(sum1)
    
    result = np.zeros(shape = (4, D))
    for i in range(0, 4):
        for j in range(0, D):
            result[i][j] = (sum2[i] + a - 1)/(sum1[i] + a + b - 2) 
    
    return result


def p_y_x_nb(p_y, p_x_1_y, X):
    """
    :param p_y: wektor prawdopodobienstw a priori o wymiarach 1xM
    :param p_x_1_y: rozklad prawdopodobienstw p(x=1|y) - macierz MxD
    :param X: dane dla ktorych beda wyznaczone prawdopodobienstwa, macierz NxD
    :return: funkcja wyznacza rozklad prawdopodobienstwa p(y|x) dla kazdej z klas z wykorzystaniem klasyfikatora Naiwnego
    Bayesa. Funkcja zwraca macierz p_y_x o wymiarach NxM.
    """
    pass


def model_selection_nb(Xtrain, Xval, ytrain, yval, a_values, b_values):
    """
    :param Xtrain: zbior danych treningowych N2xD
    :param Xval: zbior danych walidacyjnych N1xD
    :param ytrain: etykiety klas dla danych treningowych 1xN2
    :param yval: etykiety klas dla danych walidacyjnych 1xN1
    :param a_values: lista parametrow a do sprawdzenia
    :param b_values: lista parametrow b do sprawdzenia
    :return: funkcja wykonuje selekcje modelu Naive Bayes - wybiera najlepsze wartosci parametrow a i b. Funkcja zwraca
    krotke (error_best, best_a, best_b, errors) gdzie best_error to najnizszy
    osiagniety blad, best_a - a dla ktorego blad byl najnizszy, best_b - b dla ktorego blad byl najnizszy,
    errors - macierz wartosci bledow dla wszystkich par (a,b)
    """
    pass
