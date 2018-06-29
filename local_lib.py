from sklearn.model_selection import StratifiedKFold


import numpy as np

from sklearn.model_selection import train_test_split
import sklearn

import pandas
import yaml




# Il modulo carica un dataframe in memoria e gestisce una sessione di 
# esperiemnti usando il k-fold.
# 
# La procedura e' pensata per esperimenti piccoli, in quanto tutto e' fatto
# in memoria.
#
# Si suppone di avere un dataframe cosi' fatto
#
#               features
#            -----------------
#           |                 | 
#           |                 |
# campioni  |                 |
#           |                 |
#           |                 |
#           |                 |
#           ------------------
#
# Delle features una o piu' sono quelle di uscita
# 
# La funzione dividiDataset genera una lista di liste; ogni 
# elemento della lista <i> e' una lista di indici corrisponenti agli
# elementi del fold <i>
# 
#
#
#






def leggeParametri(filePar):
    """
    Legge il file yaml con i parametri

    :param nomeFilePar:
    :return:    Lista delle colonne del file di dati da leggere come input
                Nome della colonna da usare per output

                Nome del file di dati in input (dataframe pandas pickle)
                Postfisso del nome del file output
                K parametro del k-fold (generalmente 10)

    """
    # carica i valori da passare ai programmi
    with open(filePar, 'r') as ymlfile:
        cfg = yaml.load(ymlfile)

    l_input = cfg["parEsp"]["l_input"]
    l_output = cfg["parEsp"]["l_output"]


    fileDati = cfg["file"]["inputFile"]
    fileOut = cfg["file"]["outputFile"]

    k = cfg["parEsp"]["k"]

    return l_input, l_output, fileDati, fileOut, k




#...............................................................................
def nomeFileOut(filePar):
    _, _, _, fileOut, _ = leggeParametri(filePar)

    return filePar+"_"+fileOut








############################################################################
# def filtraDataFrame(dd1):
#     """
#     Per filtrare il dataframe richiama la funzione da local_lib
#
#     :param dd1: dataframe da filtrare
#     :param filePar:
#     :return: dataframe filtrato
#     """
#     import local_lib as lib
#
#     df = lib.filtraDatframe(dd1)
#
#     return df
############################################################################

#...............................................................................
def loadDataframe(filePar):
     """
     Carica il dataframe ed esegue, se necessario, un preprocessing
     :param filePar: file dei parametri del programma
     :return: dataframe pandas
     """
     _, _, nomeFile, _ , k=  leggeParametri(filePar)
     dd = pandas.read_pickle(nomeFile)
     #.....................
     #df=filtraDataFrame(dd)
     df=dd
     #.....................
     return df


#...............................................................................
def loadData(dataframe, col):
    """
    Carica le colonne del dataframe dalla lista col.
    Il dataframe contiene un vettore per ogni riga
    e trasforma l'array di array risultante in una matrice
    :param dataframe:
    :param col: lista di colonne da caricare
    :return: matrice corrispondente
    """

    # trasforma in matrice il primo blocco di dati,
    # nella colonna di etichetta col[0]
    dd = dataframe[col[0]].values
    Nr = len(dd)
    Nc = len(dd[0])
    dd = np.concatenate(dd).reshape((Nr, Nc))

    # trasforma in matrice gli altri blocchi e li concatena
    for c in range(1, len(col)):
        m = dataframe[col[c]].values
        Nr = len(m)
        Nc = len(m[0])
        m = np.concatenate(m).reshape((Nr, Nc))

        dd = np.concatenate((dd, m), axis=1)

    return dd


def normalizzaMatrice(M):
    """
    Normalizza la matrice dividendo per il massimo sulla riga
    cioe' per il valore della massima feature del campione consderato
    L'idea e' quella di rendere i valori nei vettori di rappresentazione
    delle sequenze indipendenti dalla lunghezza della sequenza stessa
    :param M: matrice da normalizzare
    :return: matrice normalizzata
    """
    out = M / M.max(axis=1).reshape(len(M), 1)
    return out


def loadDataNorm(dataframe, col):
    """
    Carica dal dataframe la matrice dei dati e la normalizza dividendo
    ogni riga per il massimo della riga stessa
    :param dataframe: sorgente dati
    :param col: lista di colonne da caricare
    :return: matrice normalizzata
    """
    mm = loadData(dataframe, col)
    # print "LoadDataNorm", mm.shape
    return normalizzaMatrice(mm)


#...............................................................................
def dividiDataset(dataframe, filePar):
    """ 
    Divide il <dataframe> in input per fare il k-fold.
    
    dataframe : contiene in inCol le feature di input alla rete
                in outCol i valori di output (o le classi)

    Restituisce :

    fold :  una struttura dati che contiene la lista degli indici
            per ogni fold
            Per il fold i si avra':

            fold[i]["train"] = <lista di indici di riga relativa al
                                train del fold i>
            fold[i]["test"] = <lista di indici di riga relativa al test
                                del fold i>


    l'uso e'con

        inCol="<colonna di vettori di features>"
        outCol="<label o valore di output>"

    """

    inCol, outCol, _, _, k =  leggeParametri(filePar)


    # selezioni i dati X: ingresso, Y: uscita
    #X=dataframe[inCol].values

    # carico unasto la funzione loadData perche' la colonna del
    # dataframe contiene il vettore di rappresentazione per la riga
    X=loadDataNorm(dataframe, inCol)
    # carico direettamente la colonna di output perche' contiene
    # la classe espressa come numero naturale
    Y=dataframe[outCol].values

    # trasforma Y da [XXX, 1] a [XXX,] ????
    Y=np.reshape(Y, [len(Y),])

    # esegue la divisione per il kfold con mescolamento
    skf = StratifiedKFold(n_splits=k, shuffle=True)

    fold=[]
    for train, test in skf.split(X, Y):

        v={"train": train, "test": test}
        fold.append(v)

    return fold





#...............................................................................
def prepara_ambiente(filePar):
    """
    Prepara la esecuzione del programma
    generando il dataframe con i dati di input e la lista
    delle righe che costituiscono ogni fold.

    :param filePar:
    :return: dd : dataframe
            fold : lista delle righe per ogni fold

    """


    dd = loadDataframe(filePar)

    # divide l'insieme di indici nei fold
    fold = dividiDataset(dd, filePar)


    return dd, fold





################################################################################
################################################################################
################################################################################
################################################################################
################################################################################


#...............................................................................
def estraiDati(listaIndici, dataframe, etichettaInput, etichettaOutput, norma):
    """
    Estare i dati dal dataframe e prepara le matrici per la'ddestramento

    :param listaIndici: indici delle righe dei dati da estrarre
    :param dataframe: nome del dataframe
    :param etichettaInput: lista delle etichette delle feature da usare
    :param etichettaOutput: etichetta dell output
    :param norma: boolean definisce se l'input deve essere normalizzato
                    (normalizzaione sul massimo della riga)

    :return: X e Y matrici di ingresso e di uscita
    """

    # seleziona le righe in listaIndici e crea un altro dataframe
    # che sara' trasformato in matrice
    Xdf=dataframe.iloc[listaIndici]

    # carico unasto la funzione loadData perche' la colonna del
    # dataframe contiene il vettore di rappresentazione per la riga
    if norma == True:
        X=loadDataNorm(Xdf, etichettaInput)
    else:
        X = loadData(Xdf, etichettaInput)

    Y=dataframe.ix[listaIndici, etichettaOutput].values

    return X, Y


#
# #...............................................................................
# def elimina_NaN(X_IN, Y_IN):
#     """
#     Elimina i NaN dalle matrici in ingresso
#
#     :param X_train_IN:
#     :param Y_train_IN:
#
#
#     :return: stessa serie di matrici ma eliminando le righe che contengono NaN
#     """
#
#     # elimina gli elementi NaN dal training e dal test
#     # LL[0] contiene l'indice delle righe da eliminare
#     LL = np.argwhere(np.isnan(X_IN))
#     canc = LL[:, 0]
#
#     #print "primo valore", X_IN[canc[0]]
#
#     X_ = np.delete(X_IN, canc, axis=0)
#     Y_ = np.delete(Y_IN, canc, axis=0)
#
#
#
#
#     #print " posti dove ci snon Nan", canc
#
#
#     return X_, Y_

