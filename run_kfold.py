
import local_lib as ll
import numpy as np
import sklearn

import yaml

import h5py

import time
import keras

#...............................................................................
def test(df, fold, filePar):
    """
    :param df:             dataframe da fornire per il test
    :return:
    """

    acc=[]
    prec=[]
    rec=[]

    with open(filePar, 'r') as ymlfile:
        cfg = yaml.load(ymlfile)

        # carico le classi
        l_classi = cfg["parEsp"]["l_classi"]


    for i in range(len(fold)):


        y_pred , Y_test, y_out, runtime = run_fold(df, fold, filePar, i)

        acc_fold = sklearn.metrics.accuracy_score(Y_test, y_pred, normalize=True,  sample_weight=None)
        prec_fold = sklearn.metrics.precision_score(Y_test, y_pred, average="micro")
        rec_fold = sklearn.metrics.recall_score(Y_test, y_pred, average="micro")
        confusion_fold = sklearn.metrics.confusion_matrix(Y_test, y_pred, labels = l_classi)

        acc.append(acc_fold)
        prec.append(prec_fold)
        rec.append(rec_fold)

        # Scrive i primi dati dell'esperimento in un file
        nFOut = ll.nomeFileOut(filePar)

        # apre il file
        with h5py.File(nFOut, "a") as f:

            fold_string = "fold="+str(i)
            grp=f.create_group(fold_string)

            # salva in un file i risultati del fold
            indici = list(fold[i]["test"])
            grp.create_dataset("indici", data=[indici])
            grp.create_dataset("predizioni", data=[y_pred])
            grp.create_dataset("valore vero",data=[Y_test])
            grp.create_dataset("valore output", data=[y_out])

            grp.create_dataset("tempo di addestramento (in sec)", data=runtime)
            grp.create_dataset("accuracy", data=acc_fold)
            grp.create_dataset("precision", data=prec_fold)
            grp.create_dataset("recall", data=rec_fold)

            grp.create_dataset("confusion", data = confusion_fold)

    M_acc=np.average(acc)
    M_prec=np.average(prec)
    M_rec=np.average(rec)

    return M_acc, M_prec, M_rec



#...............................................................................
def run_fold(df, fold, filePar, i):


    import sys


    # legge dal file di parametri il nome del file con la rete
    with open(filePar, 'r') as ymlfile:
        cfg = yaml.load(ymlfile)

        # carico le classi
        l_classi = cfg["parEsp"]["l_classi"]

        # # CARICO I PARAMETRI DELLA RETE
        input_shape = cfg["parAlg"]["input_shape"]

        num_output = cfg["parAlg"]["dl2_units_num"]


        validation_split = cfg["parAlg"]["validation_split"]
        # learning_rate = cfg["parAlg"]["learning_rate"]
        batch_size = cfg["parAlg"]["batch_size"]
        epochs = cfg["parAlg"]["epochs"]
        norma = cfg["parAlg"]["norma"]

        pathNetwork =  cfg["Alg"]["path"]
        sys.path.append( pathNetwork)

        nomeFileNetwork = cfg["Alg"]["file"]

    # importa la libreria delle reti neurali
    riga = "import " + nomeFileNetwork + " as NN"
    exec(riga)  # importa il file con la rete


    l_features, l_output, _, _, _ =ll.leggeParametri(filePar)

    # seleziona i dati di ingresso e di uscita
    X_train, Y_train = ll.estraiDati(fold[i]["train"], df, l_features, l_output, norma)
    X_test, Y_test = ll.estraiDati(fold[i]["test"], df, l_features, l_output, norma)


    model = NN.Net_f(filePar)


    ### TEMPO DI ADDESTRAMENTO <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    start_time = time.time()


    # esegue il training
    X=np.reshape(X_train, (len(X_train) , input_shape[0], input_shape[1] ) )

    # trasforma Y da [XXX, 1] a [XXX,] ????
    Y = np.reshape(Y_train, (len(Y_train) ,) )
    # trasforma il numero della classe (classeID) in categoria
    Y = keras.utils.to_categorical(Y, num_output)


    model.fit(X, Y, validation_split=validation_split, batch_size=batch_size, epochs=epochs, verbose =0)
    # "TEMPO DI ADDESTRAMENTO <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
    runtime = (time.time() - start_time)

    # PREDIZIONI =============================================================
    # genera l'output del modello
    X = np.reshape(X_test, (len(X_test), input_shape[0], input_shape[1] ) )
    y_out = model.predict(X)

    y_pred = [ np.argmax(x) for x in y_out ]
    y_pred = [l_classi[i] for i in y_pred]
    # confronto con la soglia
    #y_pred = y_out > dl2_soglia
    #y_pred.astype(np.int)
    # =======================================================================

    # trasforma Y da [XXX, 1] a [XXX,] ????
    Y_test = list( np.reshape(Y_test, [len(Y_test), ]) )
    Y_test = [l_classi[i] for i in Y_test]
    # trasforma il numero della classe (classeID) in categoria
    #Y_test = keras.utils.to_categorical(Y_test, dl2_units_num)

    #y_pred = np.reshape(y_pred, [len(y_pred), ])

    #print "y_pred", y_pred
    #print "Y_test", Y_test
    #print "y_out", y_out

    return y_pred, Y_test, y_out, runtime

