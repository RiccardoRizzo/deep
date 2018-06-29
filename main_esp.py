#
# QUESTO FILE FA GIRARE TUTTO L'ESPERIMENTO
#


import local_lib
import run_kfold as re

import sys

import h5py

import time
import yaml


def main(filePar):

    # leggo il file della rete neurale come stringa
    with open(filePar, 'r') as ymlfile:
        cfg = yaml.load(ymlfile)
        pathNetwork = cfg["Alg"]["path"]

        nomeFileNetwork = cfg["Alg"]["file"]

    nomeFileRete=pathNetwork + nomeFileNetwork +".py"
    print nomeFileRete
    # leggo il file della rete neurale come stringa
    fin = open(nomeFileRete, "r")
    file_rete = fin.readlines()
    fin.close()

    # leggo il file dei parametri come stringa
    fin = open(filePar, "r")
    parametri = fin.readlines()
    fin.close()


    # Genera il nome del file di output
    nFOut = local_lib.nomeFileOut(filePar)

    # apre il file
    with h5py.File(nFOut) as f:
        # inserisce le informazioni nel file dei risultati
        infor = main.__doc__

        f.attrs["info"] =  "nuovo file"


        #=======================================================
        # scrive le stringhe del file della rete e dei parametri
        f.create_dataset("File_rete", data=file_rete)
        f.create_dataset("Parametri_passati", data=parametri)
        # =======================================================

    print "carica il dataset e crea la lista di indici del k-fold"
    df, fold = local_lib.prepara_ambiente(filePar)

    # esegue il test ottenendo le medie
    # dei parametri di output
    print "eseguo l'esperimento"

    ### tempo di run <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    start_time = time.time()

    M_acc, M_prec, M_rec = re.test(df, fold, filePar )

    ### TEMPO DI run <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    runtime = (time.time() - start_time)

    # scrivere i risultati in un file
    nFOut = local_lib.nomeFileOut(filePar)

    # apre il file
    with h5py.File(nFOut, "a") as f:
        f.create_dataset("Tempo totale di run (in sec)", data=[runtime])
        f.create_dataset("Media Accuracy", data=[M_acc])
        f.create_dataset("Media Precision", data=[M_prec])
        f.create_dataset("Media Recall", data=[M_rec])



    print  M_acc, M_prec, M_rec

    # mandare la email

if __name__=="__main__":

    main(sys.argv[1])

