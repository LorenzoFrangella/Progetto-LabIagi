# GESTURE RECOGNITION WITH RECCURRENT NEURAL NETWORKS

Questo progetto è stato svolto per implementare un riconoscitore di gesti che vengono forniti in input tramite uno stream video in input.

Per implementare questo riconoscitore è stata usata una rete neurale di tipo ricorrente. Questo tipo di reti neurali può essere impiegato per tipi di input che sono formati da sequenze temporali.
In questo caso il nostro input essendo uno stream video, i singoli elementi del nostro input saranno i frame del nostro video. 

## Come viene fornito l'input

Il video viene inizialmente processato tramite mediapipe per estrarre i punti dello scheletro, i punti sono definiti nell immagine sottostante

