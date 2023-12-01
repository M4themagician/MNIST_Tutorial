### PYTHON SETUP ###
"""
Am besten miniconda auf dem Server installieren! Dann kann man ganz entspannt Python Umgebungen verwalten. 
    1.1. Auf den server ssh'en (Putty + username@mercurius.math.uni-bremen.de, X-forwarding aktivieren)
    1.2. Lokaler Speicher liegt auf /localdata, also am besten sowas machen wie
        cd /localdata
        mkdir username
        cd username
    1.3. Ordner für Miniconda anlegen
        mkdir miniconda
        cd miniconda
    1.4. Miniconda runterladen (https://docs.conda.io/en/latest/miniconda.html)
        wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
    1.5. ... und installieren
        bash Miniconda3-latest-Linux-x86_64.sh
    1.6. Nach akzeptieren der Lizenzbedingungen fragt er, ob der default install pfad korrekt ist:
        Miniconda3 will now be installed into this location:
        /some/path/miniconda3

        - Press ENTER to confirm the location
        - Press CTRL-C to abort the installation
        - Or specify a different location below
    
    Stattdessen dann
        [/some/path/miniconda3] >>>  /localdata/username/miniconda/miniconda3

    1.7. Nach install einmal checken ob alles geklappt hat:
        which python
    sollte '/localdata/username/miniconda/miniconda3/bin/python' zurückgeben.
"""

### PYTORCH SETUP ###
"""
    2.1. Für jede pytorch Version, die benutzt wird eine eigene python 3.8 environment anlegen (aktuell ist pytorch2.1):
        conda create --name pytorch2.1
    2.2. Aktivieren via
        conda activate pytorch2.1
    2.3. Pytorch installieren! (https://pytorch.org/get-started/locally/)
        pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    2.4. Schauen, ob's geklappt hat:
        python                              # python starten
        import torch                        # wenn das keinen Fehler gibt, hat's geklappt
        x = torch.rand((10,10,10))          # irgendeinen 10x10x10 Tensor samplen
        print(torch.cuda.is_available())    # wenn das True ist, werden die GPUs auch erkannt!
        x = x.cuda()                        # unseren Tensor in GPU speicher schieben
        quit()                              # zurück ins terminal
    2.5. Matplotlib installieren:
        pip install matplotlib
"""

### Tutorial ###
"""
Ab hier geht dann das Tutorial-Script los. Kannst gerne nach und nach die Kommentare wegnehmen und das ganze ausführen
"""

from torch.optim.sgd import SGD


if __name__ == '__main__':
    ################################################################################################################################
    ###### Getting the MNIST Dataset
    """
    Erstmal ein paar Pakete importieren
    """
    user = 'Evers'                                                       # pytorch
    from torchvision.datasets import MNIST                              # Datensatzklasse für MNIST, handled sogar das runterladen
    import os                                                           # für shellcommands aus python raus
    import matplotlib.pyplot as plt                                     # für visualisierung


    mnist_root = f'/localdata/{user}/Datasets/MNIST'                    # Da wollen wir den Datensatz mal hinhaben...
    os.makedirs(mnist_root, exist_ok=True)                              # Erstellen vom MNIST Ordner und aller Ordner auf dem Weg. Sehr nützlich!

    """
    Ein Datensatz in pytorch hat zwei Merkmale:
        1. Es gibt eine __len__() Methode, die die Anzahl der Elemente des Datensatzes angibt und
        2. Es gibt eine __getitem_(int i) Methode, die das i'te Element zurückgibt
    """
    dataset = MNIST(mnist_root, download=True, train=True)                          # Datensatzobjekt, lädt Datensatz runter
    print(f'Länge des MNIST Datensatzes ist {dataset.__len__()}')
    image, label = dataset.__getitem__(0)
    plt.imshow(image)                                                   # Bildplot erstellen
    plt.show()                                                          # und anzeigen, wenn du kein xforwarding laufen hast, gibt's hier ne Warnung
    print(f'Das Label des ersten Bildes ist {label}')

    ################################################################################################################################
    ###### Ausgabeformat anpassen
    """
    Das Ausgabeformat des Datensatzes ist gerade (PIL.image, int) (PIL = Python imaging library). 
    Damit ein von uns implementiertes Netzwerk die Daten verstehen kann, müssen sie in einen torch.tensor transformiert werden.
    Der Datensatz kann dadurch im Konstruktor mit Transformationen versehen werden, die beim Aufrufen von getitem ausgeführt werden.
    """
    import torchvision.transforms as transforms
    dataset = MNIST(mnist_root, download=True, transform=transforms.ToTensor())

    """
    Während des Trainingsvorganges werden immer Teile des Datensatzes (sogenannte Batches) durch das Netzwerk geschickt.
    Um effizient Daten zu laden definieren wir einen Dataloader.
    """
    from torch.utils.data import DataLoader
    BATCH_SIZE = 128
    NUM_WORKERS = 8
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, drop_last=True)
    """
    Um das Ausgabeformat zu checken, lassen wir uns die ersten 10 Batches ausgeben:
    """
    running_index = 0
    for (images, labels) in dataloader:
        print(images.size(), labels)                # images.size() gibt die Dimensionen des Imagetensors aus
        running_index += 1
        if running_index > 9:
            break
    """
    images ist jetzt ein BATCH_SIZE x 1 x 28 x 28 Tensor, also BATCH_SIZE Bilder zusammengeschweißt in der vierten Batch-dimension
    labels ist jetzt ein BATCH_SIZE großer Vektor mit dem label des i'ten Bildes als i'ten Eintrag
    """

    ################################################################################################################################
    ###### Defining a neural Network
    """
    Um ein Bild des MNIST Datensatzes zu klassifizieren, definieren wir eine Abbildung, 
    die jedes 28x28px Bild auf einen Wahrscheinlichkeitsvektor der Zahlen 0 bis 9 abbildet. 
    Dies geschieht in Form ein CNNs (convolutional neural networks). Das Bild wird also durch Faltungen auf einen Tensor abgebildet,
    von dem aus eine Lineare Abbildung in den Raum der Wahrscheinlichkeitsmaße auf 0 bis 9 abbildet.

    Neuronale Netze in PyTorch erben von der torch.nn.Module Klasse und besitzen eine forward(tensor x) Methode, welche den Tensor x
    entgegennimmt und die Netzwerkausgabe zurückgibt.

    Unser neuronales Netz besteht aus den Bausteinen
        Conv2d:             Faltung mit einem 2-d Kern. Stell dir den Kern als Matrix vor, die Pixelweise über das Bild geschoben wird.
                            Für jeden Pixel wird die Summe des Elementweisen Produktes mit der Matrix und den entsprechenden Bildeinträgen gebildet.
                            Diese Werte sind das Ausgabearray der Faltung.
        ReLU:               Rectified-linear-unit. ReLU(x) = max(0, x). Dies ist die Aktivierungsfunktion oder Nichtlinearität unserer Netzwerklayer.
        MaxPool2d:          Nimm von jeweils 2x2 Werten im Array den Größten und gebe das so resultierende, in beiden Dimensionen halb so große array aus.
        Flatten:            Wandle einen 2D Tensor in einen Vektor (aka 1D Tensor) um.
        Linear:             Lineare Abbildung vom R^m nach R^n. m und n kann man natürlich wählen.
    """
    import torch.nn as nn

    class Network_block(nn.Module):
        def __init__(self, in_channels, out_channels, kernel_size = 3, do_maxpool = True):
            """
            Comment
            """
            super(Network_block, self).__init__() 
            padding = kernel_size//2
            self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding)
            self.activation1 = nn.ReLU()
            self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding)
            self.activation2 = nn.ReLU()
            self.do_maxpool = do_maxpool
            if do_maxpool:
                self.maxpool = nn.MaxPool2d(kernel_size=(2,2), stride=2)
            else:
                self.flatten = nn.Flatten()
        def forward(self, x):
            x = self.conv1(x)
            x = self.activation1(x)
            x = self.conv2(x)
            x = self.activation2(x)
            if self.do_maxpool:
                x = self.maxpool(x)
            else:
                x = self.flatten(x)
            return x

    class MNIST_Classifier(nn.Module):
        """
        Dies ist ein neuronales Netz!
        """
        def __init__(self, output_dimension = 10, channels = [16, 32, 64], linear_features = 10000):
            """
            In der __init__() Methode wird das Netz erstellt. Da wir von der nn.Module Klasse erben wird zunächst der super-Konstruktor aufgerufen.
            Anschließend definieren wir die Bausteine der Abbildung des Netzwerks.
            """
            super(MNIST_Classifier, self).__init__() 
            self.channels = channels
            self.block1 = Network_block(1, channels[0])
            self.block2 = Network_block(channels[0], channels[1])
            self.block3 = Network_block(channels[1], channels[2], do_maxpool=False)

            self.linear1 = nn.Sequential(nn.Linear(7*7*channels[-1], linear_features), nn.ReLU())
            self.linear2 = nn.Linear(linear_features, output_dimension)

        def forward(self, x):
            """
            Dies ist die methode, welche das Netzwerk bei einem gegebenen Tensor auswertet!
            Man sieht, warum die Bezeichnung feedforward neural network üblich ist...
            """
            # x.size() == BATCH_SIZEx1x28*28
            x = self.block1(x) # x.size() == BATCH_SIZExself.channels[0]x14*14
            x = self.block2(x) # x.size() == BATCH_SIZExself.channels[1]x7*7
            x = self.block3(x) # KEIN MAXPOOL!!! x.size() == BATCH_SIZExself.channels[2]x7*7 ---> BATCH_SIZE x self.channels[2]*7*7 Vektor
            x = self.linear1(x)
            x = self.linear2(x)
            return x

    """
    Aufgabe: Die Netztwerkdefinition ist so nicht schön. Besser wäre es, klar geblockte Bausteine (z.B. conv+Relu+conv+Relu+maxpool) in einem Untermodul zu definieren.
    Implementiere eine Klasse 'conv_block', die genau das macht und definiere die Bausteine des Netzes dann über diese klasse.
    Aufgabe: Warum 3136 Eingangskanäle in der linear layer?
    Aufgabe: Wie groß sind die Tensoren nach jeder Faltung? 
    Aufgabe: Füge eine weitere Aktivierungsfunktion und Lineare Abbildung ein, bevor der Ausgabevektor berechnet wird.
    """

    ################################################################################################################################
    ###### Testing the network
    """
    Nach der Klassendefinition können wir das Netz als Objekt der Klasse erstellen und einmal einen Zufallstensor durch das Netz schicken.
    """
    import torch
    net = MNIST_Classifier()                        # hier wird ein Objekt der Klasse erstellt und die __init__ Methode aufgerufen.
    tensor = torch.rand((10, 1, 28, 28))
    out = net(tensor)                               # hier wird die forward methode des netzes mit dem tensor ausgeführt.
    print(out.size())
    ################################################################################################################################
    ###### GPU Handling
    """
    Wenn wir eine CUDA fähige Grafikkarte erkennen, können wir das Netzwerk auf die GPU schieben. Falls wir mehr als eine GPU haben,
    müssen wir dem Netz sagen, wie es damit umgehen soll. Da alles auf einer Maschine läuft, bietet sich sogenannter Datenparallelismus an:
    D.h. Das Netzwerk wird auf jede GPU gespiegelt und Netzauswertungen finden parallel auf jeer Grafikkarte statt.
    """
    if torch.cuda.is_available():
        print(f'Cuda available, running on {torch.cuda.device_count()} GPUs!')
        net.cuda()
        if torch.cuda.device_count() > 1:
            net = nn.DataParallel(net)

    ################################################################################################################################
    ###### Training the network
    """
    Jetzt geht es darum, die ganzen Parameter oder Gewichte des Netzwerkes so zu wählen, dass das Netz auf dem MNIST Datensatz die Zahlen klassifiziert.
    Dazu braucht es:
        Einen Optimierungsalgorithmus.
        Eine Zielfunktion.
        Ein Training Loop.

        Als Optimierungsalgorithmus nehmen wir einfach das Gradientenverfahren. Da Gradienten nur auf einem Teil des Datensatzes berechnet werden, heißt das Verfahren
        hier 'stochastic gradient descent' oder SGD. Die Optimierungsvariablen sind die Netzwerkparameter. 
        Die Schrittweite (oder bei dein KI Leuten 'Lernrate') des Algorithmus wählen wir als 0.02.
    """
    import torch.optim as opt
    optimizer = opt.SGD(net.parameters(), lr=0.02)
    """
    Als Zielfunktion nehmen wir den CrossEntropy-Loss. Das ist eine Metrik auf diskreten Wahrscheinlichkeitsmaßen. 
    Wir versuchen also den Abstand der Ausgabewahrscheinlichkeitsverteilung und der Label-Wahrscheinlichkeitsverteilung zu minimieren.
    """
    from torch.nn import CrossEntropyLoss
    objective = CrossEntropyLoss()

    """
    Definieren wir nun den Training loop: Für eine von uns bestimmte Anzahl an Epochen (eine Epoche heißt einmal der ganze Datensatz) füttern wir Daten durch das Netzwerk, 
    werten mit der Netzausgabe und den Labeln die Zielfunktion aus,
    berechnen die Ableitungen und machen einen Schritt im Gradientenverfahren
    """
    import torch.backends.cudnn as cudnn
    cudnn.benchmark = True
    epoch = 0
    step = 0
    running_loss = 0
    max_epochs = 20
    while epoch < max_epochs:
        train_dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, drop_last=True)
        for (images, labels) in train_dataloader:
            optimizer.zero_grad()
            if torch.cuda.is_available():
                images = images.cuda()
                labels = labels.cuda()

            output = net(images)
            loss = objective(output, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            step += 1
        epoch += 1
        print('loss after {} epochs is {}'.format(epoch, running_loss/step))
        running_loss = 0
        step = 0

    """
    Aufgabe: Definiere ein val_dataset Objekt (die __init__ des MNIST Datensatzes nimmt einem training= bool entgegen, der, falls False, den validierungsdatensatz zurückgibt).
                Berechne alle 10 epochen des mittleren Wert des Zielfunktion auf dem Validierungsdatensatz.
    Aufgabe: Implementiere in der Netzwerkdefinition weiter oben eine predict methode. Diese soll zu einem eingabebild die prädizierte Klasse ausgeben.
                Hinweis: Benutze dazu nn.Softmax (https://pytorch.org/docs/stable/generated/torch.nn.Softmax.html) 
                und torch.argmax (https://pytorch.org/docs/stable/generated/torch.argmax.html)
                Teste diese methode mit Elementen des val-datensatzes.
    """
    import torch.nn.functional as F
    val_dataset = MNIST(mnist_root, download=True, train=False, transform=transforms.ToTensor())
    correct = 0
    wrong = 0
    for (image, target) in val_dataset:
        if torch.cuda.is_available():
            image = image.cuda().unsqueeze(0)
        out = net(image)
        out = F.softmax(out) # x -> [e^x_i / sum_j(e^x_j)]
        out = torch.argmax(out, dim=1).cpu().numpy()
        print(F'Predicted class of input  image is {out[0]}, true label is {target}')
        if out[0] == target:
            correct += 1
        else:
            wrong += 1
    print(F'Accuracy on val dataset is {1 - wrong/correct}')










