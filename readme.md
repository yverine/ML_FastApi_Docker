classification detection age (avec Deployment)
faire un simple model de classification d'age et utiliser FastAPI pour faire un api et deployer en utilisant Docker. le fichier dockerfile inclu les commandes.

Requirements
python 3.6+
pip install requirements.txt

entrainement des models
$ cd models
$ python model_classifier_detection_age.py
demarrer le serveur
$ cd ..
$ python deploy.py

Docker image
docker build -t fastapi .
Start Docker container
docker run -d -p 8000:8000 --name fastapicontainer fastapi