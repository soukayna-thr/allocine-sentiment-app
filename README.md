**Allociné Sentiment App**\
*Déploiement et Virtualisation*

### 1. Introduction et Contexte

Ce projet constitue le volet **opérationnel (MLOps)** de l'initiative **MovieSentix**. Alors que le dépôt initial
[MovieSentix](https://github.com/soukayna-thr/MovieSentix) se concentre
sur l'exploration des données (EDA), finetuning
*CamemBERT*, streaming avec Kafka et traitement avec Apache Spark. Ce dépôt a pour objectif l'industrialisation et le
déploiement du modèle via une architecture virtualisée.


### 2. Architecture Technique

L'application repose sur une architecture micro-services orchestrée par
**Docker**.

-   **Virtualisation :** Isolation complète de l'environnement
    d'exécution (bibliothèques, dépendances système) pour garantir la
    reproductibilité.

-   **Backend (API) :** Service Python exposant le modèle de
    classification de sentiment (Positif/Négatif).

-   **Frontend (Streamlit) :** Interface utilisateur interactive
    permettant de saisir des critiques en temps réel et de visualiser
    les scores de confiance.

-   **Orchestration :** Utilisation de `docker-compose` pour gérer le
    cycle de vie multi-conteneurs.

### 3. Relation avec MovieSentix

Ce projet s'inscrit dans une démarche complète :

1.  **Phase de Recherche (MovieSentix) - Big Data :**

    -   Nettoyage du dataset Allociné.

    -   Fine-tuning du modèle `CamemBERT-base`.

    -   Évaluation des métriques (F1-score, Accuracy).

2.  **Phase de Production (Ce projet) :**

    -   Encapsulation du modèle dans une image Docker.

    -   Développement de l'interface utilisateur.

    -   Mise en place du pipeline de déploiement continu.



### 4. Installation et Lancement

Le déploiement se fait en trois étapes simples grâce à l'automatisation
Docker.

1.  **Clonage du dépôt :**

    ``` {.bash language="bash"}
    git clone https://github.com/soukayna-thr/allocine-sentiment-app.git
    cd allocine-sentiment-app
    ```

2.  **Construction et lancement des conteneurs :**

    ``` {.bash language="bash"}
    docker-compose up --build
    ```

    Cette commande compile l'image Python, installe les dépendances
    définies dans `requirements.txt` et lance le serveur Streamlit.

3.  **Accès à l'application :** Ouvrez votre navigateur à l'adresse
    suivante :

    
    <http://localhost:8501>
    
