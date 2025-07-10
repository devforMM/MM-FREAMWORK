# Batch Normalization (From Scratch)

Ce dossier contient une implémentation simple de la technique de **Batch Normalization** en utilisant **PyTorch**, dans le cadre d’un framework de deep learning minimaliste développé from scratch.

## 📌 Description

La **Batch Normalization** est une technique utilisée pour améliorer la stabilité et la vitesse d'entraînement des réseaux de neurones profonds.  
Elle normalise les activations de chaque batch pour qu'elles aient une moyenne proche de 0 et une variance proche de 1, puis réapplique deux paramètres apprenables (`gamma` et `beta`) pour conserver la capacité de moduler la distribution des activations.

Cette implémentation permet de :
- Calculer la moyenne et la variance d’un batch.
- Normaliser les valeurs.
- Réappliquer une échelle (`gamma`) et un biais (`beta`).
- Retourner les activations normalisées.


