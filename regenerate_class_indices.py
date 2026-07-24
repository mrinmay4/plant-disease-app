"""
Run this INSIDE your Colab notebook (after creating `train_generator`)
to regenerate class_indices.json with the exact mapping your model was
trained with, then download the file and drop it into this project,
replacing the placeholder one.

Usage (in a Colab cell, after cell 12 in the original notebook):

    import json
    class_indices = {v: k for k, v in train_generator.class_indices.items()}
    with open('class_indices.json', 'w') as f:
        json.dump(class_indices, f, indent=2)

    from google.colab import files
    files.download('class_indices.json')

The class_indices.json shipped in this repo is the standard PlantVillage
(38-class, "color" folder) ordering used by most public notebooks based on
this dataset. It should match if you trained on the full, unmodified
'color' directory of abdallahalidev/plantvillage-dataset. If you trained on
a subset, a different split, or the dataset folder names differ, regenerate
it with the snippet above to be safe -- a mismatched mapping will still run
without errors but will label predictions incorrectly.
"""
