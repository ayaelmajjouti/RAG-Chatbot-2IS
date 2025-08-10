from sentence_transformers import SentenceTransformer

# 1. Charger un modèle d'embedding pré-entraîné.
# Ce modèle est très populaire et performant pour ce genre de tâche.
model = SentenceTransformer('all-MiniLM-L6-v2')

# 2. Votre question
query = "hey, can you tell me what are the courses we will be having"

# 3. Générer l'embedding (le vecteur) pour la phrase entière.
# C'est ici que les 3 étapes (tokenisation, conversion, agrégation) se produisent.
embedding_vector = model.encode(query)

# 4. Visualisons le résultat
print("La question de l'utilisateur :")
print(f"'{query}'")
print("\n" + "="*50 + "\n")
print(f"A été transformée en un vecteur de {embedding_vector.shape[0]} dimensions.")
print(embedding_vector)
