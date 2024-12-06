import os

no_clusters = 3
current_path = os.path.dirname(os.getcwd())
DATA_PATH = os.path.join(current_path, 'data\\')
IMAGES_PATH = os.path.join(current_path, 'images\\')

if not os.path.exists(IMAGES_PATH):
    os.makedirs(IMAGES_PATH)

filename = DATA_PATH + f'Livestock_combined_kmeans_{no_clusters}labels.csv'