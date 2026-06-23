import kagglehub
import shutil
import os

path = kagglehub.dataset_download("adityakadiwal/water-potability")

origem = os.path.join(path, "water_potability.csv")
destino = "./water_potability.csv"

shutil.copy(origem, destino)

print("CSV copiado para:", os.path.abspath(destino))