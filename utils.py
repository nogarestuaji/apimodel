import os
import gdown

GDRIVE_IDS = {
    "Fasilitas": "15eJMZaUXL72H2HjxcNwVVWE3iqWW3wN0",     # Ganti dengan ID asli
    "Harga": "15g9XdiwwWAv_R-qwZB-DgjXZAd836Xy6",
    "Pelayanan": "15jTS0ANZQ0slwCOkVk0V4qk_4Ri5fhWb"
}

def download_model(aspect):
    file_id = GDRIVE_IDS[aspect]
    output_path = f"./tmp_pipeline/pipeline_absa_indobert_{aspect}.pkl"
    os.makedirs("./tmp_pipeline", exist_ok=True)

    if not os.path.exists(output_path):
        url = f"https://drive.google.com/uc?id={file_id}"
        print(f"Mengunduh model {aspect} dari Google Drive...")
        gdown.download(url, output_path, quiet=False)

    return output_path
