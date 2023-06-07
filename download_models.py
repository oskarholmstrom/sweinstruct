import os
import gdown

folder_urls = [
    "https://drive.google.com/drive/folders/1MT6Z-efcJmLG0AGoIbPrsQnBtO3EFPLi?usp=share_link", # GPT_SW3_1.3B Instruct
    "https://drive.google.com/drive/folders/11uMBL7g2r_b2-4gSYDg4KOgR6gvlP-7x?usp=share_link", # OPT_1.3B Instruct
]

for folder_url in folder_urls:
    print(folder_url)
    gdown.download_folder(folder_url, quiet=True)