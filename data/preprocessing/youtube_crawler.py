import os
import pandas as pd
import youtube_dl

save_path = 'C:/Users/Seogki/GoogleDrive/데이터청년캠퍼스_고려대과정/youtube_data/'
links_csv = 'C:/Users/Seogki/GoogleDrive/데이터청년캠퍼스_고려대과정/youtube_data/links.csv'

def main():
    link_list = pd.read_csv(links_csv).tolist()

    output_dir = os.path.join(save_path, '%(title)s.%(ext)s')
    ydl_opt = {
        'outtmpl': output_dir,
        'format': "bestvideo[height=720]"
    }

    with youtube_dl.YoutubeDL(ydl_opt) as ydl:
        ydl.download(link_list)