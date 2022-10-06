import cv2 as cv2
import cv2 as cv
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
import warnings
warnings.filterwarnings("ignore")

# read in raw csv file
fr = pd.read_csv("FR_marteye.csv")

# do some cleaning - We only want relevant clips
fr = fr[fr.cat_weightKgs.isna() == False] #if no weight (target), then remove
fr = fr[fr.main_cam_lot_duration_secs < 60].reset_index(drop=True).reset_index() # if clip really long, then remove

print(len(fr), "observations in the dataset")

# define function to convert hms to seconds
def get_sec(time_str):
    """Get seconds from time."""
    h, m, s = time_str.split(':')
    return int(h) * 3600 + int(m) * 60 + int(s)

# determine the seconds intervals for each relevant clip
times = [str(int(get_sec(fr.main_cam_lot_seek_hms[i]))) + "-" + str(int(get_sec(fr.main_cam_lot_seek_hms[i]))+ fr.main_cam_lot_duration_secs[i]) for i in range(len(fr)) ]

# Define the full video - we want to pull clips from this
required_video_file = "FR_marteye.mp4"

#Loop throi
for i in range(len(fr)):
    time = times[i]
    naming = "lot" + str(fr.index[i])
    starttime = int(time.split("-")[0])
    endtime = int(time.split("-")[1])
    ffmpeg_extract_subclip(required_video_file, starttime, endtime, targetname="FR_clips/" +str(naming)+".mp4")

fr = fr[["index", "cat_breedCode", "cat_weightKgs", "cat_sex", "cat_ageInDays"]]
fr["clip"] = ["FR_clips/lot" + str(fr["index"][i]) +".mp4" for i in range(len(fr))]

fr.to_csv("FR_cleaned_data.csv")

print("Successfully created clips")