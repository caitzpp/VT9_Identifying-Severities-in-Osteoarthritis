import os
# from dotenv import load_dotenv
# load_dotenv()

DEBUG = os.getenv("DEBUG", "False")

DATA_PATH = os.getenv("DATA_PATH", "./data")
OUTPUT_PATH = os.getenv("OUTPUT_PATH", "./outputs")

SCHULTHESS_DATAPATH = os.path.join(DATA_PATH, os.getenv("SCHULTHESS_DATAFOLDER", "images_knee"))
CHENETAL_DATAPATH = os.path.join(DATA_PATH, os.getenv("CHENETAL_DATAFOLDER", "kaggle_dataset"))

PATH_TO_ANOM = os.path.join(OUTPUT_PATH, "dfs")