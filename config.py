import os
from dotenv import load_dotenv
load_dotenv()

DEBUG = os.getenv("DEBUG", "False")

DATA_PATH = os.getenv("DATA_PATH", "./data")
RAW_DATA_PATH = os.path.join(DATA_PATH, "raw")
PROC_DATA_PATH = os.path.join(DATA_PATH, "processed")
PROC_DATA_PATH2 = os.getenv("PROC_DATA_PATH", PROC_DATA_PATH)
OUTPUT_PATH = os.getenv("OUTPUT_PATH", "./outputs")

SCHULTHESS_DATAFOLDER = os.getenv("SCHULTHESS_DATAFOLDER", "images_knee")

if SCHULTHESS_DATAFOLDER == "":
    SCHULTHESS_DATAPATH = DATA_PATH
else:
    SCHULTHESS_DATAPATH = os.path.join(RAW_DATA_PATH, SCHULTHESS_DATAFOLDER)

CHENETAL_DATAPATH = os.path.join(DATA_PATH, os.getenv("CHENETAL_DATAFOLDER", "kaggle dataset"))

RESULTS_PATH = os.getenv("RESULTS_PATH", "./results")
PATH_TO_ANOM = os.path.join(RESULTS_PATH, "dfs")
PATH_TO_RESULTS = os.path.join(RESULTS_PATH, "results")
FEATURE_PATH = RESULTS_PATH
DIR_PATH = os.getenv("DIR_PATH", "./SS-FewSOME_Disease_Severity_Knee_Osteoarthritis")

HDBSCAN_SYMP_WANDBAPI_KEY = os.getenv("HDBSCAN_SYMP_WANDBAPI_KEY", "")