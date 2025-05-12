import os
from dotenv import load_dotenv
load_dotenv()

DEBUG = os.getenv("DEBUG", "False")

DATA_PATH = os.getenv("DATA_PATH", "./data")
OUTPUT_PATH = os.getenv("OUTPUT_PATH", "./outputs")

SCHULTHESS_DATAFOLDER = os.getenv("SCHULTHESS_DATAFOLDER", "images_knee")

if SCHULTHESS_DATAFOLDER == "":
    SCHULTHESS_DATAPATH = DATA_PATH
else:
    SCHULTHESS_DATAPATH = os.path.join(DATA_PATH, SCHULTHESS_DATAFOLDER)

CHENETAL_DATAPATH = os.path.join(DATA_PATH, os.getenv("CHENETAL_DATAFOLDER", "kaggle dataset"))

PATH_TO_ANOM = os.path.join(os.getenv("RESULTS_PATH"), "dfs")
PATH_TO_RESULTS = os.path.join(os.getenv("RESULTS_PATH"), "results")
DIR_PATH = os.getenv("DIR_PATH", "./SS-FewSOME_Disease_Severity_Knee_Osteoarthritis")