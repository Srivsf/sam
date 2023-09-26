Version- 1.1.8
Update the file name and destination folder in the input and execute the run.py

--> run.py {path to input file} columnName COLUMN_NAME_STDZD RESULTS lookup_file {BT_Pickle_version}{PolicyTerm_pickle_version} {path to BT regex rule file} {path to PolicyTerms regex rule file}
for example if input file is 13, April 2023:
--> run.py data/BLINDSET_BT_gcp_sample_lt250_4_13_23_BT_SPLIT_10.csv columnName COLUMN_NAME_STDZD RESULTS data/Latest_Lookup_v18_2.csv 3 2 data/bt_regex_rules_v15.csv data/pt_regex_rules_v5.csv

Will create 1 files
--> Final_File_With_Integrated_Classifications_{run_date}.csv

------------------------------------------------------------------------------------------------------
If the Input is a Rest API call then Code expects input as a json string. below is the file where the Rest API Calls will hit along with other inputs

--> run_Adhoc.py 'json string' 'column name' 'name given to standardized column name' 'destination folder'

Will create 1 files
--> Final_File_With_Integrated_Classifications_{run_date}_classified.json

-------------------------------------------------------------------------------------------------------------

Input Data= 13 April 2023
Label= business_term
Standardization version = V24
Rational expression = version 1.2
rule table for rational BT = v15
rule table for rational Policy Terms = v5
Verion of Pickle Files = 3
KNN version = v1
NB version = v1
KNN SBERT = v1
DT = v1
DT SBERT = v1
TFIDF Vectorizer version = v1
LLM Vectorizer version = v1
Integration = Standardization, Rational, KNN, KNN SBERT, NB, DT, DT SBERT
Kfolds = 5 fold
Batch wrapper version = v1.1.8
Online Wrapper version = v1.1.4
Automated = Yes

---------------------------------------------------------------------------------------------------------------

Running in Domino:
1. #### Importing Libraries:
export https_proxy="http://server.proxy.vzwcorp.com:9290"
export http_proxy="http://server.proxy.vzwcorp.com:9290"
export no_proxy=.verizon.com,oneartifactoryci.verizon.com,localhost,metadata.google.internal


pip install sentence_transformers==2.2.2 
pip install protobuf==3.20.3

2. ##### Running Command from terminal

python run.py /domino/datasets/local/NextgenCC_ML_Test/{input file name}.csv columnName COLUMN_NAME_STDZD RESULTS data/{lookup file name}.csv 4 2 /domino/datasets/local/NextgenCC_ML_Test/{BT regex rules file name}.csv /domino/datasets/local/NextgenCC_ML_Test/{POLT regex rules file name}.csv




