import csv
import pandas as pd
import json
import numpy as np
from pymongo import MongoClient

csv_file="/home/rtx2070s/exp/deep-latent-sequence-model/data/yelp_long/yelp_academic_dataset_review.json"
#too heavy
# with open(csv_file, 'r') as f:
#     js_file=json.load(f)
#     print(js_file)

mongo_client=MongoClient('localhost',27017)
dblist=mongo_client.list_database_names()

db = mongo_client.yelpreview
col=db.yelp_academic_dataset_review


mongo_docs = col.find()
#easier implement
entries=list(mongo_docs)
entries[:5]
df=pd.DataFrame(entries)
print(df.head())
#test for mongoDB documents
# for doc in mongo_docs:
#     _id = doc["_id"]
#     print (doc)
#creat field 
# fields={}
#iterate from mongoDB documents
# for doc in mongo_docs:
#     #iterate key-value pairs
#     for key,val in doc.items():
#         try:
#             fields[key] =np.append(fields[key],val)
#         except KeyError:
#             fields[key]=np.array([val])
# print(fields)
# #create list for series
# for key,val in fields.item():
#     if key!='_id':
#         fields[key]=pd.Series(fields[key])
#         fields[key].index=fields["_id"]
#         print(key)
#         print(fields[key])
#         print(fields[key].index)
#         series_list += [fields[key]]
# #create dic for DataFrame
# df_series={}
# for num,series in enumerate(series_list):
#     #df_series['data_1']=series
#     df_series['data_'+str(num)]=series
# #create DataFrame from  Series dictionary
# mongo_df=panda.DataFrame(df_series)
# print('\nmonogo_df:',type(mongo_df))
# #iterate over DataFrame
# for series in mongo_df.itertuples():
#     for num,item in enumerate(series):
#         print(item)
#     print(series)
#     print('\n')





# print (mongo_docs)



print