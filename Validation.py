import sys
import pyodbc
import pandas as pd
import json as json
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
plt.rcdefaults()
from pprint import pprint
from markovchain import MarkovChain
np.set_printoptions(threshold=np.inf)

class Validation(object):
	"""docstring for Validation"""
	def __init__(self):
		super(Validation, self).__init__()

	def get_validation_Statistics(self, event_date, space):
		server = 'PEMIL' 
		database = 'HMM_HomeStates' 
		username = 'sa' 
		password = '123.123' 
		conn = pyodbc.connect('DRIVER={SQL Server};SERVER='+server+';DATABASE='+database+';UID='+username+';PWD='+ password)
		#cursor = conn.cursor()

		#cursor.execute("""EXEC [dbo].[SP_Remove_Anomalies] 1, 'kitchen' """)
		cmd_prod_executesp = """EXEC   [dbo].[Get_Hmm_Validation_Statistics] '', ''   """
		df = pd.read_sql(cmd_prod_executesp, conn)

		

		conn.close()
		return df


	def view_validation_sats(self):
		df = self.get_validation_Statistics('2016/12/22', '')
		index =  df['Category']
		values_1 = df['Event count']
		values_2 = df['percent']

		print(type(values_1.to_numpy()))
				
		df = pd.DataFrame({'No. of Records': values_1.to_numpy(),
                   'Accuracy %': values_2.to_numpy()}, index=index.to_numpy())
		ax = df.plot.bar(rot=0,subplots=True)
		plt.show()
	
		print(df)
		
ex = Validation()
print(ex.view_validation_sats())