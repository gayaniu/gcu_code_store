############################################################################################
#Author: Gayani Udawatta
#Created Date : 2019/12/20
#Modified Data: 2020/02/06
#Description: Prepare Data in Database
#############################################################################################


import pyodbc
import pandas as pd
import json as json
import pandas.io.json as pd_json
import datetime  as datetime

class Data(object):
	"""docstring for ClassName"""
	def __init__(self):
		super(Data, self).__init__()
		#self.arg = arg
		
	def insert_full_events(self):

		df = pd.read_csv(r'D:\Masters in Data Science\2nd Year\Trimester 02\SIT790\HMM Home States\Data\P1_full_events.csv')
		
		server = 'PEMIL' 
		database = 'HMM_HomeStates' 
		username = 'sa' 
		password = '123.123' 
		cnxn = pyodbc.connect('DRIVER={SQL Server};SERVER='+server+';DATABASE='+database+';UID='+username+';PWD='+ password)
		cursor = cnxn.cursor()
		# Insert Dataframe into SQL Server:
		for index, row in df.iterrows():
		    cursor.execute("INSERT INTO dbo.Full_Events (id, space_id ,provider_id ,name,space ,type, state,event_date) values(?,?,?,?,?,?,?,?)", row.id, row.space_id ,row.provider_id ,row.name,row.space ,row.type, row.state,row.date )
		cnxn.commit()
		cursor.close()

	def insert_spaces(self):
		space_data = pd.read_json (r'D:\Masters in Data Science\2nd Year\Trimester 02\SIT790\HMM Home States\Data\P1_spaces.json')
		print(space_data)

		server = 'PEMIL' 
		database = 'HMM_HomeStates' 
		username = 'sa' 
		password = '123.123' 
		cnxn = pyodbc.connect('DRIVER={SQL Server};SERVER='+server+';DATABASE='+database+';UID='+username+';PWD='+ password)
		cursor = cnxn.cursor()
		# Insert Dataframe into SQL Server:
		
		for index, row in space_data.iterrows():
			id= row.id
			name=row.name
			kind = row.kind 
			created_at = pd.to_datetime(row.created_at)
			updated_at = pd.to_datetime(row.updated_at)
			transit_points = str(row.transit_points)
			devices= str(row.devices)
			cursor.execute("INSERT INTO dbo.Master_Space_Data (id, name ,kind ,created_at,updated_at, transit_points,devices  ) values(?,?,?,?,?,?,?)", id, name ,kind ,created_at,updated_at, transit_points,devices)
		cnxn.commit()
		cursor.close()


	def insert_transitpoints(self): 
		space_data = pd.read_json (r'D:\Masters in Data Science\2nd Year\Trimester 02\SIT790\HMM Home States\Data\P1_spaces.json')			

		df = pd.DataFrame([y for x in space_data['transit_points'] for y in x])
		print(df)

		server = 'PEMIL' 
		database = 'HMM_HomeStates' 
		username = 'sa' 
		password = '123.123' 
		cnxn = pyodbc.connect('DRIVER={SQL Server};SERVER='+server+';DATABASE='+database+';UID='+username+';PWD='+ password)
		cursor = cnxn.cursor()
		# Insert Dataframe into SQL Server:
		
		for index, row in df.iterrows():			
			cursor.execute("INSERT INTO dbo.Transit_Points (id, device_id ,space_id, neighbour_space_id ,created_at,updated_at) values(?,?,?,?,?,?)", row.id, row.device_id ,row.space_id, row.neighbour_space_id ,row.created_at,row.updated_at)
		cnxn.commit()
		cursor.close()	

	def insert_devicedata(self): 
		space_data = pd.read_json (r'D:\Masters in Data Science\2nd Year\Trimester 02\SIT790\HMM Home States\Data\P1_spaces.json')			

		df = pd.DataFrame([y for x in space_data['devices'] for y in x])
		print(df)
		print(df.dtypes)

		server = 'PEMIL' 
		database = 'HMM_HomeStates' 
		username = 'sa' 
		password = '123.123' 
		cnxn = pyodbc.connect('DRIVER={SQL Server};SERVER='+server+';DATABASE='+database+';UID='+username+';PWD='+ password)
		cursor = cnxn.cursor()
		#Insert Dataframe into SQL Server:
		
		for index, row in df.iterrows():			
			cursor.execute("INSERT INTO dbo.Devices (id,	name,	components,	roles,	space_id,	created_at,	updated_at,	domoticz_component_ids,	provider,	provider_id) values(?,?,?,?,?,?,?,?,?,?)", row.id,	row.name,	str(row.components),	str(row.roles),	row.space_id,	str(row.created_at),	str(row.updated_at),	str(row.domoticz_component_ids),	str(row.provider),	str(row.provider_id))
		cnxn.commit()
		cursor.close()	

	def insert_devicesdata_all(self):
		device_data = pd.read_json (r'D:\Masters in Data Science\2nd Year\Trimester 02\SIT790\HMM Home States\Data\P1_devices.json')
		print("Device Data")
		print(device_data)
		print(device_data.dtypes)


		server = 'PEMIL' 
		database = 'HMM_HomeStates' 
		username = 'sa' 
		password = '123.123' 
		cnxn = pyodbc.connect('DRIVER={SQL Server};SERVER='+server+';DATABASE='+database+';UID='+username+';PWD='+ password)
		cursor = cnxn.cursor()
		#Insert Dataframe into SQL Server:
		
		for index, row in device_data.iterrows():			
			cursor.execute("INSERT INTO dbo.All_Devices (id,	name,	components,	roles,	space_id,	created_at,	updated_at,	domoticz_component_ids,	provider,	provider_id, space, component_states ) values(?,?,?,?,?,?,?,?,?,?,?,?)", row.id,	row.name,	str(row.components),	str(row.roles),	row.space_id,	pd.to_datetime(row.created_at),	pd.to_datetime(row.updated_at),	str(row.domoticz_component_ids),	str(row.provider),	str(row.provider_id), str(row.space), str(row.component_states) )
		cnxn.commit()
		cursor.close()

	def insert_validationData(self): 
		validation_data = pd.read_csv (r'D:\Masters in Data Science\2nd Year\Trimester 02\SIT790\HMM Home States\Data\Validation_Data.csv')			
		print(validation_data)

		server = 'PEMIL' 
		database = 'HMM_HomeStates' 
		username = 'sa' 
		password = '123.123' 
		cnxn = pyodbc.connect('DRIVER={SQL Server};SERVER='+server+';DATABASE='+database+';UID='+username+';PWD='+ password)
		cursor = cnxn.cursor()
		# Insert Dataframe into SQL Server

		print(datetime.datetime.strptime('03:55', '%H:%M').time())
		
		for index, row in validation_data.iterrows():			
			#start_time =str(row.StartTime)
			#end_time= str(row.EndTime)
			cursor.execute("INSERT INTO dbo.Validation_Facility_data ( Facility_No,  Activity, Day_, Space_, From_, To_ ) values(?,?,?,?,?,?)", row.Facility_No,  row.Activity , row.Day_, row.Space_, row.StartTime, row.EndTime )
		cnxn.commit()
		cursor.close()	
	def get_validation_Statistics(self, event_date, space):
		server = 'PEMIL' 
		database = 'HMM_HomeStates' 
		username = 'sa' 
		password = '123.123' 
		conn = pyodbc.connect('DRIVER={SQL Server};SERVER='+server+';DATABASE='+database+';UID='+username+';PWD='+ password)
		#cursor = conn.cursor()

		#cursor.execute("""EXEC [dbo].[SP_Remove_Anomalies] 1, 'kitchen' """)
		cmd_prod_executesp = """EXEC   [dbo].[Get_Hmm_Validation_Statistics]    """
		df = pd.read_sql(cmd_prod_executesp, conn)

		print(df)	

		conn.close()
		return df


ex = Data()
print(ex.insert_validationData())

