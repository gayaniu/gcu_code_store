############################################################################################
#Author: Gayani Udawatta
#Created Date : 2020/08/29
#Modified Data: 
#Description: Call HMM - Viterbi Algorithm 
#############################################################################################



import sys
import pyodbc
import pandas as pd
import json as json
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from pprint import pprint
from markovchain import MarkovChain
np.set_printoptions(threshold=np.inf)



class States(object):
	"""docstring for States"""
	def __init__(self):
		super(States, self).__init__()


	def _get_markov_edges(self,Q):
		edges = {}
		for col in Q.columns:
			for idx in Q.index:
				edges[(idx,col)] = Q.loc[idx,col]
		return edges

#Define the initial probabilities and states

	def define_initial_states(self):
		
		states = ['Transit', 'Active','Short_Inactive','Long_Inactive', 'TR_LongInactive']  #long inactive
		#Setting up initial probabilities to 0.2  each. total probability is equal to 1.0 
		pi = [0.2, 0.2, 0.2,0.2,0.2]
		state_space = pd.Series(pi, index=states, name='states')
		print(state_space)
		print(state_space.sum())


		#defining the transition probabilities: Staying in the same state or moving in to different state
		q_df = pd.DataFrame(columns=states, index=states)
		q_df.loc[states[0]] = [0.0, 0.3,0.2,0.25, 0.25]
		q_df.loc[states[1]] = [0.2, 0,0.25,0.25, 0.3]
		q_df.loc[states[2]] = [0.1, 0.3,0,0.2,0.4]
		q_df.loc[states[3]] = [0.25, 0.25,0.2,0.0, 0.3]
		q_df.loc[states[4]] = [0.30, 0.25,0.1,0.1,0.25]
	
		q = q_df.values
		print('\n', q, q.shape, '\n')
		print(q_df.sum(axis=1))


		#Transition probability graph section.....

		graph_edges = ['T', 'A' , 'SI' , 'LI' , 'TL']
		q_df_graph = pd.DataFrame(columns=graph_edges, index=graph_edges)
		q_df_graph.loc[graph_edges[0]] = [0.0, 0.3,0.2,0.25, 0.25]
		q_df_graph.loc[graph_edges[1]] = [0.2, 0,0.25,0.25, 0.3]
		q_df_graph.loc[graph_edges[2]] = [0.1, 0.3,0,0.2,0.4]
		q_df_graph.loc[graph_edges[3]] = [0.25, 0.25,0.2,0.0, 0.3]
		q_df_graph.loc[graph_edges[4]] = [0.30, 0.25,0.1,0.1,0.25]
		

		print('Graph section...........')
		print(q_df_graph)		
		edges_wts = self._get_markov_edges(q_df_graph)
		pprint(edges_wts)

		# create graph object
		G = nx.MultiDiGraph()

		# nodes correspond to states
		G.add_nodes_from(graph_edges)
		print(f'Nodes:\n{G.nodes()}\n')

		# edges represent transition probabilities
		for k, v in edges_wts.items():
			tmp_origin, tmp_destination = k[0], k[1]
			G.add_edge(tmp_origin, tmp_destination, weight=v, label=v)
	  

		pos = nx.drawing.nx_pydot.graphviz_layout(G, prog='dot')
		nx.draw_networkx(G, pos)
		print(f'Edges:')
		pprint(G.edges(data=True))  

		# create edge labels for jupyter plot but is not necessary
		edge_labels = {(n1,n2):d['label'] for n1,n2,d in G.edges(data=True)}
		nx.draw_networkx_edge_labels(G , pos, edge_labels=edge_labels)
		nx.drawing.nx_pydot.write_dot(G, 'initial_states_Markov.dot')

        
		#End of the graph section

		return states

	def get_hidden_states(self, Facility_No, space):

		#defining hidden states
		hidden_states = ['OCCUPIED_ACTIVE','OCCUPIED_INACTIVE','UNOCCUPIED','UNKNOWN']
		#defining initial probabilities for the hidden states
		pi = [0.25, 0.25,0.25, 0.25]
		state_space = pd.Series(pi, index=hidden_states, name='states')
		print(state_space)
		print('\n', state_space.sum())
	
		#creating the hidden transition matrix
		a_df = pd.DataFrame(columns=hidden_states, index=hidden_states)
		a_df.loc[hidden_states[0]] = [0.5, 0.2,0.1,0.2]
		a_df.loc[hidden_states[1]] = [0.3, 0.6, 0.0, 0.1]
		a_df.loc[hidden_states[2]] = [0.1, 0.0, 0.6, 0.3]
		a_df.loc[hidden_states[3]] = [0.5, 0.0, 0.26, 0.24]

		print(a_df)

		#Transition probability graph section.....
		
		a_df_hidden_edges = pd.DataFrame(columns=hidden_states, index=hidden_states)
		a_df_hidden_edges.loc[hidden_states[0]] = [0.5, 0.2,0.1,0.2]
		a_df_hidden_edges.loc[hidden_states[1]] = [0.3, 0.6, 0.0, 0.1]
		a_df_hidden_edges.loc[hidden_states[2]] = [0.1, 0.0, 0.6, 0.3]
		a_df_hidden_edges.loc[hidden_states[3]] = [0.5, 0.0, 0.26, 0.24]
		

		print('Graph section for hidden transition matrix...........')
		print(a_df_hidden_edges)		
		edges_wts = self._get_markov_edges(a_df_hidden_edges)
		pprint(edges_wts)

		# create graph object
		G = nx.MultiDiGraph()

		# nodes correspond to states
		G.add_nodes_from(hidden_states)
		print(f'Nodes:\n{G.nodes()}\n')

		# edges represent transition probabilities
		for k, v in edges_wts.items():
			tmp_origin, tmp_destination = k[0], k[1]
			G.add_edge(tmp_origin, tmp_destination, weight=v, label=v)
	  

		pos = nx.drawing.nx_pydot.graphviz_layout(G, prog='dot')
		nx.draw_networkx(G, pos)
		print(f'Edges:')
		pprint(G.edges(data=True))  

		# create edge labels for jupyter plot but is not necessary
		edge_labels = {(n1,n2):d['label'] for n1,n2,d in G.edges(data=True)}
		nx.draw_networkx_edge_labels(G , pos, edge_labels=edge_labels)
		nx.drawing.nx_pydot.write_dot(G, 'hidden_states_Markov.dot')

		#end of hidden transition probability graph section-------------


		a = a_df.values
		print('\n', a, a.shape, '\n')
		print(a_df.sum(axis=1))
		states = ['Transit', 'Active','Short_Inactive','Long_Inactive', 'TR_LongInactive']   #long inactive

		observable_states = states

		b_df = pd.DataFrame(columns= observable_states, index=hidden_states)
		b_df.loc[hidden_states[0]] = [0.0 ,0.9,0.0,0.0,0.1]
		b_df.loc[hidden_states[1]] = [0.05,0.1,0.15,0.65,0.05] 
		b_df.loc[hidden_states[2]] = [0.00,0.05,0.0,0.25,0.7]
		b_df.loc[hidden_states[3]] = [0.8,0.0,0.0,0.05,0.15]

		print('Emission Matrix........')
		print(b_df)
		b = b_df.values

		#----------------------------Emission probability graph section-------------------------------
		#creating graph edges and weights
		hide_edges_wts = self._get_markov_edges(a_df)
		pprint(hide_edges_wts)

		emit_edges_wts = self._get_markov_edges(b_df)
		pprint(emit_edges_wts)

		#creating the graph object
		# create graph object
		G = nx.MultiDiGraph()

		# nodes correspond to states
		G.add_nodes_from(hidden_states)
		print(f'Nodes:\n{G.nodes()}\n')

		# edges represent hidden probabilities
		for k, v in hide_edges_wts.items():
		    tmp_origin, tmp_destination = k[0], k[1]
		    G.add_edge(tmp_origin, tmp_destination, weight=v, label=v)

		# edges represent emission probabilities
		for k, v in emit_edges_wts.items():
		    tmp_origin, tmp_destination = k[0], k[1]
		    G.add_edge(tmp_origin, tmp_destination, weight=v, label=v)
		    
		print(f'Edges:')
		pprint(G.edges(data=True))  		


		pos = nx.drawing.nx_pydot.graphviz_layout(G, prog='neato')
		nx.draw_networkx(G, pos)

		# # create the plot for emmission probabilities 
		emit_edge_labels = {(n1,n2):d['label'] for n1,n2,d in G.edges(data=True)}
		nx.draw_networkx_edge_labels(G , pos, edge_labels=emit_edge_labels)
		nx.drawing.nx_pydot.write_dot(G, 'emission__markov.dot')

		#---------------------------------end of Emission probability graph section------------------------------------

		row_id, duration_min, obs, obs_seq = self.get_space_occupancy(Facility_No,space)
		path, delta, phi = self.Viterbi(pi, a, b, obs)
		#np.set_printoptions(suppress=True)
		print('\nsingle best state path: \n', path)
		#print('delta:\n', delta)
		
		#np.set_printoptions(suppress=True) 
		#print(delta)

		xv = delta.transpose()
		row, col = xv.shape			
	
		df_xv = pd.DataFrame(xv)		
		
		pd.set_option('display.max_rows', 100)
		#print('phi:\n', phi)
		
		print( pd.DataFrame(np.column_stack([obs, obs_seq]), 
                columns=['Obs_code', 'Obs_seq']) )
		

		state_map = { 0:'OCCUPIED_ACTIVE',1:'OCCUPIED_INACTIVE',2:'UNOCCUPIED',3:'UNKNOWN'}
		state_path = [state_map[v] for v in path]
		print(pd.DataFrame()
		 .assign(Observation=obs_seq)
		 .assign(Best_Path=state_path))


		pds= pd.DataFrame().assign(row_id=row_id).assign(duration_min=duration_min).assign(Observation=obs_seq).assign(Best_Path=state_path)
		frames = [pds, df_xv]
		result = pd.concat([pds, df_xv], axis =1)
		result.columns = ['row_id','duration_min','Observation','State','OCCUPIED_ACTIVE','OCCUPIED_INACTIVE','UNOCCUPIED','UNKNOWN']

		pd.set_option('display.max_rows', 500)
		#pd.set_option('display.max_columns', 500)
		pd.set_option('display.float_format', lambda x: '%.19f' % x)
		print(result.head(500))

		#---------------insert final result to database--------------------------------------
		self.insert_hmm_results(result)		
	
		

	def get_space_data(self,  Facility_No , space):

		server = 'PEMIL' 
		database = 'HMM_HomeStates' 
		username = 'sa' 
		password = '123.123' 

		if (space ==''):
			space ="ALL"	
	
		conn = pyodbc.connect('DRIVER={SQL Server};SERVER='+server+';DATABASE='+database+';UID='+username+';PWD='+ password)

		cmd_prod_executesp = """EXEC   [dbo].[SP_Get_Occupancy_Data_By_Space]  @Facility_No = ?, @Space = ?  """ 
		df = pd.read_sql(cmd_prod_executesp, conn, params=( int(Facility_No), space))

		print(df)	
		conn.close()
		return df



	def insert_hmm_results(self, result_df):

		server = 'PEMIL' 
		database = 'HMM_HomeStates' 
		username = 'sa' 
		password = '123.123' 
		conn = pyodbc.connect('DRIVER={SQL Server};SERVER='+server+';DATABASE='+database+';UID='+username+';PWD='+ password)
		cursor = conn.cursor()

		cnxn = pyodbc.connect('DRIVER={SQL Server};SERVER='+server+';DATABASE='+database+';UID='+username+';PWD='+ password)
		cursor = cnxn.cursor()
		#delete existing records from the table before inserting new results
		cursor.execute(" Delete from dbo.Hmm_Results")		
		for index, row in result_df.iterrows():
			Facility_No = 1
			id= row.row_id
			duration_Min = row.duration_min
			Observation = row.Observation
			hmm_state = row.State
			Prob_Occupied_Active=row.OCCUPIED_ACTIVE
			Prob_Occupied_Inactive = row.OCCUPIED_INACTIVE
			Prob_Unoccupied = row.UNOCCUPIED
			Prob_Unknown = row.UNKNOWN			
			cursor.execute("INSERT INTO dbo.Hmm_Results ( Facility_No, id, Duration_min,  Observation ,hmm_state ,Prob_Occupied_Active,Prob_Occupied_Inactive, Prob_Unoccupied,Prob_Unknown  ) values(?,?,?,?,?,?,?,?,?)", Facility_No, id, duration_Min, Observation ,hmm_state ,Prob_Occupied_Active,Prob_Occupied_Inactive, Prob_Unoccupied,Prob_Unknown)
		cnxn.commit()
		cursor.close()	
	
	def Viterbi(self,pi,a,b,obs):

		nStates = np.shape(b)[0]
		T = np.shape(obs)[0]

		path = np.zeros(T)
		delta = np.zeros((nStates,T))
		phi = np.zeros((nStates,T))

		delta[:,0] = pi * b[:,obs[0]]
		phi[:,0] = 0

		for t in range(1,T):
			for s in range(nStates):
				delta[s,t] = np.max(delta[:,t-1]*a[:,s])*b[s,obs[t]]
				phi[s,t] = np.argmax(delta[:,t-1]*a[:,s])

		path[T-1] = np.argmax(delta[:,T-1])
		for t in range(T-2,-1,-1):
			path[t] = phi[int(np.round(path[t+1])), int(np.round([t+1]))]

		return path,delta, phi

	def get_space_occupancy(self, Facility_No, space):

		obs_map = {'Transit' : 0 , 'Active' : 1,'Short_Inactive': 2,'Long_Inactive': 3, 'TR_LongInactive' :4 }

		df = self.get_space_data(Facility_No,space)

		obs = np.array(df['Encoding'])
		row_id = np.array (df['id'])
		duration_min = np.array (df['Time_Gap_MIN'])

		inv_obs_map = dict((v,k) for k, v in obs_map.items())
		obs_seq = [inv_obs_map[v] for v in list(obs)]
		
		print( pd.DataFrame(np.column_stack([row_id, duration_min, obs, obs_seq]),columns=['row_id','duration_min','Obs_code', 'Obs_seq']) )

		return row_id,duration_min, obs,obs_seq

	def decimal_str(self, x: float, decimals: int = 10) -> str:
		return format(x, f".{decimals}f").lstrip().rstrip('0')

	def plot_graph(self, mk_prop, Labels):
		P = np.array([mk_prop]) # Transition matrix
		mc = MarkovChain(P, Labels)
		mc.draw()


ex = States()
#ex.get_space_occupancy(1,'kitchen')
ex.get_hidden_states(1,'')
#ex.define_initial_states()

#print(ex.get_space_data_test('kitchen', 1))
#print(ex.get_space_data(1,''))


