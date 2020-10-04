
#http://www.blackarbs.com/blog/introduction-hidden-markov-models-python-networkx-sklearn/2/9/2017

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt


# create state space and initial state probabilities

states = ['sleeping', 'eating', 'pooping']
pi = [0.35, 0.35, 0.3]
state_space = pd.Series(pi, index=states, name='states')
print(state_space)
print(state_space.sum())

#defining transition probabiities (probability of changing the existing state to another state)

q_df = pd.DataFrame(columns=states, index=states)
q_df.loc[states[0]] = [0.4, 0.2, 0.4]
q_df.loc[states[1]] = [0.45, 0.45, 0.1]
q_df.loc[states[2]] = [0.45, 0.25, .3]

print(q_df)

q = q_df.values
print('\n', q, q.shape, '\n')
print(q_df.sum(axis=1))


#drawing the markov diagram


from pprint import pprint 

# create a function that maps transition probability dataframe 
# to markov edges and weights

def _get_markov_edges(Q):
    edges = {}
    for col in Q.columns:
        for idx in Q.index:
            edges[(idx,col)] = Q.loc[idx,col]
    return edges

edges_wts = _get_markov_edges(q_df)
pprint(edges_wts)


# create graph object
G = nx.MultiDiGraph()

# nodes correspond to states
G.add_nodes_from(states)
print(f'Nodes:\n{G.nodes()}\n')

# edges represent transition probabilities
for k, v in edges_wts.items():
    tmp_origin, tmp_destination = k[0], k[1]
    G.add_edge(tmp_origin, tmp_destination, weight=v, label=v)
print(f'Edges:')

pprint(G.edges(data=True))    

# pos = nx.drawing.nx_pydot.graphviz_layout(G, prog='dot')
# nx.draw_networkx(G, pos)

# # create edge labels for jupyter plot but is not necessary
# edge_labels = {(n1,n2):d['label'] for n1,n2,d in G.edges(data=True)}
# nx.draw_networkx_edge_labels(G , pos, edge_labels=edge_labels)
# nx.drawing.nx_pydot.write_dot(G, 'pet_dog_markov.dot')

# create state space and initial state probabilities

hidden_states = ['healthy', 'sick']
pi = [0.5, 0.5]
state_space = pd.Series(pi, index=hidden_states, name='states')
print(state_space)
print('\n', state_space.sum())

# create hidden transition matrix
# a or alpha 
#   = transition probability matrix of changing states given a state
# matrix is size (M x M) where M is number of states

a_df = pd.DataFrame(columns=hidden_states, index=hidden_states)
a_df.loc[hidden_states[0]] = [0.7, 0.3]
a_df.loc[hidden_states[1]] = [0.4, 0.6]

print(a_df)

a = a_df.values
print('\n', a, a.shape, '\n')
print(a_df.sum(axis=1))


# create matrix of observation (emission) probabilities
# b or beta = observation probabilities given state
# matrix is size (M x O) where M is number of states 
# and O is number of different possible observations

observable_states = states

b_df = pd.DataFrame(columns=observable_states, index=hidden_states)
b_df.loc[hidden_states[0]] = [0.2, 0.6, 0.2]
b_df.loc[hidden_states[1]] = [0.4, 0.1, 0.5]

print(b_df)

b = b_df.values
print('\n', b, b.shape, '\n')
print(b_df.sum(axis=1))


hide_edges_wts = _get_markov_edges(a_df)
pprint(hide_edges_wts)

emit_edges_wts = _get_markov_edges(b_df)
pprint(emit_edges_wts)



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


# observation sequence of dog's behaviors
# observations are encoded numerically

obs_map = {'sleeping':0, 'eating':1, 'pooping':2}
obs = np.array([1,1,2,1,0,1,2,1,0,2,2,0,1,0,1])

inv_obs_map = dict((v,k) for k, v in obs_map.items())
obs_seq = [inv_obs_map[v] for v in list(obs)]

print( pd.DataFrame(np.column_stack([obs, obs_seq]), 
                columns=['Obs_code', 'Obs_seq']) )



# define Viterbi algorithm for shortest path
# code adapted from Stephen Marsland's, Machine Learning An Algorthmic Perspective, Vol. 2
# https://github.com/alexsosn/MarslandMLAlgo/blob/master/Ch16/HMM.py

def viterbi(pi, a, b, obs):
    
    nStates = np.shape(b)[0]
    T = np.shape(obs)[0]
    
    # init blank path
    path = np.zeros(T)
    # delta --> highest probability of any path that reaches state i
    delta = np.zeros((nStates, T))
    # phi --> argmax by time step for each state
    phi = np.zeros((nStates, T))
    
    # init delta and phi 
    delta[:, 0] = pi * b[:, obs[0]]
    phi[:, 0] = 0

    print('\nStart Walk Forward\n')    
    # the forward algorithm extension
    for t in range(1, T):
        for s in range(nStates):
            delta[s, t] = np.max(delta[:, t-1] * a[:, s]) * b[s, obs[t]] 
            phi[s, t] = np.argmax(delta[:, t-1] * a[:, s])
            print('s={s} and t={t}: phi[{s}, {t}] = {phi}'.format(s=s, t=t, phi=phi[s, t]))
    
    # find optimal path
    print('-'*50)
    print('Start Backtrace\n')
    path[T-1] = np.argmax(delta[:, T-1])
    #p('init path\n    t={} path[{}-1]={}\n'.format(T-1, T, path[T-1]))
    
    return  delta, phi

#path, 
delta, phi = viterbi(pi, a, b, obs)
#print('\nsingle best state path: \n', path)
print('delta:\n', delta)
print('phi:\n', phi)
print ('pi:\n', pi)


def viterbi_(obs, states, start_p, trans_p, emit_p):
	print("second implementation")
	V=[{}]
	for i in states:
		V[0][i]=start_p[i]*emit_p[i][obs[0]]
	# Run Viterbi when t > 0
	for t in range(1, len(obs)):
		V.append({})
		for y in states:
			(prob, state) = max((V[t-1][y0] * trans_p[y0][y] * emit_p[y][obs[t]], y0) for y0 in states)
			V[t][y] = prob
		for i in dptable(V):
			print (i)
		opt=[]
		for j in V:
			for x,y in j.items():
				if j[x]==max(j.values()):
					opt.append(x)
	#the highest probability
	h=max(V[-1].values())
	print ('The steps of states are '+' '.join(opt)+' with highest probability of %s'%h)
	#it prints a table of steps from dictionary

def dptable(V):
	yield " ".join(("%10d" % i) for i in range(len(V)))
	for y in V[0]:
		yield "%.7s: " % y+" ".join("%.7s" % ("%f" % v[y]) for v in V)




states = ('Healthy', 'Fever')
observations = ('normal', 'cold', 'dizzy')
start_probability = {'Healthy': 0.6, 'Fever': 0.4}
transition_probability = {
	'Healthy' : {'Healthy': 0.7, 'Fever': 0.3},
	'Fever' : {'Healthy': 0.4, 'Fever': 0.6}
	}


states = ('OCCUPIED ACTIVE','OCCUPIED INACTIVE','UNOCCUPIED','UNKNOWN')
observations = ('Transit', 'Short Active', 'Long Active','Short Inactive','Long Inactive', 'N/A')
start_probability = {'OCCUPIED ACTIVE': 0.25, 'OCCUPIED INACTIVE': 0.25, 'UNOCCUPIED' : 0.25, 'UNKNOWN':0.25}
transition_probability = {	
	'OCCUPIED ACTIVE' : {'OCCUPIED ACTIVE': 0.6 ,'OCCUPIED INACTIVE': 0.2,'UNOCCUPIED': 0.1,'UNKNOWN':0.1} ,
	'OCCUPIED INACTIVE' : {'OCCUPIED ACTIVE': 0.3 ,'OCCUPIED INACTIVE': 0.6,'UNOCCUPIED': 0.0,'UNKNOWN':0.1} ,
	'UNOCCUPIED' : {'OCCUPIED ACTIVE': 0.3 ,'OCCUPIED INACTIVE': 0.0,'UNOCCUPIED': 0.6,'UNKNOWN':0.1} ,
	'UNKNOWN' : {'OCCUPIED ACTIVE': 0.5 ,'OCCUPIED INACTIVE': 0.0,'UNOCCUPIED': 0.26,'UNKNOWN':0.24} ,

	}	


emission_probability = {
	'OCCUPIED ACTIVE' : {'Transit':0.0, 'Short Active' :0.5, 'Long Active':0.45,'Short Inactive':0.0,'Long Inactive':0.0, 'N/A':0.05},
	'OCCUPIED INACTIVE' : {'Transit':0.0, 'Short Active':0.0, 'Long Active':0.0,'Short Inactive':0.6,'Long Inactive':0.35, 'N/A':0.05},
	'UNOCCUPIED' : {'Transit':0.0, 'Short Active':0.0, 'Long Active':0.0,'Short Inactive':0.0,'Long Inactive':0.9, 'N/A':0.1},
	'UNKNOWN' : {'Transit':0.7, 'Short Active':0.1, 'Long Active':0.0,'Short Inactive':0.1,'Long Inactive':0.0, 'N/A':0.1}
	}





viterbi_(observations,
                   states,
                   start_probability,
                   transition_probability,
                   emission_probability)
