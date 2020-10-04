import mchmm as mc
import os
os.environ["PATH"] += os.pathsep + 'C:\Program Files (x86)\Graphviz 2.43.20200430.0238\bin'
a = mc.MarkovChain().from_data('AABCABCBAAAACBCBACBABCABCBACBACBABABCBACBBCBBCBCBCBACBABABCBCBAAACABABCBBCBCBCBCBCBAABCBBCBCBCCCBABCBCBBABCBABCABCCABABCBABC')
a.observed_matrix
a.observed_p_matrix
graph = a.graph_make(
      format="png",
      graph_attr=[("rankdir", "LR")],
      node_attr=[("fontname", "Roboto bold"), ("fontsize", "20")],
      edge_attr=[("fontname", "Iosevka"), ("fontsize", "12")]
    )
graph.render()