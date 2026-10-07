"""Analytic graph examples, invariance and missing-data checks."""
import unittest, sys, tempfile
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
import pandas as pd
import igraph as ig
from metrics import graph_scores,s_dbw
from pipeline import build_graph,align,prepare
class Core(unittest.TestCase):
 def test_two_disconnected_triangles(self):
  g=ig.Graph(n=6,edges=[(0,1),(1,2),(2,0),(3,4),(4,5),(5,3)])
  g.es['weight']=[1.]*6
  s=graph_scores(g,np.array([0,0,0,1,1,1]))
  self.assertAlmostEqual(s['AVI'],1);self.assertAlmostEqual(s['AVU'],0)
  self.assertAlmostEqual(s['MQ'],1);self.assertAlmostEqual(s['Q'],.5)
 def test_complete_graph_split(self):
  g=ig.Graph.Full(6);g.es['weight']=[1.]*g.ecount()
  s=graph_scores(g,np.array([0,0,0,1,1,1]))
  self.assertAlmostEqual(s['AVI'],.4);self.assertAlmostEqual(s['AVU'],1)
  self.assertAlmostEqual(s['MQ'],0)
 def test_labels_invariant(self):
  X=np.random.default_rng(4).normal(size=(30,6));g=build_graph(X,4)
  a=np.arange(30)%3;b=np.array([9,11,4])[a]
  for k,v in graph_scores(g,a).items():self.assertAlmostEqual(v,graph_scores(g,b)[k])
  u,v=s_dbw(X,a),s_dbw(X,b)
  self.assertTrue((np.isnan(u) and np.isnan(v)) or np.isclose(u,v))
 def test_tracking_label_swap_and_birth(self):
  a=np.array([5,5,9,9,9]);b=np.array([1,1,0,0,2]);c,n=align(a,b,10)
  np.testing.assert_array_equal(c,[5,5,9,9,10]);self.assertEqual(n,11)
 def test_graph_no_self_loops_duplicate_ties(self):
  g=build_graph(np.zeros((30,6)),4)
  self.assertFalse(any(g.is_loop()));self.assertFalse(g.is_multiple().count(True));self.assertEqual(g.vcount(),30)
  self.assertTrue(np.isfinite(g.es['weight']).all())
 def test_real_panel_completeness(self):
  root=Path(__file__).resolve().parents[1]
  dates,ids,shares,totals,audit=prepare(root/'data/raw')
  self.assertEqual(len(dates),24);self.assertEqual(len(ids),2016)
  self.assertEqual(audit['excluded_territories'],174)
  self.assertTrue(np.allclose(shares.sum(2),1));self.assertTrue((shares>=0).all())
if __name__=='__main__':unittest.main()
