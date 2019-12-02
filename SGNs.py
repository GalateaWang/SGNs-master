# -*- coding: UTF-8 -*-
import argparse
import numpy as np
import networkx as nx
import classification
from itertools import chain
from sklearn.decomposition import PCA
import os


def parse_args():
	'''
	Parses the SGN arguments.
	'''
	parser = argparse.ArgumentParser(description="Run SGNs.")

	parser.add_argument('--input', nargs='?', default='mutag',
	                    help='Input graph path')

	parser.add_argument('--label', nargs='?', default='mutag.Labels',
	                    help='Input graph path')

	parser.add_argument('--types', type=int, default=1,
	                    help='Type of processing the features. Default is 1 or 2.')

	parser.add_argument('--N', type=int, default=2,  ## SGN0, SGN1
	                    help='Number of convert to line graph. Default is 3.')

	return parser.parse_args()


def read_graph(fullpath):
	'''
	Reads the input network in networkx.
	'''
	G = nx.read_edgelist(fullpath, delimiter=',', nodetype=int, data=(('weight',float),), create_using=nx.DiGraph())
	return G


def read_label():
	with open(args.label) as f:
		singlelist = [line.strip()[-1] for line in f.readlines()]
		labels = np.array(singlelist)

	return labels


def feature_processing(all_features):
	'''
	:param all_features:
	      1:PCA;
	      2:get the mean value
	:return: the reduced dimension network feature vector.
	'''
	if args.types == 1:
		fea_list = []
		for fea in all_features:
			features = list(chain(*fea))  ## flatten a list
			fea_list.append(features)
		x = np.array(fea_list)
		pca = PCA(n_components=11)   # decided by the feature abstract method.
		reduced_x = pca.fit_transform(x)

	elif args.types == 2:
		fea_list = []
		for fea in all_features:
			features = np.array(fea)
			fea_list.append(list(np.mean(features, axis=0)))
		reduced_x = np.array(fea_list)

	elif args.types == 3:
		fea_list = []
		for fea in all_features:
			features = list(chain(*fea))  ## flatten a list
			fea_list.append(features)
		reduced_x = np.array(fea_list)

	else:
		raise Exception("Invalid feature processing type!", type)

	return reduced_x


def to_line(graph):
	'''
	:param graph
	:return G_line: line/Subgraph network
	'''
	graph_to_line = nx.line_graph(graph)
	graph_line = nx.convert_node_labels_to_integers(graph_to_line, first_label=0, ordering='default')
	return graph_line


def main(args, fullpath):
	'''
	Transform graphs to different-order SGNs.
	'''
	nx_G = read_graph(fullpath)
	cur= nx_G

	sgn_features = list()
	sgn_features.append(classification.character(nx_G))  ## add the original network feature

	for n in range(args.N):  ## line1,line2,line3
		sgn = to_line(cur)
		cur = sgn
		feature = classification.character(cur)  ## get the feature of one network
		sgn_features.append(feature)

	return sgn_features


if __name__ == "__main__":

	args = parse_args()

	all_features = []
	files = os.listdir(args.input)
	files.sort(key=lambda x: int(x.split('.')[0]))
	for path in files:
		full_path = os.path.join(args.input, path)
		print('num_graph:', path)
		features_list = main(args, full_path)
		all_features.append(features_list)

	reduced_x = feature_processing(all_features)
	labels = read_label()
	classification.result_class(reduced_x, labels)
