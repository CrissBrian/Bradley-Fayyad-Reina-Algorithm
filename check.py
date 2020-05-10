import os
import argparse
import numpy as np
from sklearn.metrics.cluster import normalized_mutual_info_score

def read_input(file_path):
	with open(file_path, 'r') as f:
		a = [line.split(',')[1] for line in f]
	return a

def read_output(file_path):
	a = []
	with open(file_path, 'r') as f:
		key = 'The clustering results:\n'
		flag = False
		for line in f:
			if flag:
				a.append(line.split(',')[1])
			if line	== key:
				flag = True
	return a

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("input", default="hw6_clustering.txt", help="input yelp file")
	parser.add_argument("output", default="output", help="output file")
	args = parser.parse_args()

	a = read_input(file_path=args.input)
	b = read_output(file_path=args.output)

	print("Info Score is", normalized_mutual_info_score(a, b))

if __name__ == '__main__':
	main()