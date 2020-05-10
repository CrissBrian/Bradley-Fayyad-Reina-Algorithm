from pyspark import SparkContext
import os
import argparse
import time
import numpy as np
from sklearn.cluster import KMeans
from collections import namedtuple, defaultdict

Centroid = namedtuple("Centroid", ["N", "SUM", "SUMSQ"])

class BFR(object):
	def __init__(self, num_partition):
		self._num_partition = num_partition
		self._result = None

	def readfile(self, spark_context, input_file_path):
		sc = spark_context
		RDD = sc.textFile(input_file_path) \
				.repartition(self._num_partition)
		return RDD

	def data_process(self, rdd):
		rdd = rdd.map(self.get_split)
		data = np.array(rdd.collect(), dtype=np.float64)
		length = rdd.count()
		self._result = np.full(length, -1)
		data = np.array_split(data,5)
		return data

	def first_round(self, data, n_clusters):
		kmeans = KMeans(n_clusters=5*n_clusters, random_state=0).fit(data[:,2:])
		RS = self.find_RS(clusters=kmeans.labels_)
		data, RS = self.split_RS(data=data, single_clusters=RS)
		kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(data[:,2:])
		DS = self.get_DS(data=data, 
			clusters=kmeans.labels_, n_clusters=n_clusters)
		return DS, RS, len(data)

	def inter_round(self, data, DS, CS_c, CS_p, RS, dps, n_clusters):
		threshold = 2 * np.sqrt(len(data[0][2:]))
		for point in data:
			index = self.check_set(t=threshold, p=point, S=DS)
			# print(index)
			if index != -1:
				DS = self.update_DS(i=index, p=point, S=DS)
				self._result[int(point[0])] = index
				dps += 1
			else:
				index = self.check_set(t=threshold, p=point, S=CS_c)
				if index != -1:
					CS_c, CS_p = self.update_CS(i=index, p=point, CS_c=CS_c, CS_p=CS_p)
				else:
					RS = np.concatenate((RS, [point]), axis=0)

		# my_n_clusters = min(5*n_clusters, len(RS))
		if len(RS) > 5*n_clusters:
			kmeans = KMeans(n_clusters=5*n_clusters, random_state=0).fit(RS[:,2:])
			CS_c1, CS_p1, RS = self.form_CSRS(n_clusters=5*n_clusters, clusters=kmeans.labels_, data=RS)
			CS_c.extend(CS_c1)
			CS_p.extend(CS_p1)
		# self.print_len(CS_p)
		if len(CS_c) > 1:
			CS_c, CS_p = self.merge_all_CS(CS_c=CS_c, CS_p=CS_p)
		# self.print_len(CS_p)
		return DS, CS_c, CS_p, RS, dps

	def merge_DSCS(self, DS, CS_c, CS_p, dps):
		threshold = 2 * np.sqrt(len(DS[0].SUM))

		for i,c in enumerate(CS_c):
			point = np.append([0,0], c.SUM / c.N)
			index = self.check_set_final(t=threshold, p=point, S=DS)
			if index != -1:
				DS = self.update_DS(i=index, p=point, S=DS)
				# print("jinlaile")
				for p in CS_p[i]:
					self._result[int(p)] = index
					dps += 1
				CS_c[i] = []
				CS_p[i] = []
		CS_c, CS_p = self.update_merged_CS(CS_c=CS_c, CS_p=CS_p)
		# self.print_len(CS_p)
		return DS, dps, CS_c, CS_p


	def merge_all_CS(self, CS_c, CS_p):
		max_len = len(CS_c)
		threshold = 2 * np.sqrt(len(CS_c[0].SUM))
		flag = True
		while flag:
			flag = False
			for i in range(max_len):
				for j in range(i+1, max_len):
					if CS_c[i]!=[] and CS_c[j]!=[]:
						# if self.mahalanobis_c(c1=CS_c[i], c2=CS_c[j]) < threshold:
						if self.mahalanobis_merge(c1=CS_c[i], c2=CS_c[j]):
							CS_c, CS_p = self.merge_CS(CS_c=CS_c, CS_p=CS_p, i=i, j=j)
							flag = True
			CS_c, CS_p = self.update_merged_CS(CS_c=CS_c, CS_p=CS_p)
			max_len = len(CS_c)
			# self.print_len(CS_c)
			# self.print_len(CS_p)
		return CS_c, CS_p

	def merge_CS(self, CS_c, CS_p, i, j):
		c1 = CS_c[i]
		c2 = CS_c[j]
		CS_c[i] = Centroid(N=c1.N+c2.N, SUM=c1.SUM+c2.SUM, SUMSQ=c1.SUMSQ+c2.SUMSQ)
		CS_c[j] = []
		CS_p[i].extend(CS_p[j])
		CS_p[j] = []
		return CS_c, CS_p

	def update_merged_CS(self, CS_c, CS_p):
		CS_c = [i for i in CS_c if len(i)>0]
		CS_p = [i for i in CS_p if len(i)>0]
		return CS_c, CS_p

	def form_CSRS(self, n_clusters, clusters, data):
		SUM = np.zeros((n_clusters, len(data[0,2:])))
		SUMSQ = np.zeros((n_clusters, len(data[0,2:])))
		count = np.bincount(clusters)
		single = np.where(count==1)[0]
		CS_dict = defaultdict(list)
		RS = []
		for i,c in enumerate(clusters):
			if c in single:
				RS.append(data[i])
			else:
				SUM[c,:] += data[i,2:]
				SUMSQ[c,:] += np.square(data[i,2:])
				CS_dict[c].append(data[i][0])
		CS_c = []
		CS_p = []
		for i in range(n_clusters):
			if i not in single:
				CS_c.append(Centroid(N=count[i], SUM=SUM[i], SUMSQ=SUMSQ[i]))
				CS_p.append(CS_dict[i])
		# print(len(CS_c))
		# print(len(CS_p))
		# print((CS_c[0]))
		# print((CS_p[0]))
		return CS_c, CS_p, RS
		
	def check_set(self, t, p, S):
		_min = float('inf')
		index = -1
		for i, centroid in enumerate(S):
			d = self.mahalanobis(c=centroid, p=p)
			if d < _min:
				_min = d
				index = i
		if _min < t:
			return index
		return -1

	def check_set_final(self, t, p, S):
		_min = float('inf')
		index = -1
		for i, centroid in enumerate(S):
			d = self.mahalanobis(c=centroid, p=p)
			if d < _min:
				_min = d
				index = i

		# print(_min, t)
		# print(_min < t)
		if _min < t:
			return index
		return -1

	def update_DS(self, i, p, S):
		c = S[i]
		centroid = Centroid(N=c.N+1, SUM=c.SUM+p[2:], SUMSQ=c.SUMSQ+np.square(p[2:]))
		S[i] = centroid
		return S

	def update_CS(self, i, p, CS_c, CS_p):
		c = CS_c[i]
		centroid = Centroid(N=c.N+1, SUM=c.SUM+p[2:], SUMSQ=c.SUMSQ+np.square(p[2:]))
		CS_c[i] = centroid
		CS_p[i].append(p[0])
		return CS_c, CS_p

	def get_DS(self, data, clusters, n_clusters):
		N = np.bincount(clusters)
		DS = []
		for i in range(n_clusters):
			SUM = np.dot(clusters==i, data[:,2:])
			SUMSQ = np.dot(clusters==i, np.square(data[:,2:]))
			DS.append(Centroid(N=N[i], SUM=SUM, SUMSQ=SUMSQ))
		for i, c in enumerate(clusters):
			self._result[int(data[i,0])] = c
		return DS

	def find_RS(self, clusters):
		count = np.bincount(clusters)
		single = np.where(count==1)[0]
		single_clusters = []
		for i,v in enumerate(clusters):
			if v in single:
				single_clusters.append(i)
		return single_clusters

	@staticmethod
	def split_RS(data, single_clusters):
		data = np.delete(data, single_clusters, axis=0)
		RS = data[single_clusters].astype(np.float64)
		return data, RS

	@staticmethod
	def print_result(data, output_file_path):
		with open(output_file_path, 'a') as f:
			f.write("\nThe clustering results:\n")
			for i,d in enumerate(data):
				f.write(str(i)+','+str(d))
				f.write("\n")

	@staticmethod
	def print_round(i, dps, ncs, cps, rps, output_file_path):
		if i==0:
			with open(output_file_path, 'w') as f:
				f.write("The intermediate results:\n")
		with open(output_file_path, 'a') as f:
			f.write('Round {i}: {dps},{ncs},{cps},{rps}'
				.format(i=str(i+1), dps=str(dps), ncs=str(ncs), cps=str(int(cps)), rps=str(rps)))
			f.write("\n")

	@staticmethod
	def mahalanobis(c, p):
		centroid = c.SUM / c.N
		sigma = c.SUMSQ / c.N - np.square(centroid)
		d = np.sqrt(np.sum(np.square(p[2:] - centroid) / sigma))
		return d

	### this formula is wrong!!!
	@staticmethod
	def mahalanobis_c(c1, c2):
		### wrong way
		centroid1 = c1.SUM / c1.N
		centroid2 = c2.SUM / c2.N
		sigma1 = np.sqrt(c1.SUMSQ / c1.N - np.square(centroid1))
		sigma2 = np.sqrt(c2.SUMSQ / c2.N - np.square(centroid2))
		d = np.sqrt(np.sum(np.square(centroid1 - centroid2) / (sigma1 * sigma2)))
		return d

	### try sigma_merged < (sigma_a + sigma_b) * 0.5
	@staticmethod
	def mahalanobis_merge(c1, c2):
		c3 = Centroid(N=c1.N+c2.N, SUM=c1.SUM+c2.SUM, SUMSQ=c1.SUMSQ+c2.SUMSQ)
		centroid1 = c1.SUM / c1.N
		centroid2 = c2.SUM / c2.N
		centroid3 = c3.SUM / c3.N
		sigma1 = np.sqrt(c1.SUMSQ / c1.N - np.square(centroid1))
		sigma2 = np.sqrt(c2.SUMSQ / c2.N - np.square(centroid2))
		sigma3 = np.sqrt(c3.SUMSQ / c3.N - np.square(centroid3))
		# print((sigma3 < ((sigma1 + sigma2) * 0.5)).all())
		return (sigma3 < ((sigma1 + sigma2) * 0.5)).all()

	@staticmethod
	def get_split(x):
		x = x.split(',')
		return x

	@staticmethod
	def print_len(x):
		output = [len(i) for i in x]
		print(output)

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("input", default="hw6_clustering.txt", help="input yelp file")
	parser.add_argument("n_cluster", default="output.csv", help="output file")
	parser.add_argument("output", default="output.csv", help="output file")
	args = parser.parse_args()

	sc = SparkContext('local[*]', 'FrequentItems')
	sc.setLogLevel("ERROR")

	start_time = time.time()
	n_clusters = int(args.n_cluster)
	bfr = BFR(num_partition=5)
	rdd = bfr.readfile(spark_context=sc, input_file_path=args.input)
	data = bfr.data_process(rdd=rdd)
	DS, RS, dps = bfr.first_round(data=data[0], n_clusters=n_clusters)
	CS_c, CS_p = [], []

	bfr.print_round(i=0, dps=dps, ncs=0, cps=0, rps=len(RS), output_file_path=args.output)
	
	for i in range(1,5):
		DS, CS_c, CS_p, RS, dps = bfr.inter_round(data=data[i], DS=DS, CS_c=CS_c, CS_p=CS_p, RS=RS, dps=dps, n_clusters=n_clusters)
		if i == 4:
			DS, dps, CS_c, CS_p = bfr.merge_DSCS(DS=DS, CS_c=CS_c, CS_p=CS_p, dps=dps)
		bfr.print_round(i=i, dps=dps, ncs=len(CS_c), cps=np.sum([len(i) for i in CS_p]), rps=len(RS), output_file_path=args.output)
		print(i)

	bfr.print_result(data=bfr._result, output_file_path=args.output)
	
	total_time = time.time() - start_time
	print("Duration:", total_time)

if __name__ == '__main__':
	main()