import numpy as np
class numpy_matrix_list():
	def __init__(self,matrix_list):
		self.list = matrix_list

	def __sub__(self, l2):
		if len(self.list) != len(l2.list):
			raise Exception('resting matrices are not the same lenght')
			return
		else:
			res = []
			for i in range(len(self.list)):
				res.append(self.list[i]-l2.list[i])
			return numpy_matrix_list(res)

	def __add__(self,l2):
		if len(self.list) != len(l2.list):
			raise Exception('resting matrices are not the same lenght')
			return
		else:
			res = []
			for i in range(len(self.list)):
				res.append(self.list[i]+l2.list[i])
			return numpy_matrix_list(res)

	def __mul__(self,l2):
		res = []
		if type(l2) == int or type(l2) == float:
			for i in range(len(self.list)):
				res.append(self.list[i]*l2)
			return numpy_matrix_list(res)

	def scalar_mult(self,l2):
		res = []
		for i in range(len(self.list)):
			res.append(self.list[i]*l2)
		return numpy_matrix_list(res)

	def __getitem__(self, key):
		return self.list[key]

	def __str__(self):
		res = "_____numpy_matrix_list:_____ \n"
		for i in range(len(self.list)):
			res += str(self.list[i])
			res += "\n\n"
			res += "______________________________"
		return res

	@staticmethod
	def len(l1):
		return len(l1.list)

