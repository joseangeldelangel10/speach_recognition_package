class numpy_matrix_list():
	"""
	numpy_matrix_list is a class designed to hold
	several 2D numpy matrices with different shapes
	and perform operations with such matrices
	"""
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
			raise Exception('adding matrices are not the same lenght')
			return
		else:
			res = []
			for i in range(len(self.list)):
				res.append(self.list[i]+l2.list[i])
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
			res += "______________________________\n"
		return res

	def shape(self):
		s = []
		for i in self.list:
			s.append(i.shape)
		return s


