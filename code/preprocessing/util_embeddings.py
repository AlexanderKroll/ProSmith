import os
from os.path import join


def create_empty_path(path):
	try:
		os.mkdir(path)
	except:
		pass

	all_files = os.listdir(path)
	for file in all_files:
		os.remove(join(path, file))
