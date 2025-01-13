import os
import gc
import copy
import json
import random
from torchvision.datasets import ImageFolder
from PIL import Image
from torchvision import transforms
import math
from semilearn.datasets.cv_datasets.datasetbase import BasicDataset, default_loader

class SaloonDataset(BasicDataset, ImageFolder):
	def __init__(self, root, transform, ulb, alg, strong_transform=None, percentage=-1, include_lb_to_ulb=True, lb_index=None):
		self.alg = alg
		self.is_ulb = ulb
		self.percentage = percentage
		self.transform = transform
		self.root = root
		self.include_lb_to_ulb = include_lb_to_ulb
		self.lb_index = lb_index

		is_valid_file = None
		extensions = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')
		classes, class_to_idx = self.find_classes(self.root)
		samples = self.make_dataset(self.root, class_to_idx, extensions, is_valid_file)
		if len(samples) == 0:
			msg = "Found 0 files in subfolders of: {}\n".format(self.root)
			if extensions is not None:
				msg += "Supported extensions are: {}".format(",".join(extensions))
			raise RuntimeError(msg)

		self.loader = default_loader
		self.extensions = extensions

		self.classes = classes
		self.class_to_idx = class_to_idx
		self.data = [s[0] for s in samples]
		self.targets = [s[1] for s in samples]

		count = [0 for _ in range(len(classes))]
		for c in self.targets:
			count[c] += 1
		print("count: {}".format(count))

		self.strong_transform = strong_transform
		if self.strong_transform is None:
			if self.is_ulb:
				assert self.alg not in ['fullysupervised', 'supervised', 'pseudolabel', 'vat', 'pimodel', 'meanteacher', 'mixmatch'], f"alg {self.alg} requires strong augmentation"


	def __sample__(self, index):
		path = self.data[index]
		sample = self.loader(path)
		target = self.targets[index]
		return sample, target

	def make_dataset(
			self,
			directory,
			class_to_idx,
			extensions=None,
			is_valid_file=None,
	):
		instances = []
		directory = os.path.expanduser(directory)
		both_none = extensions is None and is_valid_file is None
		both_something = extensions is not None and is_valid_file is not None
		if both_none or both_something:
			raise ValueError("Both extensions and is_valid_file cannot be None or not None at the same time")
		if extensions is not None:
			def is_valid_file(x: str) -> bool:
				return x.lower().endswith(extensions)

		lb_idx = {}
		for target_class in sorted(class_to_idx.keys()):
			class_index = class_to_idx[target_class]
			target_dir = os.path.join(directory, target_class)
			if not os.path.isdir(target_dir):
				continue
			for root, _, fnames in sorted(os.walk(target_dir, followlinks=True)):
				random.shuffle(fnames)
				if self.percentage != -1:
					fnames = fnames[:int(len(fnames) * self.percentage)]
				if self.percentage != -1:
					lb_idx[target_class] = fnames
				for fname in fnames:
					if not self.include_lb_to_ulb:
						if fname in self.lb_index[target_class]:
							continue
					path = os.path.join(root, fname)
					if is_valid_file(path):
						item = path, class_index
						instances.append(item)
		gc.collect()
		self.lb_idx = lb_idx
		return instances