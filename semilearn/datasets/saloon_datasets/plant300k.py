# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import gc
import copy
import json
import random
from torchvision.datasets import ImageFolder
from PIL import Image
from torchvision import transforms
import math

from semilearn.datasets.utils import split_ssl_data
from semilearn.datasets.augmentation import RandAugment, RandomResizedCropAndInterpolation, str_to_interp_mode
from semilearn.datasets.cv_datasets.datasetbase import BasicDataset, LoaderDataset, default_loader

mean, std = {}, {}
mean['plant300k'] = [0.485, 0.456, 0.406]
std['plant300k'] = [0.229, 0.224, 0.225]

def get_plant300k(args, alg, name, num_labels, num_classes, data_dir='./data', include_lb_to_ulb=True, lb_index=None, ulb_index=None, no_labels = False):
	img_size = args.img_size
	crop_ratio = args.crop_ratio

	print("in getter: ", num_labels, flush=True)

	transform_weak = transforms.Compose([
		transforms.Resize((int(math.floor(img_size / crop_ratio)), int(math.floor(img_size / crop_ratio)))),
		transforms.RandomCrop((img_size, img_size)),
		transforms.RandomHorizontalFlip(),
		transforms.ToTensor(),
		transforms.Normalize(mean['plant300k'], std['plant300k'])
	])

	transform_strong = transforms.Compose([
		transforms.Resize((int(math.floor(img_size / crop_ratio)), int(math.floor(img_size / crop_ratio)))),
		RandomResizedCropAndInterpolation((img_size, img_size)),
		transforms.RandomHorizontalFlip(),
		RandAugment(3, 10),
		transforms.ToTensor(),
		transforms.Normalize(mean['plant300k'], std['plant300k'])
	])

	transform_val = transforms.Compose([
		transforms.Resize(math.floor(int(img_size / crop_ratio))),
		transforms.CenterCrop(img_size),
		transforms.ToTensor(),
		transforms.Normalize(mean['plant300k'], std['plant300k'])
	])

	data_dir = os.path.join(data_dir, "plantnet_300K/images")

	dataset = Plant300kDataset(root=os.path.join(data_dir, "train"), transform=transform_weak, ulb=False, alg=alg, strong_transform=transform_strong)
	data, targets = dataset.data, dataset.targets
	percentage = num_labels / len(dataset)

	lb_data, lb_targets, ulb_data, ulb_targets = split_ssl_data(args, data, targets, num_classes,
																lb_num_labels=num_labels,
																ulb_num_labels=args.ulb_num_labels,
																lb_imbalance_ratio=args.lb_imb_ratio,
																ulb_imbalance_ratio=args.ulb_imb_ratio,
																include_lb_to_ulb=include_lb_to_ulb,
																lb_index=lb_index, ulb_index=ulb_index, no_labels = no_labels
																)

	lb_data = lb_data.tolist()
	lb_targets = lb_targets.tolist()
	ulb_data = ulb_data.tolist()
	ulb_targets = ulb_targets.tolist()

	lb_count = [0 for _ in range(num_classes)]
	ulb_count = [0 for _ in range(num_classes)]
	for c in lb_targets:
		lb_count[c] += 1
	for c in ulb_targets:
		ulb_count[c] += 1

	#data_vals = [lb_data[idx] for idx in range(len(lb_data))]
	#print(data_vals)
	print("lb count: {}".format(lb_count))
	print("ulb count: {}".format(ulb_count))

	if alg == 'fullysupervised':
		lb_data = data
		lb_targets = targets
	lb_dset = LoaderDataset(alg, lb_data, lb_targets, num_classes, transform_weak, False, transform_strong, False)
	#print("basic", type(lb_dset.data), flush=True)
	#print("basic", lb_dset.data[0], flush=True)
	#print("basic", type(lb_dset.data[0]), flush=True)
	ulb_dset = LoaderDataset(alg, ulb_data, ulb_targets, num_classes, transform_weak, True, transform_strong, False)

	#rint("lb")
	#lb_dset = Plant300kDataset(root=os.path.join(data_dir, "train"), transform=transform_weak, ulb=False, alg=alg, percentage=percentage)
	#print("plant", type(lb_dset.data), flush=True)
	#print("plant", lb_dset.data[0], flush=True)
	#print("plant", type(lb_dset.data[0]), flush=True)
	#print("ulb")
	#ulb_dset = Plant300kDataset(root=os.path.join(data_dir, "train"), transform=transform_weak, alg=alg, ulb=True, strong_transform=transform_strong, include_lb_to_ulb=include_lb_to_ulb, lb_index=lb_dset.lb_idx)



	print("val")
	val_dset = Plant300kDataset(root=os.path.join(data_dir, "val"), transform=transform_val, alg=alg, ulb=False)


	print("eval")
	eval_dset = Plant300kDataset(root=os.path.join(data_dir, "test"), transform=transform_val, alg=alg, ulb=False)

	return lb_dset, ulb_dset, val_dset, eval_dset



class Plant300kDataset(BasicDataset, ImageFolder):
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

		count = [0 for _ in range(1081)]
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

