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

from semilearn.datasets.saloon_datasets.saloon_dataset import SaloonDataset
from semilearn.datasets.utils import split_ssl_data
from semilearn.datasets.augmentation import RandAugment, RandomResizedCropAndInterpolation, str_to_interp_mode
from semilearn.datasets.cv_datasets.datasetbase import BasicDataset, LoaderDataset, default_loader

mean, std = {}, {}
mean['dollarstreet'] = [0.485, 0.456, 0.406]
std['dollarstreet'] = [0.229, 0.224, 0.225]

def get_dollarstreet(args, alg, name, num_labels, num_classes, data_dir='./data', include_lb_to_ulb=True, lb_index=None, ulb_index=None, no_labels = False):
	img_size = args.img_size
	crop_ratio = args.crop_ratio

	print("in getter: ", num_labels, flush=True)


	if name == "dollarstreet_imagenet":
		prefix = "imagenet"
	elif name == "dollarstreet_country":
		prefix = "country"
	else:
		prefix = "income"

	transform_weak = transforms.Compose([
		transforms.Resize((int(math.floor(img_size / crop_ratio)), int(math.floor(img_size / crop_ratio)))),
		transforms.RandomCrop((img_size, img_size)),
		transforms.RandomHorizontalFlip(),
		transforms.ToTensor(),
		transforms.Normalize(mean['dollarstreet'], std['dollarstreet'])
	])

	transform_strong = transforms.Compose([
		transforms.Resize((int(math.floor(img_size / crop_ratio)), int(math.floor(img_size / crop_ratio)))),
		RandomResizedCropAndInterpolation((img_size, img_size)),
		transforms.RandomHorizontalFlip(),
		RandAugment(3, 10),
		transforms.ToTensor(),
		transforms.Normalize(mean['dollarstreet'], std['dollarstreet'])
	])

	transform_val = transforms.Compose([
		transforms.Resize(math.floor(int(img_size / crop_ratio))),
		transforms.CenterCrop(img_size),
		transforms.ToTensor(),
		transforms.Normalize(mean['dollarstreet'], std['dollarstreet'])
	])

	data_dir = os.path.join(data_dir, "dataset_dollarstreet")

	dataset = SaloonDataset(root=os.path.join(data_dir, f"{prefix}_train"), transform=transform_weak, ulb=False, alg=alg, strong_transform=transform_strong)
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



	print("eval")
	eval_dset = SaloonDataset(root=os.path.join(data_dir, f"{prefix}_test"), transform=transform_val, alg=alg, ulb=False)


	return lb_dset, ulb_dset, eval_dset




