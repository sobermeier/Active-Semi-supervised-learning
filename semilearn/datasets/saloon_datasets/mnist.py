import os
import numpy as np
from torchvision import transforms
from semilearn.datasets.cv_datasets.datasetbase import BasicDataset
from semilearn.datasets.augmentation import RandAugment
from semilearn.datasets.utils import split_ssl_data


def get_mnist(args, alg, num_labels, num_classes, data_dir='./data', include_lb_to_ulb=True, lb_index=None,
              ulb_index=None):
	data_dir = os.path.join(data_dir, "mnist")
	data_train = np.load(data_dir + "/mnist_x_tr.npy")
	data_test = np.load(data_dir + "/mnist_x_te.npy")
	data = np.concatenate([data_train, data_test])

	targets_train = np.load(data_dir + "/mnist_y_tr.npy")
	targets_test = np.load(data_dir + "/mnist_y_te.npy")
	targets = np.concatenate([targets_train, targets_test])

	crop_size = args.img_size
	crop_ratio = args.crop_ratio

	transform_weak = transforms.Compose([
		transforms.Resize(crop_size),
		transforms.RandomCrop(crop_size, padding=int(crop_size * (1 - crop_ratio)), padding_mode='reflect'),
		transforms.RandomHorizontalFlip(),
		transforms.ToTensor(),
		transforms.Normalize((0.1307,), (0.3081,))
	])

	transform_strong = transforms.Compose([
		transforms.Resize(crop_size),
		transforms.RandomCrop(crop_size, padding=int(crop_size * (1 - crop_ratio)), padding_mode='reflect'),
		transforms.RandomHorizontalFlip(),
		RandAugment(3, 5),
		transforms.ToTensor(),
		transforms.Normalize((0.1307,), (0.3081,))
	])

	transform_val = transforms.Compose([
		transforms.Resize(crop_size),
		transforms.ToTensor(),
		transforms.Normalize((0.1307,), (0.3081,))
	])

	lb_data, lb_targets, ulb_data, ulb_targets = split_ssl_data(args, data, targets, num_classes,
	                                                            lb_num_labels=num_labels,
	                                                            ulb_num_labels=args.ulb_num_labels,
	                                                            lb_imbalance_ratio=args.lb_imb_ratio,
	                                                            ulb_imbalance_ratio=args.ulb_imb_ratio,
	                                                            include_lb_to_ulb=include_lb_to_ulb,
	                                                            lb_index=lb_index, ulb_index=ulb_index
	                                                            )

	lb_count = [0 for _ in range(num_classes)]
	ulb_count = [0 for _ in range(num_classes)]
	for c in lb_targets:
		lb_count[c] += 1
	for c in ulb_targets:
		ulb_count[c] += 1
	print("lb count: {}".format(lb_count))
	print("ulb count: {}".format(ulb_count))

	if alg == 'fullysupervised':
		lb_data = data
		lb_targets = targets

	lb_dset = BasicDataset(alg, lb_data, lb_targets, num_classes, transform_weak, False, transform_strong, False)

	ulb_dset = BasicDataset(alg, ulb_data, ulb_targets, num_classes, transform_weak, True, transform_strong, False)

	eval_dset = BasicDataset(alg, data, targets, num_classes, transform_val, False, None, False)

	return lb_dset, ulb_dset, eval_dset

def get_mnist_imb(args, alg, name, num_labels, num_classes, data_dir='./data', include_lb_to_ulb=True, lb_index=None,
              ulb_index=None):
	data_dir = os.path.join(data_dir, name)
	data = np.load(data_dir + f"/{name}_x.npy")
	targets = np.load(data_dir + f"/{name}_y.npy")

	if name == "bcs_mnist":
		data = data.reshape(data.shape[0], 28, 28)
		print(data.shape, flush=True)

	crop_size = args.img_size
	crop_ratio = args.crop_ratio

	transform_weak = transforms.Compose([
		transforms.Resize(crop_size),
		transforms.RandomCrop(crop_size, padding=int(crop_size * (1 - crop_ratio)), padding_mode='reflect'),
		transforms.RandomHorizontalFlip(),
		transforms.ToTensor(),
		transforms.Normalize((0.1307,), (0.3081,))
	])

	transform_strong = transforms.Compose([
		transforms.Resize(crop_size),
		transforms.RandomCrop(crop_size, padding=int(crop_size * (1 - crop_ratio)), padding_mode='reflect'),
		transforms.RandomHorizontalFlip(),
		RandAugment(3, 5),
		transforms.ToTensor(),
		transforms.Normalize((0.1307,), (0.3081,))
	])

	transform_val = transforms.Compose([
		transforms.Resize(crop_size),
		transforms.ToTensor(),
		transforms.Normalize((0.1307,), (0.3081,))
	])

	lb_data, lb_targets, ulb_data, ulb_targets = split_ssl_data(args, data, targets, num_classes,
	                                                            lb_num_labels=num_labels,
	                                                            ulb_num_labels=args.ulb_num_labels,
	                                                            lb_imbalance_ratio=args.lb_imb_ratio,
	                                                            ulb_imbalance_ratio=args.ulb_imb_ratio,
	                                                            include_lb_to_ulb=include_lb_to_ulb,
	                                                            lb_index=lb_index, ulb_index=ulb_index
	                                                            )

	lb_count = [0 for _ in range(num_classes)]
	ulb_count = [0 for _ in range(num_classes)]
	for c in lb_targets:
		lb_count[c] += 1
	for c in ulb_targets:
		ulb_count[c] += 1
	print("lb count: {}".format(lb_count))
	print("ulb count: {}".format(ulb_count))

	if alg == 'fullysupervised':
		lb_data = data
		lb_targets = targets

	lb_dset = BasicDataset(alg, lb_data, lb_targets, num_classes, transform_weak, False, transform_strong, False)

	ulb_dset = BasicDataset(alg, ulb_data, ulb_targets, num_classes, transform_weak, True, transform_strong, False)


	test_data, test_targets = data, targets
	eval_dset = BasicDataset(alg, test_data, test_targets, num_classes, transform_val, False, None, False)

	return lb_dset, ulb_dset, eval_dset
