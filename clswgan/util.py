#import h5py
import numpy as np
import scipy.io as sio
import torch
from sklearn import preprocessing
import sys


def weights_init(m):
	classname = m.__class__.__name__
	if classname.find('Linear') != -1:
		m.weight.data.normal_(0.0, 0.02)
		m.bias.data.fill_(0)
	elif classname.find('BatchNorm') != -1:
		m.weight.data.normal_(1.0, 0.02)
		m.bias.data.fill_(0)

def map_label(label, classes):
	mapped_label = torch.LongTensor(label.size())
	for i in range(classes.size(0)):
		mapped_label[label==classes[i]] = i

	return mapped_label


class DATA_LOADER(object):


	def __init__(self, opt):
		self.read_matdataset(opt)
		self.index_in_epoch = 0
		self.epochs_completed = 0


	def read_matdataset(self, opt):
		matcontent = sio.loadmat(opt.dataroot + "/" + opt.dataset + "/" + opt.image_embedding + ".mat")
		feature = matcontent['features'].T
		label = matcontent['labels'].astype(int).squeeze() - 1
		matcontent = sio.loadmat(opt.dataroot + "/" + opt.dataset + "/" + opt.class_embedding + "_splits" + opt.class_embedding_type + ".mat")
		# numpy array index starts from 0, matlab starts from 1
		trainval_loc = matcontent['trainval_loc'].squeeze() - 1
		# train_loc = matcontent['train_loc'].squeeze() - 1
		# val_unseen_loc = matcontent['val_loc'].squeeze() - 1
		test_seen_loc = matcontent['test_seen_loc'].squeeze() - 1
		test_unseen_loc = matcontent['test_unseen_loc'].squeeze() - 1

		np.set_printoptions(threshold=np.inf)
		# print(trainval_loc)
		# print(test_seen_loc)
		# trainval_loc = np.sort(trainval_loc)
		# test_seen_loc = np.sort(test_seen_loc)
		# print(trainval_loc)
		# print(test_seen_loc)

		self.attribute = torch.from_numpy(matcontent['att'].T).float()
		if opt.standardization:
			print('standardization...')
			scaler = preprocessing.StandardScaler()
		else:
			scaler = preprocessing.MinMaxScaler()
		_train_feature = scaler.fit_transform(feature[trainval_loc])
		_test_seen_feature = scaler.transform(feature[test_seen_loc])
		_test_unseen_feature = scaler.transform(feature[test_unseen_loc])
		self.train_feature = torch.from_numpy(_train_feature).float()
		mx = self.train_feature.max()
		self.train_feature.mul_(1/mx)
		self.train_label = torch.from_numpy(label[trainval_loc]).long()
		self.test_unseen_feature = torch.from_numpy(_test_unseen_feature).float()
		self.test_unseen_feature.mul_(1/mx)
		self.test_unseen_label = torch.from_numpy(label[test_unseen_loc]).long()
		self.test_seen_feature = torch.from_numpy(_test_seen_feature).float()
		self.test_seen_feature.mul_(1/mx)
		self.test_seen_label = torch.from_numpy(label[test_seen_loc]).long()


		print(self.test_unseen_label)

		self.seenclasses = torch.from_numpy(np.unique(self.train_label.numpy()))
		self.unseenclasses = torch.from_numpy(np.unique(self.test_unseen_label.numpy()))
		self.ntrain = self.train_feature.size()[0]
		self.ntrain_class = self.seenclasses.size(0)

		print(self.seenclasses)
		print(self.unseenclasses)
		# exit(0)

    #CORRELAZIONE CLASSI SEEN (TRAINING)

		seen_cls_corr = []
		for cl_id1, cl1 in enumerate(self.attribute):
			cls_correlations = []

			if cl_id1 in self.seenclasses:
				for cl_id2, cl2 in enumerate(self.attribute):
					if cl_id2 in self.seenclasses:
						cl1_2_corr = np.correlate(cl1,cl2)
						cls_correlations.append(cl1_2_corr)
				seen_cls_corr.append(np.sum(cls_correlations))
		print("Seen classes mean correlation : "+str(np.mean(seen_cls_corr)))



		#CORRELAZIONE CLASSI UNSEEN (TEST)

		unseen_cls_corr = []
		for cl_id1, cl1 in enumerate(self.attribute):
			cls_correlations = []

			if cl_id1 in self.unseenclasses:
				for cl_id2, cl2 in enumerate(self.attribute):
					if cl_id2 in self.unseenclasses:
						cl1_2_corr = np.correlate(cl1, cl2)
						cls_correlations.append(cl1_2_corr)
				unseen_cls_corr.append(np.sum(cls_correlations))
		print("Uneen classes mean correlation : " + str(np.mean(unseen_cls_corr)))



		self.ntest_class = self.unseenclasses.size(0)
		self.train_class = self.seenclasses.clone()
		self.allclasses = torch.arange(0, self.ntrain_class+self.ntest_class).long()
		self.train_mapped_label = map_label(self.train_label, self.seenclasses)


	def next_batch(self, batch_size):
		idx = torch.randperm(self.ntrain)[0:batch_size]
		batch_feature = self.train_feature[idx]
		batch_label = self.train_label[idx]
		batch_att = self.attribute[batch_label]
		return batch_feature, batch_label, batch_att
