'''
Notes so far:

- Most of these split don't change much
- Some of these split dramatically worsen results
- Sometimes, random splits are the best.

- Why are random splits so good? And how (if) can they be improved?
'''


from sys import path
import numpy as np
import scipy.io as sio
import torch



# np.set_printoptions(threshold=np.inf)
PATH = '../content/gdrive/MyDrive/Colab_Notebooks/tfvaegan/datasets/data/FLO1'
SPLIT = 'rnd'					# 'rnd', 'gcs', 'ccs', 'mas', 'mcs', 'pca'
# IMPORTANT before running attribute-based splits, reset att_splits in folder
PCA_COMPONENTS = 200
LDA_COMPONENTS = 50
INV = False
SAVE = True


''' Random Split (RND)
Control split
'''
def random_split(features, labels, attributes, inverse):
	old_seen = torch.from_numpy(np.unique(seen_labels))
	old_unseen = torch.from_numpy(np.unique(unseen_labels))
	old_classes = np.concatenate((old_seen, old_unseen))
	np.random.shuffle(old_classes)
	new_seen = old_classes[:int(n_seen_classes)]
	new_unseen = old_classes[int(n_seen_classes):]
	return new_seen, new_unseen, attributes


''' Greedy Class Split (GCS)
Tries to avoid the "horse with stripes without stripes images" scenario by keeping as much semantic information as possible among the seen classes.
In the binary definition of the semantic space, the value 1 indicates the presence of an attribute in an image, while the value 0 indicates its absence.
This means that ones are more useful than zeros, so we maximize the former in the seen classes split.
In other words, for each class, we simply sum the values of its signature vector and we sort the classes by these sums in descending order.
Consequently, we select the first Ns classes as seen classes, and the other Nu as unseen classes.
'''
def greedy_class_split(features, labels, attributes, inverse):
	# for each class, sum the values of its signature vector
	sums = np.sum(attributes, axis=1)
	# sorted_sums = np.sort(sums)
	sorted_sums = np.argsort(sums)
	new_seen = sorted_sums[:n_seen_classes] if inverse else sorted_sums[n_unseen_classes:]
	new_unseen = sorted_sums[n_seen_classes:] if inverse else sorted_sums[:n_unseen_classes]
	return new_seen, new_unseen, attributes


''' Clustered Class Split (CCS)
Tries to maximize the Class Semantic Distance between seen classes and unseen classes.
We define the Class Semantic Distance matrix where each element is the euclidean distance between class two class signatures (attribute vectors).
Seen and unseen classes are defined by sorting the classes by the sum of their row (or column) values in descending order.
The first Ns classes are those with the lowest distances overall, meaning that they form a cluster in the semantic space. Those classes will be the seen classes.
The other Nu are far from this cluster in the semantic space, so they will form another cluster
(although it is not a proper cluster since those classes are probably far away from each other as well), and they will be the unseen classes.
'''
def clustered_class_split(features, labels, attributes, inverse):
	distances = []
	for a1 in attributes:
		att_distances = []
		for a2 in attributes:
			d = np.linalg.norm(a1 - a2)
			att_distances.append(d)
		sum_att_distances = np.sum(att_distances)
		distances.append(sum_att_distances)
	sorted_distances = np.argsort(distances)			# from smaller to largest sum
	new_seen = sorted_distances[:n_seen_classes] if inverse else sorted_distances[n_unseen_classes:]
	new_unseen = sorted_distances[n_seen_classes:] if inverse else sorted_distances[:n_unseen_classes]
	return new_seen, new_unseen, attributes

"""
LEAST CORRELATED CLASS SPLIT (LCCS):
The idea here is to use a set of seen classes which attributes
are the least correlated to others.
This is done in order to check if using a more diversified set helps
with the learning phase
"""

def least_correlated_class_split(features, labels, attributes, inverse):
    from collections import defaultdict
    unique_lalbels = np.unique(labels)
    class_correlations = defaultdict(int)
    for c1 in unique_lalbels:
        for c2 in unique_lalbels:
            corr_12 = np.correlate(attributes[c1], attributes[c2])
            class_correlations[c1] += corr_12
    class_correlations = {k: v for k, v in sorted(class_correlations.items(), key=lambda item: item[1])}

    new_seen = list(class_correlations)[:n_seen_classes]
    new_unseen = list(class_correlations)[-n_unseen_classes:]
    return new_seen, new_unseen, attributes




''' Minimal Attribute Split (MAS)
Removes unnecessary (i.e. highly correlated) attributes.
We measure correlation between attributes i and j in a class as the ratio of co-occurrencies of i and j over i or j. Notice that this is not symmetric.



TODO NO! NO! NO! THE POINT WAS TO REMOVE ATTRIBUTES, NOT TO DEFINE NEW SPLITS!

MAS E MCS MODIFICATI

'''
def minimal_attribute_split(features, labels, attributes, inverse):
  correlations = []
  for a1 in attributes.T:
    att_correlations = []
    for a2 in attributes.T:
      d = np.correlate(a1, a2)
      att_correlations.append(d)
    sum_att_correlations = np.sum(att_correlations)
    correlations.append(sum_att_correlations)
  sorted_correlations = np.argsort(correlations)
  #get rid of the n most correlated attributes
  new_att = attributes
  #flo : n = 100
  #awa  : n = 10
  #cub  : n = 40
  indexes = sorted_correlations[-100:]
  indexes = sorted(indexes,reverse=True)
  for idx in indexes:
    new_att = np.delete(new_att,idx,1)

  new_seen = sorted_correlations[:n_seen_classes] if inverse else sorted_correlations[n_unseen_classes:]
  new_unseen = sorted_correlations[n_seen_classes:] if inverse else sorted_correlations[:n_unseen_classes]
  return new_seen, new_unseen, attributes


# TODO minimal correlation split: generate a series of random splits until you get one with correlation < K
def minimal_correlation_split(features, labels, attributes, inverse):
	att_correlations = 100000
  #awa1 : 910
  #cub : 11300
  #flo : 635
	while att_correlations > 635:
		old_seen = torch.from_numpy(np.unique(seen_labels))
		old_unseen = torch.from_numpy(np.unique(unseen_labels))
		old_classes = np.concatenate((old_seen, old_unseen))
		np.random.shuffle(old_classes)
		new_seen = old_classes[:int(n_seen_classes)]
		new_unseen = old_classes[int(n_seen_classes):]
		# TODO correlations
		seen_attributes = attributes[new_seen]
		att_correlations = 0
		for a1 in seen_attributes.T:
			corr = []
			for a2 in seen_attributes.T:
				d = np.correlate(a1, a2)
				corr.append(d)
			sum_corr = np.sum(corr)
			att_correlations += sum_corr
	return new_seen, new_unseen, attributes



def pca_attribute_split(features, labels, attributes, inverse):

	# print(attributes)
	# print(attributes.shape)
	# print('----------------------------')

	new_attributes = attributes
	from sklearn.decomposition import PCA
	pca = PCA(n_components=PCA_COMPONENTS)
	new_attributes = pca.fit_transform(new_attributes)
  
	# print(new_attributes)
	# print(new_attributes.shape)
	# exit(0)


	old_seen = torch.from_numpy(np.unique(seen_labels))
	old_unseen = torch.from_numpy(np.unique(unseen_labels))
	return old_seen, old_unseen, new_attributes


#Linear Discriminative Analysis

def lda_attribute_split(features, labels, attributes, inverse):

  new_attributes = attributes
  matcontent = sio.loadmat(path + "/res101.mat")
  from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
  lda = LinearDiscriminantAnalysis(n_components=LDA_COMPONENTS)
  new_attributes = lda.fit_transform(new_attributes,)

  old_seen = torch.from_numpy(np.unique(seen_labels))
  old_unseen = torch.from_numpy(np.unique(unseen_labels))
  return old_seen, old_unseen, new_attributes



def class_cardinality_split(features, labels, attributes, inverse):
  old_seen = torch.from_numpy(np.unique(seen_labels))
  old_unseen = torch.from_numpy(np.unique(unseen_labels))
  old_classes = np.concatenate((old_seen, old_unseen))
  new_seen = old_classes[:40]
  new_unseen = old_classes[40:]
  return new_seen, new_unseen, attributes



# TODO unambiguous split: 1 if > 0.8, 0 otherwise


split_types = {
	'rnd': random_split,
	'gcs': greedy_class_split,
	'ccs': clustered_class_split,
	'mas': minimal_attribute_split,
	'mcs': minimal_correlation_split,
	'pca': pca_attribute_split,
  'lda': lda_attribute_split,
  'lccs' : least_correlated_class_split,
  'card' : class_cardinality_split
}


# load dataset
matcontent_res101 = sio.loadmat(PATH + '/res101.mat')
matcontent_att_splits = sio.loadmat(PATH + '/att_splits.mat')

# get data: features, labels, and attributes
features = matcontent_res101['features'].T
labels = matcontent_res101['labels'].astype(int).squeeze() - 1
attributes = matcontent_att_splits['att'].T

# get loc data and splits
test_seen_loc = matcontent_att_splits['test_seen_loc'].squeeze() - 1		# tot 4958 - seen classes (GZSL testing) - tot (test_seen + test_unseen + trainval) 30475
test_unseen_loc = matcontent_att_splits['test_unseen_loc'].squeeze() - 1	# tot 5685 - unseen classes (ZSL/GZSL testing) - tot test (seen + unseen) 10643
trainval_loc = matcontent_att_splits['trainval_loc'].squeeze() - 1			# tot 19832 - (train + val - test_seen)
train_loc = matcontent_att_splits['train_loc'].squeeze() - 1				# ONLY VALIDATION MODE - tot 16864
val_loc = matcontent_att_splits['val_loc'].squeeze() - 1					# ONLY VALIDATION MODE - tot 7926 - ...val_unseen_loc (but includes test_seen, why?)

test_seen_ratio = test_seen_loc.size / (test_seen_loc.size + trainval_loc.size)

trainval_loc = matcontent_att_splits['trainval_loc'].squeeze() - 1
test_unseen_loc = matcontent_att_splits['test_unseen_loc'].squeeze() - 1
seen_labels = torch.from_numpy(labels[trainval_loc]).long().numpy()
unseen_labels = torch.from_numpy(labels[test_unseen_loc]).long().numpy()
n_seen_classes = torch.from_numpy(np.unique(seen_labels)).size(0)
n_unseen_classes = torch.from_numpy(np.unique(unseen_labels)).size(0)

# print data info
print('Features: ' + str(features.shape))
print('Labels: ' + str(labels.shape))
print('Attributes: ' + str(attributes.shape))
print('Seen classes: ' + str(n_seen_classes))
print('Unseen classes: ' + str(n_unseen_classes))


# new_seen, new_unseen = greedy_class_split(features, labels, attributes)
# new_seen, new_unseen = inv_greedy_class_split(features, labels, attributes)
new_seen, new_unseen, new_attributes = split_types[SPLIT](features, labels, attributes, INV)


matcontent_att_splits_new = matcontent_att_splits.copy()

# seen_loc = np.concatenate((test_seen_loc, trainval_loc))			# without splits
# np.random.shuffle(seen_loc)
# test_seen_loc = seen_loc[:int(test_seen_ratio * seen_loc.size)]
# trainval_loc = seen_loc[int(test_seen_ratio * seen_loc.size):]
# matcontent_att_splits_new['test_seen_loc'] = test_seen_loc + 1
# matcontent_att_splits_new['trainval_loc'] = trainval_loc + 1

# print(trainval_loc)
# print(test_seen_loc)
# trainval_loc = np.sort(trainval_loc)
# test_seen_loc = np.sort(test_seen_loc)
# print(trainval_loc)
# print(test_seen_loc)

# get new seen_loc and unseen_loc from new splits
seen_loc = np.where(np.in1d(labels, new_seen))[0]
test_unseen_loc = np.where(np.in1d(labels, new_unseen))[0]

# TODO found the problem: you have to randomize here because labels are ordered
np.random.shuffle(seen_loc)
test_seen_loc = seen_loc[:int(test_seen_ratio * seen_loc.size)]
trainval_loc = seen_loc[int(test_seen_ratio * seen_loc.size):]
matcontent_att_splits_new['test_seen_loc'] = test_seen_loc + 1
matcontent_att_splits_new['test_unseen_loc'] = test_unseen_loc + 1
matcontent_att_splits_new['trainval_loc'] = trainval_loc + 1


# TODO attributes
print(matcontent_att_splits_new['att'])
print(matcontent_att_splits_new['att'].shape)
matcontent_att_splits_new['att'] = new_attributes.T
print('------------------------------------')
print(matcontent_att_splits_new['att'])
print(matcontent_att_splits_new['att'].shape)



if SAVE:
  if SPLIT == 'card':
    SPLIT+= 'card4062'
  if SPLIT == 'pca':
    SPLIT += str(PCA_COMPONENTS)

  if SPLIT == 'lda':
    SPLIT += str(LDA_COMPONENTS)
  sio.savemat(PATH + '/att_splits_' + SPLIT + '8.mat', matcontent_att_splits_new)
  print('Saved')


# print(seen_loc.size)
# print(trainval_loc.size)
# print(test_seen_loc.size)
# print(test_unseen_loc.size)
# print()
# print(np.sort(matcontent_att_splits['trainval_loc'].squeeze()))
# print(np.sort(matcontent_att_splits['test_unseen_loc'].squeeze()))
# print(np.sort(matcontent_att_splits['test_seen_loc'].squeeze()))
# print()
# print(np.sort(matcontent_att_splits_new['trainval_loc']))
# print(np.sort(matcontent_att_splits_new['test_unseen_loc']))
# print(np.sort(matcontent_att_splits_new['test_seen_loc']))

# _train_feature = scaler.fit_transform(feature[trainval_loc])
# _test_seen_feature = scaler.transform(feature[test_seen_loc])
# _test_unseen_feature = scaler.transform(feature[test_unseen_loc])
# self.train_label = torch.from_numpy(label[trainval_loc]).long()
# self.test_unseen_feature = torch.from_numpy(_test_unseen_feature).float()
# self.test_unseen_label = torch.from_numpy(label[test_unseen_loc]).long()
# self.test_seen_feature = torch.from_numpy(_test_seen_feature).float()
# self.test_seen_label = torch.from_numpy(label[test_seen_loc]).long()

# self.seenclasses = torch.from_numpy(np.unique(self.train_label.numpy()))
# self.unseenclasses = torch.from_numpy(np.unique(self.test_unseen_label.numpy()))
# self.ntrain = self.train_feature.size()[0]
# self.ntrain_class = self.seenclasses.size(0)
# self.ntest_class = self.unseenclasses.size(0)
# self.train_class = self.seenclasses.clone()
# self.allclasses = torch.arange(0, self.ntrain_class+self.ntest_class).long()
# self.train_mapped_label = map_label(self.train_label, self.seenclasses)
