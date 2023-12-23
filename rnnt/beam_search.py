import math
import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
import itertools


class BeamSearch(object):
	def __init__(self, model, henc):
		self.model = model
		self.seqmap = {} # seq:BeamNode. Same seqs (different prefixes) share BeamNode

		self.n_forward = 0 # how many times model.forward() is called
		self.maxlen_found = 0 # longest seq found

		# jointnet_fc does not need to be repeated.
		self.henc = henc

	def search(self, beam_width=5, maxlen=None):
		blank = self.model.blank
		nbclasses = self.model.nbclasses
		xlen = self.henc.size(1)
		tt = torch.cuda if self.henc.is_cuda else torch

		if maxlen is None:
			maxlen = xlen # do not generate more than max_len chars

		beam = [((blank,), 1)] # make b[0] tuple so it is hashable
		for t in range(xlen):
			candidates = beam
			beam = []

			# recalculate probs when beam is extended and also same prefix probs are sumed
			key_lambda = lambda b: tuple([x for x in b[0] if x != blank])
			seq_grouped = itertools.groupby(candidates,
											key=key_lambda)
			candidates = []
			for k, g in seq_grouped:
				group = list(g)
				node = self.get_node(group[0][0]) # all sharing same node
				pr = sum([float(node.calc_prob(b[0])) for b in group])
				for b in group:
					candidates.append((b[0], pr))

			while len(candidates) > 0: # Extend beam on U axis until nothing to extend
				candidates_p = numpy.array([x[1] for x in candidates])
				idx = candidates_p.argmax()
				best = candidates[idx]
				pr_best = best[1]
				candidates.remove(best)

				better_than_best = [b for b in beam if b[1] > pr_best]
				if len(better_than_best) >= beam_width:
					break

				node = self.get_node(best[0])
				u = len(node.seq)
				if u >= maxlen:
					continue

				# Probabilities are affected by U. Recalculate
				# [BLANK,1,2] => [BLANK,1,2,BLANK]
				extend_null_pr = pr_best * float(node.out[t, u, blank]) # Pr(y∗) = Pr(y∗) Pr(∅|y, t)
				beam.append((best[0]+(blank,), extend_null_pr)) # Add y∗ to B

				# LABELs + k
				for k in range(nbclasses): # for k ∈ Y do
					if k == blank:
						continue
					pr_k = pr_best * float(node.out[t, u, k]) # Pr(y∗ + k) = Pr(y∗) Pr(k|y∗, t)
					candidates.append((best[0]+(k,), pr_k)) # Add y∗ + k to A

			# Remove all but the W most probable from B
			beam.sort(key=lambda x: float(x[1]), reverse=True)
			beam = beam[:beam_width]

		best = beam[0]
		seq = [x for x in best[0] if x != blank]
		return seq, best[1]

	def get_node(self, prefix):
		key = tuple([x for x in prefix if x != self.model.blank and x != self.model.sos])
		node = self.seqmap.get(key)
		if node is None:
			parent = None
			if len(key) > 0:
				parent = self.seqmap.get(key[:-1])
			node = BeamNode(self.model, self.henc, prefix, parent)
			self.seqmap[key] = node
			self.n_forward += 1
			self.maxlen_found = max(len(key), self.maxlen_found)
		return node


class BeamNode(object):
	def __init__(self, model, henc, seq, parent=None):
		"""
		:param model:
		:param henc:
		:param prefix: including null (and probably ending in null)
		"""
		self.model = model
		self.henc = henc
		self.seq = [s for s in seq if s != 27 and s != 29 and s != 30]
		self.out = None
		# for incremental calculation...
		self.parent = parent
		self.h0 = None
		self.py = None
		self.advance()

	def advance(self):
		parent = self.parent
		if parent:
			c = self.seq[len(parent.seq):]
			seqvar = self.model._create_label_var(c, self.henc)
			y = self.model.embedding(seqvar)
			py, h0 = self.model.prednet(y, parent.h0)
			py = torch.cat([parent.py, py], dim=1)
		else:
			seqvar = self.model._create_label_var([self.model.sos] + self.seq, self.henc)
			y = self.model.embedding(seqvar)
			py, h0 = self.model.prednet(y)

		self.py = py
		self.h0 = h0

		py = py.unsqueeze(dim=1)

		# joint network
		#print(py.size()) # 32, 1, 7, 256
		out = F.tanh(self.model.jointnet_fc(py) + self.henc)
		out = self.model.fc(out)
		out = F.softmax(out, dim=3)
		# 32x15x7x29
		out = out.squeeze(dim=0)
		self.out = out.data.cpu()
		return self.out

	def calc_prob(self, prefix):
		return _calc_prob_from_output(self.out, prefix, 27)

	def calc_prob_all(self):
		# EQ (17)
		T = self.out.size(0)
		U = self.out.size(1)
		blank = self.model.blank
		self.pr = self._alpha(T-1, U-1) * self._k(T-1, U-1, blank)
		return self.pr

	def _k(self, t, u, k):
		if u < 0 or t < 0:
			return 1
		return self.out[t, u, k]

	def _alpha(self, t, u):
		if u <= 0 or t <= 0:
			return 1
		blank = self.model.blank
		k = self.seq[u-1]
		a = self._alpha(t-1, u)*self._k(t-1, u, blank) + self._alpha(t, u-1)*self._k(t, u-1, k)
		return a


def _calc_prob_from_output(output, prefix, blank=0):
	# skip first blank
	prefix = prefix[1:]

	T = output.size(0)
	U = output.size(1)
	assert(len(prefix) <= T+U)

	u = 0
	t = 0
	pr = 1
	for k in prefix:
		pr *= output[t, u, k]
		if k == blank:
			t += 1
		else:
			u += 1
	return pr


def static_search(output, blank=0, beam_width=3):
	"""
	:param output: TxUxH output from model.forward() with first dim removed
	"""
	output = torch.exp(output.data).cpu()
	xlen = output.size(0)
	nbclasses = output.shape[-1]
	n_loop = 0

	beam = [((blank,), 1)] # make b[0] tuple so it is hashable
	for t in range(xlen):
		candidates = beam
		beam = []

		# skipping this just for performance
		# recalculate probs when beam is extended and also same prefix probs are sumed
		# key_lambda = lambda b: tuple([x for x in b[0] if x != blank])
		# seq_grouped = itertools.groupby(candidates,
		# 								key=key_lambda)
		# candidates = []
		# for k, g in seq_grouped:
		# 	group = list(g)
		# 	pr = sum([float(_calc_prob_from_output(output, b[0], blank)) for b in group])
		# 	# if len(group) == 1:
		# 	# 	print(pr, group[0][1])
		# 	# else:
		# 	# 	print('grouped {} items'.format(len(group)))
		# 	for b in group:
		# 		candidates.append((b[0], pr))

		while len(candidates) > 0: # Extend beam on U axis until nothing to extend
			candidates_p = numpy.array([x[1] for x in candidates])
			idx = candidates_p.argmax()
			best = candidates[idx]
			pr_best = best[1]
			candidates.remove(best)

			better_than_best = [b for b in beam if b[1] > pr_best]
			if len(better_than_best) >= beam_width:
				break

			seq = [x for x in best[0] if x != blank]
			u = len(seq)
			if u >= output.size(1):
				continue

			# Probabilities are affected by U. Recalculate
			# [BLANK,1,2] => [BLANK,1,2,BLANK]
			extend_null_pr = pr_best * float(output[t, u, blank]) # Pr(y∗) = Pr(y∗) Pr(∅|y, t)
			beam.append((best[0]+(blank,), extend_null_pr)) # Add y∗ to B

			# LABELs + k
			for k in range(nbclasses): # for k ∈ Y do
				if k == blank:
					continue
				pr_k = pr_best * float(output[t, u, k]) # Pr(y∗ + k) = Pr(y∗) Pr(k|y∗, t)
				candidates.append((best[0]+(k,), pr_k)) # Add y∗ + k to A

			n_loop += 1

		# Remove all but the W most probable from B
		beam.sort(key=lambda x: float(x[1]), reverse=True)
		beam = beam[:beam_width]

	#print('loop: {}'.format(n_loop))

	best = beam[0]
	seq = [x for x in best[0] if x != blank]
	return seq, best[1]