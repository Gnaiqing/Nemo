import numpy as np
import scipy
from scipy import sparse
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler
from tqdm import tqdm, trange
from multiprocessing import Pool
from lf_utils import SentimentLF, compute_gt_precision
from label_models import get_label_model
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import accuracy_score, precision_score, recall_score
import yake
from lf_utils import SentimentLF
from utils import filter_abtain
# import spacy
import pdb


class Lexicon(object):
    def __init__(self, keyword_dict):
        self.keyword_dict = keyword_dict

    def get_keywords(self, x):
        keywords = list()
        for token in x:
            if token in self.keyword_dict:
                keywords.append(token)
        return keywords


def build_keyword_dict(xs, dict_size):
    print('buliding keyword list...')
    corpus = [' '.join(doc) for doc in xs]
    # vectorizer = CountVectorizer(max_df=0.1, min_df=20/len(corpus), stop_words='english', max_features=dict_size)
    vectorizer = CountVectorizer(min_df=20, stop_words='english')
    # min_df = int(0.01 * len(corpus))
    # vectorizer = CountVectorizer(min_df=min_df, stop_words='english')
    X = vectorizer.fit_transform(corpus).toarray()
    keyword_dict = vectorizer.vocabulary_
    print(f'Lexicon Size: {len(keyword_dict)}')

    return keyword_dict


def cross_entropy(ps, qs):
    ce = -(ps * np.log(np.clip(qs, 1e-8, 1-1e-8)) + (1-ps) * np.log(np.clip(1-qs, 1e-8, 1-1e-8)))
    ce = ce.sum()
    return ce


class LFModel:
    def __init__(self, xs, sentiment_lexicon, pn, kw, dict_size):
        self.xs = xs
        self.lexicon = sentiment_lexicon
        self.pn = pn
        self.kw = kw
        self.dict_size = dict_size
        self.X_pos, self.X_neg, self.vectorizer_pos, self.vectorizer_neg = self.build_X(pn, kw, dict_size) # build the example-pos_keyword matrix

    
    def update(self, keyword):
        # remove keyword from X_pos or X_neg
        if keyword in self.vectorizer_pos.vocabulary_:
            idx = self.vectorizer_pos.vocabulary_[keyword]
            self.X_pos[:, idx] = 0.

        if keyword in self.vectorizer_neg.vocabulary_:
            idx = self.vectorizer_neg.vocabulary_[keyword]
            self.X_neg[:, idx] = 0.


    def update_none(self, keywords):
        for keyword in keywords:
            self.update(keyword)


    def compute_lf_prob(self, ys_pred=None):
        X_pos = np.copy(self.X_pos) # example_pos_keyword frequency matrix
        X_neg = np.copy(self.X_neg) # example_neg_keyword frequency matrix

        #### uniform
        # norm = X_pos.sum(axis=1)
        # norm[norm!=0] = 1. / norm[norm!=0]
        # X_lambda_pos_prob = (X_pos.T * norm).T

        # norm = X_neg.sum(axis=1)
        # norm[norm!=0] = 1. / norm[norm!=0]
        # X_lambda_neg_prob = (X_neg.T * norm).T


        #### precision weighted
        # t = 0.01
        t = 1.
        p_neg = ys_pred[:, 0].mean(axis=0)
        p_pos = ys_pred[:, 1].mean(axis=0)

        X_pos = (X_pos != 0).astype(float)
        X_neg = (X_neg != 0).astype(float)

        precision_pos = (X_pos.T * ys_pred[:, 1]).T.sum(axis=0)
        norm = X_pos.sum(axis=0)
        precision_pos[norm!=0] /= norm[norm!=0]

        precision_neg = (X_neg.T * ys_pred[:, 0]).T.sum(axis=0)
        norm = X_neg.sum(axis=0)
        precision_neg[norm!=0] /= norm[norm!=0]

        X_lambda_pos_prob = np.exp(precision_pos / t) * X_pos
        norm = X_lambda_pos_prob.sum(axis=1)
        norm[norm!=0] = 1. / norm[norm!=0]
        X_lambda_pos_prob = (X_lambda_pos_prob.T * norm).T

        X_lambda_neg_prob = np.exp(precision_neg / t) * X_neg
        norm = X_lambda_neg_prob.sum(axis=1)
        norm[norm!=0] = 1. / norm[norm!=0]
        X_lambda_neg_prob = (X_lambda_neg_prob.T * norm).T

        lf_probs = np.hstack((X_lambda_pos_prob*0.5, X_lambda_neg_prob*0.5))
        return lf_probs


    def compute_lf_score_cluster(self, label_matrix, disc_model, xs_feature):
        X_pos = np.copy(self.X_pos)
        X_neg = np.copy(self.X_neg)
        X_pos = (X_pos!=0).astype(int)
        X_neg = (X_neg!=0).astype(int)

        pos_lf_centroids = xs_feature.T.dot(X_pos)
        num_coverage = X_pos.sum(axis=0)
        non_abstain = num_coverage != 0
        pos_lf_centroids[:, non_abstain] /= num_coverage[non_abstain]
        pos_lf_centroids = pos_lf_centroids.T
        
        neg_lf_centroids = xs_feature.T.dot(X_neg)
        num_coverage = X_neg.sum(axis=0)
        non_abstain = num_coverage != 0
        neg_lf_centroids[:, non_abstain] /= num_coverage[non_abstain] 
        neg_lf_centroids = neg_lf_centroids.T

        pred_pos = disc_model.predict_proba(pos_lf_centroids)
        pos_lf_scores = -(pred_pos * np.log(np.clip(pred_pos, 1e-8, 1-1e-8))).sum(axis=1)

        pred_neg = disc_model.predict_proba(neg_lf_centroids)
        neg_lf_scores = -(pred_neg * np.log(np.clip(pred_neg, 1e-8, 1-1e-8))).sum(axis=1)

        lf_scores = np.hstack((pos_lf_scores, neg_lf_scores))
        return lf_scores

    def compute_lf_score_moc(self, label_matrix, label_model, disc_model):
        X_pos = np.copy(self.X_pos)
        X_neg = np.copy(self.X_neg)

        num_neg = np.sum(label_matrix == -1, axis=1)
        num_pos = np.sum(label_matrix == 1, axis=1)
        
        # current mv prediction based on "positive label"
        num_nonabstain = num_pos + num_neg
        cur_pred = np.full(label_matrix.shape[0], 0.5)
        cur_pred[num_nonabstain!=0] = num_pos[num_nonabstain!=0] / (num_pos + num_neg)[num_nonabstain!=0]

        # prepare possible updated states
        pos = (num_pos + 1) / (num_pos + num_neg + 1)
        neg = (num_pos) / (num_pos + num_neg + 1)
        non_abstain = (num_pos + num_neg) != 0
        abstain = np.zeros_like(pos).astype(int)
        abstain[non_abstain] = (num_pos)[non_abstain] / (num_pos + num_neg)[non_abstain]

        # updated predictions for pos lfs
        choices = np.hstack([abstain.reshape(-1, 1), pos.reshape(-1, 1)])
        X_pos = X_pos.astype(int)
        X_pos[X_pos!=0] = 1
        new_pred_pos_lfs = np.take_along_axis(choices, X_pos, axis=1)

        # updated prediction for neg lfs
        choices = np.hstack([abstain.reshape(-1, 1), neg.reshape(-1, 1)])
        X_neg = X_neg.astype(int)
        X_neg[X_neg!=0] = 1
        new_pred_neg_lfs = np.take_along_axis(choices, X_neg, axis=1)


        # calculate score for pos lfs
        pos_lf_scores = np.array([cross_entropy(cur_pred, new_pred_pos_lfs[:, j]) for j in range(new_pred_pos_lfs.shape[1])])
        # pos_lf_scores = np.abs(new_pred_pos_lfs.T - cur_pred).sum(axis=1)

        # calculate score for neg lfs
        neg_lf_scores = np.array([cross_entropy(cur_pred, new_pred_neg_lfs[:, j]) for j in range(new_pred_neg_lfs.shape[1])])
        # neg_lf_scores = np.abs(new_pred_neg_lfs.T - cur_pred).sum(axis=1)

        lf_scores = np.hstack((pos_lf_scores, neg_lf_scores))
        return lf_scores


    def compute_lf_score(self, method, xs_score, ys_pred=None):
        X_pos = np.copy(self.X_pos)
        X_neg = np.copy(self.X_neg)

        lambda_pos = (X_pos!=0).astype(float) # shape(num_xs, num_pos_keywords), each column corresponds to the coverage of an positive LF
        lambda_neg = (X_neg!=0).astype(float) # shape(num_xs, num_neg_keywords), each column corresponds to the coverage of an negative LF

        if method == 'sum':
            pos_lf_scores = (lambda_pos.T * xs_score).T.sum(axis=0)
            neg_lf_scores = (lambda_neg.T * xs_score).T.sum(axis=0)
        elif method == 'mean':
            pos_lf_scores = (lambda_pos.T * xs_score).T.sum(axis=0)
            neg_lf_scores = (lambda_neg.T * xs_score).T.sum(axis=0)
            pos_lf_scores[pos_lf_scores!=0] /= lambda_pos.sum(axis=0)[pos_lf_scores!=0]
            neg_lf_scores[neg_lf_scores!=0] /= lambda_neg.sum(axis=0)[neg_lf_scores!=0]
        elif method == 'mean-shift':
            xs_score = xs_score - np.mean(xs_score)
            pos_lf_scores = (lambda_pos.T * xs_score).T.sum(axis=0)
            neg_lf_scores = (lambda_neg.T * xs_score).T.sum(axis=0)
        elif method == 'weighted':
            pos_lf_scores = ((lambda_pos.T * xs_score) *  (2 * ys_pred[:, 1] - 1)).T.sum(axis=0)
            neg_lf_scores = ((lambda_neg.T * xs_score) *  (2 * ys_pred[:, 0] - 1)).T.sum(axis=0)
        elif method == 'weighted-mean':
            pos_lf_scores = ((lambda_pos.T * xs_score) *  (2 * ys_pred[:, 1] - 1)).T.sum(axis=0)
            neg_lf_scores = ((lambda_neg.T * xs_score) *  (2 * ys_pred[:, 0] - 1)).T.sum(axis=0)
            pos_lf_scores[pos_lf_scores!=0] /= lambda_pos.sum(axis=0)[pos_lf_scores!=0]
            neg_lf_scores[neg_lf_scores!=0] /= lambda_neg.sum(axis=0)[neg_lf_scores!=0]
        elif method == 'uncertainty':
            raise NotImplementedError
        else:
            raise NotImplementedError

        lf_scores = np.hstack((pos_lf_scores, neg_lf_scores))
        return lf_scores

    def build_X(self, pn=False, kw=False, dict_size=500):
        """ Build the example-keyword matrix and keyword dictionary.
        One for positive keywords, one for negative keywords.
        """
        xs = self.xs

        if not kw:
            # extract generic keywords without external corpus
            keyword_dict = build_keyword_dict(xs, dict_size=dict_size)
            lexicon = Lexicon(keyword_dict)
            self.lexicon = lexicon

        xs_pos_keywords = list()
        xs_neg_keywords = list()

        if not pn:
            for x in tqdm(xs):
                if not kw:
                    keywords = self.lexicon.get_keywords(x)
                else:
                    pos_keywords = self.lexicon.tokens_with_sentiment(x, 1)
                    neg_keywords = self.lexicon.tokens_with_sentiment(x, -1)
                    keywords = list(pos_keywords) + list(neg_keywords)
                xs_pos_keywords.append(keywords) # Note that there might be repetitive tokens
                xs_neg_keywords.append(keywords) # Note that there might be repetitive tokens
        else:
            if not kw:
                raise ValueError('No externel keyword set provided')
            for x in tqdm(xs):
                xs_pos_keywords.append(self.lexicon.tokens_with_sentiment(x, 1)) # Note that there might be repetitive tokens
                xs_neg_keywords.append(self.lexicon.tokens_with_sentiment(x, -1)) # Note that there might be repetitive tokens

        # build dictionary for positive keywords
        vectorizer_pos = CountVectorizer(preprocessor=lambda x:x, tokenizer=lambda x:x)
        X_pos = vectorizer_pos.fit_transform(xs_pos_keywords).toarray().astype(float)
        
        # build dictionary for negative keywords
        vectorizer_neg = CountVectorizer(preprocessor=lambda x:x, tokenizer=lambda x:x)
        X_neg = vectorizer_neg.fit_transform(xs_neg_keywords).toarray().astype(float)

        return X_pos, X_neg, vectorizer_pos, vectorizer_neg


class LearnedLFModel(LFModel):
    def __init__(self, xs, sentiment_lexicon, pn, kw, dict_size, embedding_method, rand_state, lf_score):
        super(LearnedLFModel, self).__init__(xs, sentiment_lexicon, pn, kw, dict_size)
        self.user_model = UserModel(embedding_method, self.X_pos != 0, self.lexicon.keyword_dict, rand_state)
        self.lambda_matrix = self.X_pos != 0
        self.n_class = 2
        self.keyword_space_size = len(self.lexicon.keyword_dict)
        self.lf_space_size = len(self.lexicon.keyword_dict) * self.n_class
        self.id_to_kw = {}
        self.lf_score = lf_score
        for key in self.lexicon.keyword_dict:
            self.id_to_kw[self.lexicon.keyword_dict[key]] = key

    def get_lf_idx(self, lf=None, k=None, c=None):
        if lf is not None:
            keyword = lf.keyword
            label = lf.label

            if keyword not in self.vectorizer_pos.vocabulary_:
                raise ValueError(f"Keyword {keyword} not exist in vocabulary")
            if label == 1:
                # positive LF
                idx = self.vectorizer_pos.vocabulary_[keyword]
            else:
                # negative LF
                idx = self.vectorizer_neg.vocabulary_[keyword] + self.X_pos.shape[1]
        else:
            assert k is not None and c is not None
            idx = k + self.keyword_space_size * (c == -1)

        return idx
    def get_lf_from_idx(self, idx):
        keyword = self.id_to_kw[idx % self.keyword_space_size]
        if idx >= self.keyword_space_size:
            c = -1
        else:
            c = 1

        lf = SentimentLF(keyword=keyword, label=c)
        return lf

    def compute_lf_prob(self, ys_pred=None):
        """
        Compute the likelihood that user would consider LF as helpful for all possible LFs.
        """
        lf_probs = []
        for idx in range(self.lf_space_size):
            lf = self.get_lf_from_idx(idx)
            p = self.user_model.predict_proba(lf=lf, ys_pred=ys_pred)
            lf_probs.append(p)

        lambda_matrix = self.X_pos != 0
        lf_probs_mat = np.tile(lambda_matrix, self.n_class)
        lf_probs_mat = lf_probs_mat * np.array(lf_probs)

        return lf_probs_mat

    def compute_lf_score_valid(self, lf_agent):
        """
        Compute the score of LF based on metric on validation set. Assume majority voting is used as label model
        for runtime reason.
        """
        lf_scores = []
        used_lfs = []
        for lf in lf_agent.lfs:
            lf_idx = self.get_lf_idx(lf)
            used_lfs.append(lf_idx)

        L_val = lf_agent.L_val
        L_val_pos = np.sum(L_val == 1, axis=1)
        L_val_neg = np.sum(L_val == -1, axis=1)
        Wl_val = np.hstack((lf_agent.M_val, lf_agent.M_val * -1))  # the weak labels provided by each LF
        pos_count = (Wl_val == 1) + L_val_pos.reshape(-1,1)  # count positive weak label after adding each LF
        neg_count = (Wl_val == -1) + L_val_neg.reshape(-1,1)  # count negative weak label after adding each LF.

        ys_val = lf_agent.ys_val
        # compute the baseline accuracy and coverage
        probs = np.zeros((len(L_val), 2), dtype=float)
        probs[:,0] = L_val_neg >= L_val_pos
        probs[:,1] = L_val_pos >= L_val_neg
        filter_mask_val = L_val_pos + L_val_neg > 0
        probs = probs[filter_mask_val]
        pred = np.array([np.random.choice(np.where(y == np.max(y))[0]) for y in probs])
        pred[pred == 0] = -1
        acc_base = np.mean(pred == ys_val[filter_mask_val])
        cov_base = np.mean(filter_mask_val)

        for lf_idx in np.arange(self.lf_space_size):
            if lf_idx in used_lfs:
                lf_scores.append(0.0)
            else:
                probs = np.zeros((len(L_val), 2), dtype=float)
                probs[:,0] = neg_count[:, lf_idx] >= pos_count[:, lf_idx]
                probs[:,1] = pos_count[:, lf_idx] >= neg_count[:, lf_idx]
                filter_mask_val = pos_count[:, lf_idx] + neg_count[:, lf_idx] > 0
                probs = probs[filter_mask_val]
                pred = np.array([np.random.choice(np.where(y == np.max(y))[0]) for y in probs])
                pred[pred==0] = -1
                acc = np.mean(pred == ys_val[filter_mask_val])
                cov = np.mean(filter_mask_val)
                if self.lf_score == "val_label_acc":
                    lf_scores.append(acc-acc_base)
                else:
                    raise ValueError(f"LF score {self.lf_score} not supported.")

        lf_scores = np.array(lf_scores)
        return lf_scores

    def evaluate_model_acc(self, lf_agent, history, ys_pred=None):
        y_pred = []
        y_true = []

        for c in [-1, 1]:
            for k in range(self.X_pos.shape[1]):
                lf = SentimentLF(keyword=self.id_to_kw[k], label=c)
                p = self.user_model.predict_proba(lf=lf, ys_pred=ys_pred)
                y_pred.append(p > 0.5)
                lf_idx = self.get_lf_idx(lf)
                y_true.append(lf_agent.check_lf(lf_idx))

        acc = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = 2 * precision * recall / (precision + recall)
        print(f"User model accuracy: {acc:.3f}")
        print(f"User model precision: {precision:.3f}")
        print(f"User model recall: {recall:.3f}")
        print(f"User model f1: {f1:.3f}")
        history["user_model_accuracy"].append(acc)
        history["user_model_precision"].append(precision)
        history["user_model_recall"].append(recall)
        history["user_model_f1"].append(f1)


class ScoringFunction:
    def __init__(self, method):
        self.method = method

    def apply(self, xs_feature=None, label_matrix=None, label_model=None, disc_model=None):
        if self.method == 'random':
            scores = np.ones(len(xs_feature))
        elif self.method == 'abstain':
            scores = (label_matrix == 0).sum(axis=1)
        elif self.method == 'disagreement':
            scores = label_matrix.shape[1] - np.abs(label_matrix.sum(axis=1))
        elif self.method == 'uncertainty_lm':
            ys_pred = label_model.predict_proba(label_matrix)
            scores = -(ys_pred * np.log(np.clip(ys_pred, 1e-8, 1-1e-8))).sum(axis=1)
        elif self.method == 'uncertainty_dm':
            ys_pred = disc_model.predict_proba(xs_feature)
            scores = -(ys_pred * np.log(np.clip(ys_pred, 1e-8, 1-1e-8))).sum(axis=1)
        elif self.method == 'uncertainty_mix':
            ys_pred = label_model.predict_proba(label_matrix) # TODO: consider update label model every query
            scores_lm = -(ys_pred * np.log(np.clip(ys_pred, 1e-8, 1-1e-8))).sum(axis=1)
            ys_pred = disc_model.predict_proba(xs_feature)
            scores_dm = -(ys_pred * np.log(np.clip(ys_pred, 1e-8, 1-1e-8))).sum(axis=1)
            scores = scores_lm * scores_dm

        return scores


class QueryAgent:
    def __init__(self, xs_feature, xs_token, query_method, query_size, rand_state, allow_repeat, qei, aggregate, lf_model):
        self.xs_feature = xs_feature
        self.xs_token = xs_token
        self.query_method = query_method
        self.query_size = query_size
        self.rand_state = rand_state
        self.allow_repeat = allow_repeat
        self.qei = qei
        self.aggregate = aggregate
        self.lf_model = lf_model

        self.queried_idxs = list()  # queried idxs by order
        self.candidate_idxs = set(range(len(xs_feature)))  # candidate idxs from which next query would be chosen

        self.scoring_function = ScoringFunction(self.query_method)


    def warm_start(self):
        cur_query_idxs = self.rand_state.choice(sorted(self.candidate_idxs), size=self.query_size, replace=False) 
        self.update_query_model(cur_query_idxs)

        return cur_query_idxs



    def query(self, label_matrix, label_model=None, ys_pred=None, use_ys_pred=False, disc_model=None):
        candidate_idxs = np.array(sorted(self.candidate_idxs))
        if self.qei:
            assert self.lf_model is not None
            xs_score = self.scoring_function.apply(self.xs_feature, label_matrix, label_model, disc_model)
            lf_scores = self.lf_model.compute_lf_score(self.aggregate, xs_score, ys_pred)
            X_lf_probs = self.lf_model.compute_lf_prob(ys_pred)
            xs_expected_score = (X_lf_probs * lf_scores).sum(axis=1)
            # X_lf_pos_prob , X_lf_neg_prob = self.lf_model.compute_lf_prob(ys_pred)
            #
            # if not use_ys_pred or ys_pred is None:
            #     class_p = np.array([0.5, 0.5])
            #
            # xs_pos_expected_score = (X_lf_pos_prob * pos_lf_scores).sum(axis=1)
            # xs_neg_expected_score = (X_lf_neg_prob * neg_lf_scores).sum(axis=1)
            #
            # xs_expected_score = (class_p * np.vstack([xs_neg_expected_score, xs_pos_expected_score]).T).sum(axis=1)

            scores = xs_expected_score[candidate_idxs]
        else:
            # TODO: maybe avoid confusion between local idxs and global idxs    
            xs_feature = self.xs_feature[candidate_idxs] 
            label_matrix = label_matrix[candidate_idxs]
            scores = self.scoring_function.apply(xs_feature, label_matrix, label_model, disc_model)
        cur_query_idxs = list(self.rand_state.choice(np.where(scores == np.max(scores))[0], size=self.query_size, replace=False)) # subset idxs
        cur_query_idxs = candidate_idxs[cur_query_idxs] # global idxs
        self.update_query_model(cur_query_idxs)

        return cur_query_idxs

    def update_query_model(self, cur_query_idxs):
        self.queried_idxs += list(cur_query_idxs)
        if not self.allow_repeat:
            self.candidate_idxs -= set(cur_query_idxs)


class LearnedQueryAgent(QueryAgent):
    def query_(self, lf_agent, ys_pred=None):
        candidate_idxs = np.array(sorted(self.candidate_idxs))
        lf_probs = self.lf_model.compute_lf_prob(ys_pred=ys_pred)
        lf_scores = self.lf_model.compute_lf_score_valid(lf_agent)
        x_scores = np.max(lf_probs * lf_scores, axis=1)
        scores = x_scores[candidate_idxs]
        cur_query_idxs = list(self.rand_state.choice(np.where(scores == np.max(scores))[0], size=self.query_size,
                                                     replace=False))  # subset idxs
        cur_query_idxs = candidate_idxs[cur_query_idxs]  # global idxs
        self.update_query_model(cur_query_idxs)

        return cur_query_idxs

class LFEmbedder:
    def __init__(self, n_class=2):
        self.n_class = n_class

    def get_embedding_dim(self, embedding_method):
        embedding_dim = 3 * self.n_class  # record cov, acc, cov*acc for each possible class
        return embedding_dim

    def get_lf_embedding(self, lambda_matrix, k, c, ys_pred=None):
        """
        Get the embedding of LF that predict class c based on feature k.
        lambda_matrix: activation matrix of size (n_instance, n_feature)
        k: feature idx
        c: class idx
        ys_pred: pseudo labels of training data
        """
        wl = lambda_matrix[:,k] * c  # weak labels
        embedding = []
        if ys_pred is not None:
            est_acc = np.sum(wl == ys_pred) / np.sum(wl != 0)
        else:
            est_acc = 0.5  # default accuracy value
        cov = np.sum(wl != 0) / len(wl)
        if self.n_class == 2:
            classes = [-1, 1]
        else:
            classes = np.arange(self.n_class)
        for y in classes:
            if y == c:
                embedding = embedding + [cov, est_acc, cov*est_acc]
            else:
                embedding = embedding + [0.0, 0.0, 0.0]

        embedding = np.array(embedding, dtype=float)
        return embedding

    def get_kw_embedding(self, lambda_matrix, k, ys_pred=None):
        if self.n_class == 2:
            classes = [-1, 1]
        else:
            classes = np.arange(self.n_class)

        embedding = []
        cov = np.sum(lambda_matrix[:,k] != 0) / len(lambda_matrix)

        for c in classes:
            if ys_pred is not None:
                wl = lambda_matrix[:,k] * c
                est_acc = np.sum(wl == ys_pred) / np.sum(wl)
            else:
                est_acc = 0.5

            embedding = embedding + [cov, est_acc, cov*est_acc]

        embedding = np.array(embedding, dtype=float)
        return embedding


class LinearModel(torch.nn.Module):
    def __init__(self, embedding_dim, output_dim):
        super(LinearModel, self).__init__()
        self.embedding_dim = embedding_dim
        self.output_dim = output_dim
        self.fc1 = torch.nn.Linear(embedding_dim, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        return x

    def predict_proba(self, x):
        with torch.no_grad():
            p = F.softmax(self.forward(x), dim=1)

        p = p.cpu().detach().numpy()
        return p

class MLPModel(torch.nn.Module):
    def __init__(self, embedding_dim, output_dim, hidden_dim=10):
        super(MLPModel, self).__init__()
        self.embedding_dim = embedding_dim
        self.output_dim = output_dim
        self.fc1 = torch.nn.Linear(embedding_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def predict_proba(self, x):
        with torch.no_grad():
            p = F.softmax(self.forward(x), dim=1)

        p = p.cpu().detach().numpy()
        return p


class UserModel:
    def __init__(self, embedding_method, lambda_matrix, keyword_dict, rand_state, max_lf_per_x=1, n_class=2):
        self.embedding_method = embedding_method
        self.lambda_matrix = lambda_matrix
        self.n_class = n_class
        self.keyword_dict = keyword_dict
        self.rand_state = rand_state
        self.max_lf_per_x = max_lf_per_x  # maximum number of LFs returned per instance
        self.lf_embedder = LFEmbedder(n_class=n_class)
        embedding_dim = self.lf_embedder.get_embedding_dim(embedding_method)
        if self.embedding_method == "lf":
            # the model predicts whether the user would treat specific LF as helpful
            output_dim = 2
        elif self.embedding_method == "kw":
            # the model predicts which class the user would return for specific keyword
            output_dim = self.n_class + 1
        else:
            raise ValueError("Embedding method not supported")

        self.model = LinearModel(embedding_dim, output_dim)
        self.queried_indices = []
        self.returned_lfs = []

    def add(self, query_idx, lfs):
        self.queried_indices.append(query_idx)
        self.returned_lfs.append(lfs)

    def create_training_set(self, ys_pred=None):
        train_X = []
        train_y = []
        lf_map = {}
        used_kw = np.zeros(len(self.keyword_dict),dtype=bool)
        for query_idx, lfs in zip(self.queried_indices, self.returned_lfs):
            # add user provided LFs to training data
            for lf in lfs:
                if lf.keyword in self.keyword_dict:
                    k = self.keyword_dict[lf.keyword]
                else:
                    raise ValueError(f"Keyword {lf.keyword} not exist in keyword dict.")

                c = lf.label
                lf_map[k] = c  # record the existing LFs in the map
                used_kw[k] = True
                if self.embedding_method == "lf":
                    lf_embedding = self.lf_embedder.get_lf_embedding(self.lambda_matrix, k, c, ys_pred=ys_pred)
                    train_X.append(lf_embedding)
                    train_y.append(1)  # user regard it as helpful LF
                elif self.embedding_method == "kw":
                    kw_embedding = self.lf_embedder.get_kw_embedding(self.lambda_matrix, k, ys_pred=ys_pred)
                    if self.n_class == 2:
                        c = (c + 1) // 2  # map -1 to 0
                    train_X.append(kw_embedding)
                    train_y.append(c)

            # add negative examples to training data
            if len(lfs) == self.max_lf_per_x:
                if self.embedding_method == "lf":
                    for lf in lfs:
                        k = self.keyword_dict[lf.keyword]
                        if self.n_class == 2:
                            alter_c = - lf.label
                        else:
                            classes = [i for i in range(self.n_class) if i != lf.label]
                            alter_c = self.rand_state.choice(classes)
                        lf_embedding = self.lf_embedder.get_lf_embedding(self.lambda_matrix, k, alter_c, ys_pred=ys_pred)
                        train_X.append(lf_embedding)
                        train_y.append(0)  # user regard it as nonhelpful LF
            else:
                if self.embedding_method == "lf":
                    candidate_k = np.nonzero(self.lambda_matrix[query_idx,:])[0]
                    if len(candidate_k) == 0:
                        continue
                    n_neg = max(len(lfs), 1)
                    for i in range(n_neg):
                        k = np.random.choice(candidate_k)
                        if self.n_class == 2:
                            classes = [-1, 1]
                        else:
                            classes = range(self.n_class)
                        if k in lf_map:
                            classes = [i for i in classes if i != lf_map[k]]
                        alter_c = self.rand_state.choice(classes)
                        lf_embedding = self.lf_embedder.get_lf_embedding(self.lambda_matrix, k, alter_c,
                                                                         ys_pred=ys_pred)
                        train_X.append(lf_embedding)
                        train_y.append(0)

                elif self.embedding_method == "kw":
                    candidate_k_list = np.nonzero(self.lambda_matrix[query_idx,:] & ~used_kw)[0]
                    for k in candidate_k_list:
                        used_kw[k] = True
                        kw_embedding = self.lf_embedder.get_kw_embedding(self.lambda_matrix, k, ys_pred=ys_pred)
                        c = self.n_class
                        train_X.append(kw_embedding)
                        train_y.append(c)

        train_X = np.vstack(train_X)
        train_X = torch.tensor(train_X, dtype=torch.float)
        train_y = torch.tensor(train_y, dtype=torch.long)
        return train_X, train_y

    def train_model(self, ys_pred=None, valid_size=0.2):
        # build training dataset
        train_X, train_y = self.create_training_set(ys_pred=ys_pred)
        train_X, valid_X, train_y, valid_y = train_test_split(train_X, train_y, test_size=valid_size)

        trainset = TensorDataset(train_X, train_y)
        validset = TensorDataset(valid_X, valid_y)
        print("User model training set size:", len(trainset))
        print("User model validation set size: ", len(validset))
        train_dataloader = DataLoader(trainset, batch_size=32, shuffle=True)
        valid_dataloader = DataLoader(validset, batch_size=512, shuffle=False)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)
        criterion = torch.nn.CrossEntropyLoss()

        # Train the model
        best_val_loss = float('inf')
        best_model = None
        pbar = trange(100, desc="Training", unit="epoch")
        for epoch in pbar:
            running_loss = 0.0
            n_correct = 0.0
            for i, data in enumerate(train_dataloader):
                inputs, labels = data
                optimizer.zero_grad()
                outputs = self.model(inputs.float())
                predictions = outputs.argmax(dim=1, keepdim=True).squeeze()
                n_correct += (predictions == labels).sum().item()
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            # Compute validation loss
            val_loss = 0.0
            val_correct = 0.0
            with torch.no_grad():
                for data in valid_dataloader:
                    inputs, labels = data
                    outputs = self.model(inputs.float())
                    predictions = outputs.argmax(dim=1, keepdim=True).squeeze()
                    val_correct += (predictions == labels).sum().item()
                    val_loss += criterion(outputs, labels).item()

            # Update best model if validation loss improved
            val_loss /= len(valid_dataloader)
            accuracy = val_correct / len(validset)
            pbar.set_postfix(train_acc=f"{100. * n_correct / len(trainset):.2f}",
                             val_acc=f"{100. * accuracy:.2f}", loss=f"{running_loss / len(train_dataloader):.3f}",
                             val_loss=f"{val_loss:.3f}")
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model = self.model.state_dict()

        # Load the best model
        self.model.load_state_dict(best_model)

    def predict_proba(self, lf=None, k=None, c=None, ys_pred=None):
        """
        Return the probability that the user would consider lf as helpful
        """
        if k is None or c is None:
            assert lf is not None
            k = self.keyword_dict[lf.keyword]
            c = lf.label

        if self.embedding_method == "lf":
            x = self.lf_embedder.get_lf_embedding(self.lambda_matrix, k, c, ys_pred=ys_pred).reshape((1,-1))
            x = torch.tensor(x, dtype=torch.float)
            p = self.model.predict_proba(x).flatten()[1]

        elif self.embedding_method == "kw":
            y = (lf.label + 1) // 2
            x = self.lf_embedder.get_kw_embedding(self.lambda_matrix, k, ys_pred=ys_pred).reshape((1,-1))
            x = torch.tensor(x, dtype=torch.float)
            p = self.model.predict_proba(x)[y]
        else:
            raise ValueError("Embedding method not supported.")

        return p




class BanditQueryAgent(QueryAgent):
    def __init__(self, xs_feature, xs_token, query_method, query_size, rand_state, allow_repeat, qei, aggregate,
                 lf_model, update_freq=10, epsilon_max=1.0, epsilon_min=0.05, epsilon_decay=0.05):
        super(BanditQueryAgent, self).__init__(xs_feature, xs_token, query_method, query_size, rand_state, allow_repeat,
                                               qei, aggregate, lf_model)
        self.n_queried = 0
        self.update_freq = update_freq
        self.epsilon_max = epsilon_max
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

    def query(self, label_matrix, label_model=None, ys_pred=None, use_ys_pred=False, disc_model=None):
        candidate_idxs = np.array(sorted(self.candidate_idxs))
        epsilon = self.epsilon_min + (self.epsilon_max - self.epsilon_min)* np.exp(-self.epsilon_decay * self.n_queried)
        self.n_queried += self.query_size

        r = self.rand_state.rand()
        if r < epsilon:
            # random sampling
            cur_query_idxs = self.rand_state.choice(candidate_idxs, size=self.query_size, replace=False)
            self.update_query_model(cur_query_idxs)
            return cur_query_idxs
        else:
            assert self.lf_model is not None
            xs_score = self.scoring_function.apply(self.xs_feature, label_matrix, label_model, disc_model)
            lf_scores = self.lf_model.compute_lf_score(self.aggregate, xs_score, ys_pred)
            X_lf_probs = self.lf_model.compute_lf_prob(ys_pred)
            xs_expected_score = (X_lf_probs * lf_scores).sum(axis=1)
            # pos_lf_scores, neg_lf_scores = self.lf_model.compute_lf_score(self.aggregate, xs_score, ys_pred)
            # lf_scores = np.concatenate((pos_lf_scores, neg_lf_scores, (0.0,)))
            # xs_expected_score = np.sum(lf_prob * lf_scores, axis=1)
            scores = xs_expected_score[candidate_idxs]
            cur_query_idxs = list(self.rand_state.choice(np.where(scores == np.max(scores))[0], size=self.query_size,
                                                         replace=False))  # subset idxs
            cur_query_idxs = candidate_idxs[cur_query_idxs]  # global idxs
            self.update_query_model(cur_query_idxs)
            return cur_query_idxs








