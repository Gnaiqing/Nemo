import numpy as np
import scipy
from scipy import sparse
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler
from tqdm import tqdm, trange
from multiprocessing import Pool
from lf_utils import SentimentLF, compute_gt_precision
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_selection import VarianceThreshold
import yake
from lf_utils import SentimentLF
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
        self.null_lf_idx = self.X_pos.shape[1] + self.X_neg.shape[1]  # LF idx for returning null LF
    
    
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

        lf_probs = np.hstack((X_lambda_pos_prob*0.5, X_lambda_neg_prob*0.5, np.zeros((len(X_lambda_pos_prob),1),dtype=float)))
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

        lf_scores = np.hstack((pos_lf_scores, neg_lf_scores, np.zeros((len(pos_lf_scores),1),dtype=float)))
        return lf_scores


    def get_lf_idx(self, lf=None):
        if lf is not None:
            keyword = lf.keyword
            label = lf.label
            if label == 1 and keyword in self.vectorizer_pos.vocabulary_:
                # positive LF
                idx = self.vectorizer_pos.vocabulary_[keyword]
            elif label == -1 and keyword in self.vectorizer_neg.vocabulary_:
                # negative LF
                idx = self.vectorizer_neg.vocabulary_[keyword] + self.X_pos.shape[1]
            else:
                # null LF or LF unmodeled due to low coverage
                idx = self.null_lf_idx
        else:
            idx = self.null_lf_idx
        return idx

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

        lf_scores = np.hstack((pos_lf_scores, neg_lf_scores, np.zeros((len(pos_lf_scores), 1), dtype=float)))
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

        lf_scores = np.hstack((pos_lf_scores, neg_lf_scores, np.zeros(1, dtype=float)))
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

    def evaluate_model_acc(self, query_indices, lf_indices, ys_pred=None):
        pred_lf_probs = self.compute_lf_prob(ys_pred=ys_pred)
        eps = 1e-6
        pred_lf_probs_eps = (pred_lf_probs + eps) / (1.0 + eps * pred_lf_probs.shape[1]) # used for computing NLL
        labels = np.array(lf_indices)
        nlls = -np.log(pred_lf_probs[np.arange(len(labels)), labels])
        pred_lf_probs = pred_lf_probs[query_indices,:]  # get the predicted LF probs on queried indices
        pred_lf = np.argmax(pred_lf_probs, axis=1)
        non_abstain = labels != self.null_lf_idx
        accuracy = np.mean(pred_lf == labels)
        nll = np.mean(nlls)
        non_abstain_accuracy = np.mean(pred_lf[non_abstain] == labels[non_abstain])
        non_abstain_nll = np.mean(nlls[non_abstain])
        abstain_accuracy = np.mean(pred_lf[~non_abstain] == labels[~non_abstain])
        abstain_nll = np.mean(nlls[~non_abstain])
        print(f"Evaluation dataset size: {len(lf_indices)}.")
        print(f"Num of instance that user return LF: {len(labels[non_abstain])}")
        print(f"Accuracy of User Model: {100. * accuracy:.2f}")
        print(f"Accuracy of User Model when user return LF: {100. * non_abstain_accuracy:.2f}")
        print(f"Accuracy of User Model when user do not return LF: {100. * abstain_accuracy:.2f}")

        print(f"NLL of User Model: {nll:.3f}")
        print(f"NLL of User Model when user return LF: {non_abstain_nll:.3f}")
        print(f"NLL of User Model when user do not return LF: {abstain_nll:.3f}")


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

    def evaluate_lf_model_acc(self, lf_agent, n_valid, ys_pred=None):
        # Evaluate the accuracy of user model by randomly sample data point and predict the LF returned
        candidate_idxs = np.array(sorted(self.candidate_idxs))
        simulate_query_idxs = self.rand_state.choice(candidate_idxs, size=n_valid, replace=False)
        lf_indices = []
        for idx in simulate_query_idxs:
            lf = lf_agent.create_lf([idx])
            lf_idx = self.lf_model.get_lf_idx(lf)
            lf_indices.append(lf_idx)

        self.lf_model.evaluate_model_acc(simulate_query_idxs, lf_indices, ys_pred=ys_pred)


class LFEmbedder:
    def __init__(self):
        self.embedding_dim = 6

    def get_single_lf_embedding(self, lambda_pos, lambda_neg, k, c, ys_pred=None):
        # get the embedding of LF that predict class c based on feature k
        if c==1:
            wl = lambda_pos[:,k]
        elif c==-1:
            wl = -lambda_neg[:,k]
        else:
            # model the case that user do not return a LF (null LF)
            embedding = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0]).astype(float)
            return embedding

        if ys_pred is not None:
            est_acc = np.sum(wl == ys_pred) / np.sum(wl)
        else:
            est_acc = 0.5  # default accuracy value
        cov = np.sum(wl != 0) / len(wl)
        score = cov * (2* est_acc - 1)  # LF score heuristic by IWS
        pos = float(c == 1)
        neg = float(c == -1)
        embedding = np.array([est_acc, cov, score, pos, neg, 0.0]).astype(float)
        return embedding

    def get_lf_embedding(self,lambda_pos, lambda_neg, ys_pred=None):
        # calculate the embedding of all candidate LFs. return a numpy array of (n_LF * n_dim)
        # first n_pos_dim are embeddings of LF return +1. next n_neg_dim are embeddings of LF return -1. Last dim
        # is embedding of null LF.
        pos_dim = np.shape(lambda_pos)[1]
        neg_dim = np.shape(lambda_neg)[1]
        if ys_pred is not None:
            ys_pred = ys_pred[:,1].reshape(-1,1).astype(bool)
            a = (lambda_pos & ys_pred).sum(axis=0).astype(float)
            b = lambda_pos.sum(axis=0).astype(float)
            pos_est_acc = np.divide(a, b, out=np.zeros_like(a), where=b!=0)
            a = (lambda_neg & ~ys_pred).sum(axis=0).astype(float)
            b = lambda_neg.sum(axis=0).astype(float)
            neg_est_acc = np.divide(a, b, out=np.zeros_like(a), where=b!=0)
        else:
            pos_est_acc = np.ones(pos_dim) * 0.5
            neg_est_acc = np.ones(neg_dim) * 0.5

        pos_cov = np.sum(lambda_pos, axis=0) / len(lambda_pos)
        pos_score = pos_cov * (2*pos_est_acc - 1)
        pos_embedding = np.vstack((pos_est_acc, pos_cov, pos_score, np.ones(pos_dim), np.zeros(pos_dim), np.zeros(pos_dim))).astype(float)
        neg_cov = np.sum(lambda_neg, axis=0) / len(lambda_neg)
        neg_score = neg_cov * (2*neg_est_acc - 1)
        neg_embedding = np.vstack((neg_est_acc, neg_cov, neg_score, np.zeros(neg_dim), np.ones(neg_dim), np.zeros(neg_dim))).astype(float)
        null_embedding = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0]).reshape(1,-1)
        embedding_matrix = np.vstack((pos_embedding.T, neg_embedding.T, null_embedding))
        return embedding_matrix

    def get_sentence_embedding(self, embedding_matrix, lambda_pos, lambda_neg):
        """
        Get the embedding of each sentence by masking the non-exist features in sentences as 0
        return: np.array with size (n_sentence, n_LF * n_feature)
        """
        embed_dim = embedding_matrix.shape[1]
        n = lambda_pos.shape[0]
        lambda_pos = np.repeat(lambda_pos, embed_dim, axis=1)
        lambda_neg = np.repeat(lambda_neg, embed_dim, axis=1)
        lambda_mat = np.hstack((lambda_pos, lambda_neg, np.ones((n, embed_dim)))).astype(float)
        LF_embedding = embedding_matrix.flatten()
        sentence_embeddings = lambda_mat * LF_embedding
        return sentence_embeddings

class UserModel(torch.nn.Module):
    def __init__(self, embedding_dim, output_dim):
        super(UserModel, self).__init__()
        self.embedding_dim = embedding_dim
        self.output_dim = output_dim
        self.fc1 = torch.nn.Linear(embedding_dim, 1)

    def forward(self, x):
        x = torch.reshape(x, (-1, self.output_dim, self.embedding_dim))  # (n_data * n_LF * n_feat)
        x = self.fc1(x).squeeze()
        return x

class UserModelTrainer:
    def __init__(self, output_dim):
        self.queried_indices = []
        self.lf_indices = []
        self.lf_embedder = LFEmbedder()
        self.output_dim = output_dim
        self.model = UserModel(self.lf_embedder.embedding_dim, self.output_dim)

    def add(self, query_idx, lf_idx):
        self.queried_indices.append(query_idx)
        self.lf_indices.append(lf_idx)

    def train_model(self, lambda_pos, lambda_neg, ys_pred=None):
        lf_embedding = self.lf_embedder.get_lf_embedding(lambda_pos, lambda_neg, ys_pred=ys_pred)
        sentence_embedding = self.lf_embedder.get_sentence_embedding(lf_embedding, lambda_pos, lambda_neg)
        # build training dataset
        X = torch.tensor(sentence_embedding[np.array(self.queried_indices),:], dtype=torch.float)
        y = torch.tensor(self.lf_indices,dtype=torch.long)
        label_unique, counts = np.unique(self.lf_indices, return_counts=True)
        class_weights = {}
        for i,c in enumerate(label_unique):
            class_weights[c] = sum(counts) / counts[i]

        example_weights = [class_weights[c] for c in self.lf_indices]
        sampler = WeightedRandomSampler(example_weights,len(self.lf_indices))
        trainset = TensorDataset(X, y)
        train_dataloader = DataLoader(trainset, batch_size=32, sampler=sampler)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)
        criterion = torch.nn.CrossEntropyLoss()

        # Train the model
        pbar = trange(1000, desc="Training", unit="epoch")
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

            accuracy = n_correct / len(trainset)
            pbar.set_postfix(accuracy=f"{100.*accuracy:.2f}", loss=f"{running_loss/len(train_dataloader):.3f}")


        print("Training set size:", len(y))


class LearnedLFModel(LFModel):
    def __init__(self, xs, sentiment_lexicon, pn, kw, dict_size):
        super(LearnedLFModel, self).__init__(xs, sentiment_lexicon, pn, kw, dict_size)
        output_dim = self.X_pos.shape[1] + self.X_neg.shape[1] + 1
        self.trainer = UserModelTrainer(output_dim)
        self.lambda_pos_origin = (self.X_pos != 0)  # record the existence of positive feature
        self.lambda_neg_origin = (self.X_neg != 0)  # record the existence of negative feature

    def compute_lf_prob(self, ys_pred=None):
        xs_embedding = torch.Tensor(self.get_xs_embedding(ys_pred=ys_pred))  # embed training data
        lf_prob = self.trainer.model(xs_embedding).detach().cpu().numpy()
        lf_prob = scipy.special.softmax(lf_prob, axis=1)
        return lf_prob

    def get_xs_embedding(self, ys_pred=None):
        """
        Get embedding of training dataset
        """
        lambda_pos = (self.X_pos != 0)
        lambda_neg = (self.X_neg != 0)
        lf_embedding = self.trainer.lf_embedder.get_lf_embedding(lambda_pos, lambda_neg, ys_pred=ys_pred)
        xs_embedding = self.trainer.lf_embedder.get_sentence_embedding(lf_embedding, lambda_pos, lambda_neg)
        return xs_embedding

    def update_user_model(self, ys_pred=None):
        self.trainer.train_model(self.lambda_pos_origin, self.lambda_neg_origin, ys_pred=ys_pred)

    def record(self, query_idxs, lf_idxs):
        # record interactions
        for (query_idx, lf_idx) in zip(query_idxs, lf_idxs):
            self.trainer.add(query_idx, lf_idx)


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








