import sys
import os
import argparse
import json
import numpy as np
from collections import defaultdict

from data_utils import load_data, SentimentLexicon
from lf_utils import LFAgent
from query_utils import QueryAgent, LFModel, LearnedLFModel, LearnedQueryAgent, UserModel, LFEmbedder
from discriminator import get_discriminator
from label_models import *
from utils import * 
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from implyloss import ImplyLoss
from kliep import DensityRatioEstimator
import wandb
import pandas as pd
import pdb


def run_fs(args, train_dataset, valid_dataset, warmup_dataset, test_dataset):
    print('Running fully-supervised baseline...')

    disc_model = get_discriminator(model_type=args.model_type)

    xs_tr = np.vstack([train_dataset.xs_feature, warmup_dataset.xs_feature])
    ys_tr = np.hstack([train_dataset.ys, warmup_dataset.ys])

    disc_model.tune_params(xs_tr, ys_tr, valid_dataset.xs_feature, valid_dataset.ys)
    disc_model.fit(xs_tr, ys_tr)

    ys_pred = disc_model.predict(test_dataset.xs_feature)

    acc_test = accuracy_score(test_dataset.ys, ys_pred)
    f1_test = f1_score(test_dataset.ys, ys_pred)
    auc_test = roc_auc_score(test_dataset.ys, disc_model.predict_proba(test_dataset.xs_feature)[:, 1])

    print('Fully-Supervised Acc: {}'.format(acc_test))
    print('Fully-Supervised AUC: {}'.format(auc_test))
    print('Fully-Supervised F1: {}'.format(f1_test))


def run_vs(args, valid_dataset, test_dataset):
    xs = valid_dataset.xs_feature
    ys = valid_dataset.ys
    if args.model_type == 'torch':
        raise NotImplementedError
    else:
        params = {
            'solver': ['liblinear'],
            'max_iter': [1000],
            'C': [0.001, 0.01, 0.1, 1, 10, 100]
        }
        model = GridSearchCV(LogisticRegression(random_state=rand_state), params, refit=True)
        model.fit(xs, ys)
        ys_pred_prob = model.predict_proba(test_dataset.xs_feature)
        ys_pred = model.predict(test_dataset.xs_feature)

    acc_test = accuracy_score(test_dataset.ys, ys_pred)
    f1_test = f1_score(test_dataset.ys, ys_pred)
    auc_test = roc_auc_score(test_dataset.ys, ys_pred_prob[:, 1])
    print(f'acc_test: {acc_test}')
    print(f'auc_test: {auc_test}')
    print(f'f1_test: {f1_test}')


def train_disc_model(args, xs_tr, ys_tr_soft, ys_tr_hard, xs_tr_unlabeled, valid_dataset, warmup_dataset):
    # prepare discriminator
    seed = np.random.randint(1e6)
    disc_model = get_discriminator(model_type=args.model_type, prob_labels=args.soft_training, seed=seed)

    if args.dist_shift == 'kliep':        
        kliep = DensityRatioEstimator()
        kliep.fit(xs_tr, valid_dataset.xs_feature)
        weights = kliep.predict(xs_tr)
        ys_tr_soft = (ys_tr_soft.T * weights).T

    if args.soft_training:
        ys_tr = ys_tr_soft[:, 1]
        ys_warmup = (warmup_dataset.ys == 1).astype(float)
    else:
        ys_tr = ys_tr_hard
        ys_warmup = warmup_dataset.ys

    if xs_tr is None:
        assert len(warmup_dataset) > 0
        xs_tr = warmup_dataset.xs_feature
        ys_tr = ys_warmup
    else:
        xs_tr = np.vstack((xs_tr, warmup_dataset.xs_feature))
        ys_tr = np.hstack((ys_tr, ys_warmup))

    sample_weights = None

    if args.model_type == 'ssl':
        raise NotImplementedError
    else:
        disc_model.tune_params(xs_tr, ys_tr, valid_dataset.xs_feature, valid_dataset.ys, sample_weights)
        disc_model.fit(xs_tr, ys_tr, sample_weights)
    
    return disc_model


def train_label_model(args, train_dataset, valid_dataset, lf_agent, discard=None):
    # prepare training and validation label matrix
    L_tr = lf_agent.L_tr
    L_val = lf_agent.L_val

    # filter out abstained entries
    L_tr_filtered, ys_tr_filtered, filter_mask_tr = filter_abtain(L_tr, train_dataset.ys)
    L_val_filtered, ys_val_filtered, filter_mask_val = filter_abtain(L_val, valid_dataset.ys)

    xs_tr_filtered = train_dataset.xs_feature[filter_mask_tr]
    xs_tr_unlabeled = train_dataset.xs_feature[~filter_mask_tr]

    # get lf labels
    lf_labels = lf_agent.get_lf_labels()
    anchors = lf_agent.get_anchors()

    # create label model here 
    class_balance = [0.87, 0.13] if args.dataset == 'sms' else [0.5, 0.5]
    kwargs = {
        'num_lfs': L_tr_filtered.shape[1], 
        'lf_labels': lf_labels,
        'discard': discard,
        'anchors': anchors,
        'class_balance': class_balance
    }

    if L_tr.shape[1] < 3:
        label_model = get_label_model('mv', **kwargs)
    else:
        label_model = get_label_model(args.label_model, **kwargs)

    L_tr_filtered, filter_mask_tr = label_model.fit(L_tr_filtered, L_val_filtered, ys_val_filtered,
            xs_tr=train_dataset.xs_feature[filter_mask_tr], xs_val=valid_dataset.xs_feature[filter_mask_val])
    
    xs_tr_unlabeled = np.vstack((xs_tr_unlabeled, xs_tr_filtered[~filter_mask_tr]))
    xs_tr_filtered = xs_tr_filtered[filter_mask_tr] # after further discarding
    ys_tr_filtered = ys_tr_filtered[filter_mask_tr] # after further discarding

    ys_pred_tr_filtered_soft = label_model.predict_proba(L_tr_filtered)
    ys_pred_tr_filtered_hard = label_model.predict(L_tr_filtered)

    return (label_model, xs_tr_filtered, ys_tr_filtered, ys_pred_tr_filtered_soft,
           ys_pred_tr_filtered_hard, xs_tr_unlabeled)


def train_implyloss(args, train_dataset, valid_dataset, lf_agent, discard=None):
    # prepare training and validation label matrix
    L_tr = lf_agent.L_tr
    L_val = lf_agent.L_val

    # filter out abstained entries
    L_tr_filtered, ys_tr_filtered, filter_mask_tr = filter_abtain(L_tr, train_dataset.ys)

    # get lf labels
    lf_labels = lf_agent.get_lf_labels()
    anchors = lf_agent.get_anchors()
    anchors_idx = lf_agent.get_anchors_idx()

    # create label model here
    class_balance = [0.87, 0.13] if args.dataset == 'sms' else [0.5, 0.5]
    kwargs = {
        'num_lfs': L_tr_filtered.shape[1],
        'lf_labels': lf_labels,
        'discard': discard,
        'anchors': anchors,
        'anchors_idx': anchors_idx,
        'class_balance': class_balance
    }

    # for implyloss we don't have to filter valid data
    implyloss_model = ImplyLoss(**kwargs)
    implyloss_model.fit(L_tr, L_val, valid_dataset.ys,
                    xs_tr=train_dataset.xs_feature, xs_val=valid_dataset.xs_feature)

    return implyloss_model


def eval_disc_model(t, disc_model, test_dataset, history):
    ys_pred_prob = disc_model.predict_proba(test_dataset.xs_feature)
    ys_pred = disc_model.predict(test_dataset.xs_feature)

    acc_test = accuracy_score(test_dataset.ys, ys_pred)
    f1_test = f1_score(test_dataset.ys, ys_pred)
    auc_test = roc_auc_score(test_dataset.ys, ys_pred_prob[:, 1])

    print('Number of queries: {}; Test Acc: {}'.format(t, acc_test))
    print('Number of queries: {}; Test AUC: {}'.format(t, auc_test))
    print('Number of queries: {}; Test F1: {}'.format(t, f1_test))
    history['test_acc'].append(acc_test)
    history['test_auc'].append(auc_test)
    history['test_f1'].append(f1_test)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Interactive Data Programming')
    # paths
    parser.add_argument('--root_dir', type=str, default='./')
    parser.add_argument('--save_dir', type=str, default='vldb_results')
    parser.add_argument('--data_dir', type=str, default="data")
    # dataset settings
    parser.add_argument('--dataset', type=str, default='youtube')
    parser.add_argument('--test-ratio', type=float, default=0.1)
    parser.add_argument('--valid-ratio', type=float, default=0.1)
    parser.add_argument('--warmup-ratio', type=float, default=0)
    parser.add_argument('--feature', type=str, default='tfidf')
    # framework settings
    parser.add_argument('--num-query', type=int, default=50)
    parser.add_argument('--query-size', type=int, default=1)
    parser.add_argument('--train-iter', type=int, default=5)
    # lf simulation settings
    parser.add_argument('--lf-model', type=str, default='heuristic', choices=['learned', 'heuristic'])
    parser.add_argument('--embedding-method', type=str, default='lf', choices=['lf', 'kw'])
    parser.add_argument('--lf-method', type=str, default='sentiment')
    parser.add_argument('--lf-score', type=str, default='val_label_acc')
    parser.add_argument('--lf-acc', type=float, default=0.5)
    parser.add_argument('--lf-simulate', type=str, default='lexicon')
    parser.add_argument('--lf-restrict', type=str, default="lf_model")
    # model settings
    parser.add_argument('--model-type', type=str, default='logistic')
    parser.add_argument('--label-model', type=str, default='mv')
    # query method settings
    parser.add_argument('--query-agent', type=str, default='greedy', choices=['greedy', 'learned'])
    parser.add_argument('--query-method', type=str, default='uncertainty_lm')
    parser.add_argument('--seu', action='store_true')
    parser.add_argument('--lexicon', type=int, default=None)
    parser.add_argument('--pn', action='store_true')
    parser.add_argument('--kw', action='store_true')
    parser.add_argument('--use-ys-pred', action='store_true')
    parser.add_argument('--aggregate', type=str, default=None)
    # learning method settings
    parser.add_argument('--soft-training', action='store_true')
    parser.add_argument('--dist-shift', type=str, default=None)
    parser.add_argument('--discard', type=str, default=None)
    # experiment settings
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--runs', type=int, nargs='+', default=range(1))
    parser.add_argument('--qei', action='store_true')
    parser.add_argument('--shap-repeat', type=int, default=10)
    # run supervised methods
    parser.add_argument('--run-fs', action='store_true')
    parser.add_argument('--run-vs', action='store_true')
    # run exploration step
    parser.add_argument('--run-explore', action='store_true')


    args = parser.parse_args()

    group_id = wandb.util.generate_id()

    # handle relative paths
    save_dir = f'{args.root_dir}/{args.save_dir}'
    data_dir = f'{args.root_dir}/{args.data_dir}'

    np.random.seed(args.seed)
    rand_state = np.random.RandomState(args.seed)

    # Load datasets
    (train_dataset, valid_dataset,
    test_dataset, warmup_dataset) = load_data(data_dir, args.dataset, args.feature, args.test_ratio,
                                    args.valid_ratio, args.warmup_ratio, rand_state=rand_state)

    print(f'train size: {len(train_dataset)}')
    print(f'valid size: {len(valid_dataset)}')
    print(f'test size: {len(test_dataset)}')
    print(f'warmup size: {len(warmup_dataset)}')

    # run fully-supervised baseline
    if args.run_fs:
        run_fs(args, train_dataset, valid_dataset, warmup_dataset, test_dataset)
        sys.exit(1)

    # run validation-supervised baseline
    if args.run_vs:
        run_vs(args, valid_dataset, test_dataset)
        sys.exit(1)

    # run exploration process
    if args.run_explore:
        pos_frac = np.mean(train_dataset.ys == 1)
        neg_frac = np.mean(train_dataset.ys == -1)
        print(f"Pos frac: {pos_frac}, Neg frac: {neg_frac}")
        keyword = input("Enter keyword (Enter T to terminate)")

        while keyword != "T":
            active_instances = {
                "idx": [],
                "text": [],
                "label": []
            }
            for i in range(len(train_dataset.xs_token)):
                if keyword in train_dataset.xs_token[i]:
                    active_instances["idx"].append(i)
                    active_instances["text"].append(train_dataset.xs_text[i])
                    active_instances["label"].append(train_dataset.ys[i])
                    print("Text:", train_dataset.xs_text[i])
                    print("Label:", train_dataset.ys[i])

            active_instances = pd.DataFrame(active_instances)
            pos_frac = np.mean(active_instances["label"] == 1)
            neg_frac = np.mean(active_instances["label"] == -1)
            coverage = len(active_instances) / len(train_dataset.xs_token)
            print(f"Keyword: {keyword}, coverage: {coverage}, Pos frac: {pos_frac}, Neg frac: {neg_frac}")
            cor_keyword = input("Enter correlation keyword to check (Enter T to terminate)")
            while cor_keyword != "T":
                cor_exist_labels = []
                cor_nonexist_labels = []
                for i in range(len(active_instances)):
                    idx = active_instances["idx"][i]
                    if cor_keyword in train_dataset.xs_token[idx]:
                        cor_exist_labels.append(active_instances["label"][i])
                    else:
                        cor_nonexist_labels.append(active_instances["label"][i])
                cor_exist_labels = np.array(cor_exist_labels)
                ce_cov = len(cor_exist_labels) / len(train_dataset.xs_token)
                ce_pos = np.mean(cor_exist_labels == 1)
                ce_neg = np.mean(cor_exist_labels == -1)
                cor_nonexist_labels = np.array(cor_nonexist_labels)
                cn_cov = len(cor_nonexist_labels) / len(train_dataset.xs_token)
                cn_pos = np.mean(cor_nonexist_labels == 1)
                cn_neg = np.mean(cor_nonexist_labels == -1)
                print(f"When {keyword} co-exist with {cor_keyword}, coverage: {ce_cov}, Pos frac: {ce_pos}, Neg frac: {ce_neg}")
                print(f"When {cor_keyword} is absent, coverage: {cn_cov}, Pos frac: {cn_pos}, Neg frac: {cn_neg}")
                cor_keyword = input("Enter correlation keyword to check (Enter T to terminate)")

            keyword = input("Enter keyword (Enter T to terminate)")
        sys.exit(1)





        
    # create sentiment lexicon
    sentiment_lexicon = SentimentLexicon(data_dir)

    for run in args.runs:
        # record results to wandb
        wandb.init(
            project="hf-idp",
            config={
                "dataset": args.dataset,
                "lf-model": args.lf_model,
                "lf-score": args.lf_score,
                "query-agent": args.query_agent,
                "user-model-input": args.embedding_method,
                "lf-simulate": args.lf_simulate,
                "lf-acc-threshold": args.lf_acc,
                "group-id": group_id
            }
        )

        n_candidate_lfs = []
        # set random seed
        np.random.seed(run)
        rand_state = np.random.RandomState(run)

        # init LF model (used only in QEU)
        if args.lf_model == "heuristic":
            lf_model = LFModel(train_dataset.xs_token, sentiment_lexicon, pn=args.pn, kw=args.kw, dict_size=args.lexicon)
        elif args.lf_model == "learned":
            lf_model = LearnedLFModel(train_dataset.xs_token, sentiment_lexicon, pn=args.pn, kw=args.kw, dict_size=args.lexicon,
                                      embedding_method=args.embedding_method, rand_state=rand_state, lf_score=args.lf_score)
        else:
            lf_model = None

        # init query agent
        if args.query_agent in ["greedy", "random"]:
            query_agent = QueryAgent(train_dataset.xs_feature, train_dataset.xs_token,
                                    args.query_method, args.query_size, rand_state, False, args.qei, args.aggregate, lf_model)
        elif args.query_agent == "learned":
            query_agent = LearnedQueryAgent(train_dataset.xs_feature, train_dataset.xs_token,
                                    args.query_method, args.query_size, rand_state, False, args.qei, args.aggregate, lf_model)
        else:
            raise ValueError("Query agent not supported.")

        # init lf agent
        if args.lf_restrict == "lf_model":
            keyword_dict = lf_model.lexicon.keyword_dict
        else:
            keyword_dict = None

        lf_agent = LFAgent(train_dataset, valid_dataset, sentiment_lexicon, method=args.lf_method, rand_state=rand_state,
                           keyword_dict=keyword_dict, vectorizer=lf_model.vectorizer_pos, lf_acc=args.lf_acc, lf_simulate=args.lf_simulate)
        

        # init training data / model / log
        xs_tr = None
        ys_pred_tr_soft = None
        ys_pred_tr_hard = None
        xs_tr_unlabeled = train_dataset.xs_feature

        label_model = None
        disc_model = None

        history = defaultdict(list)

        for t in range(args.num_query + 1):
            # train end model
            if t % args.train_iter == 0:
                if t == 0 and len(warmup_dataset) == 0:
                    pass
                else:
                    if -1 in ys_pred_tr_hard and 1 in ys_pred_tr_hard:
                        disc_model = train_disc_model(args, xs_tr, ys_pred_tr_soft, ys_pred_tr_hard, xs_tr_unlabeled, valid_dataset, warmup_dataset)
                        eval_disc_model(t, disc_model, test_dataset, history)
                        ys_pred = disc_model.predict(train_dataset.xs_feature)
                        if args.lf_model == "learned":
                            lf_model.user_model.train_model(ys_pred=ys_pred)
                            lf_model.evaluate_model_acc(lf_agent, history, ys_pred=ys_pred)
                            wandb.log(
                                {
                                    "num_query": t,
                                    "train_precision": history["train_precision"][-1],
                                    "train_coverage": history["train_coverage"][-1],
                                    "test_acc": history["test_acc"][-1],
                                    "test_auc": history["test_auc"][-1],
                                    "test_f1": history["test_f1"][-1],
                                    "user_model_acc": history["user_model_accuracy"][-1],
                                    "user_model_precision": history["user_model_precision"][-1],
                                    "user_model_recall": history["user_model_recall"][-1],
                                    "user_model_f1": history["user_model_f1"][-1],
                                }
                            )
                        else:
                            wandb.log(
                                {
                                    "num_query": t,
                                    "train_precision": history["train_precision"][-1],
                                    "train_coverage": history["train_coverage"][-1],
                                    "test_acc": history["test_acc"][-1],
                                    "test_auc": history["test_auc"][-1],
                                    "test_f1": history["test_f1"][-1],
                                }
                            )

                    else:
                        raise ValueError("Only one class exist in current prediction")

                    if t == args.num_query:
                        break

            # query for new lf
            if (t // args.train_iter) == 0 or args.query_agent == "random" or \
                    (args.query_method in ['uncertainty_mix', 'uncertainty_dm'] and disc_model is None):
                cur_query_idxs = query_agent.warm_start()
            else:
                assert label_model is not None

                L_tr = lf_agent.L_tr

                if disc_model is None:
                    ys_pred = None
                else:
                    # ys_pred = disc_model.predict_proba(train_dataset.xs_feature)
                    ys_pred = disc_model.predict(train_dataset.xs_feature)

                if args.query_agent == "learned":
                    cur_query_idxs = query_agent.query_(lf_agent, ys_pred=to_onehot(ys_pred))
                else:
                    cur_query_idxs = query_agent.query(L_tr, label_model, ys_pred=to_onehot(ys_pred),
                                                       use_ys_pred=args.use_ys_pred, disc_model=disc_model)

            print('Queried Example: {}'.format(train_dataset.xs_text[cur_query_idxs[0]]))

            lf = lf_agent.create_lf(cur_query_idxs)
            n_candidate_lfs.append(lf_agent.n_candidate_lfs)

            if lf is not None:
                if args.lf_method == 'sentiment':
                    print('lf: {} --> {}'.format(lf.keyword, lf.label))
                    if args.lf_model == "learned":
                        lf_model.user_model.add(cur_query_idxs[0], [lf])


                L_tr, _ = lf_agent.update(lf)

                coverage = compute_coverage(L_tr[:, -1])
                precision = compute_gt_precision(L_tr[:, -1], train_dataset.ys)
                print('LF coverage: {}; LF precision: {}'.format(coverage, precision))

                if t % args.train_iter == (args.train_iter - 1):
                    discard = args.discard
                else:
                    discard = None

                (label_model, xs_tr, ys_tr, ys_pred_tr_soft,
                ys_pred_tr_hard, xs_tr_unlabeled) = train_label_model(args, train_dataset, valid_dataset, lf_agent, discard) 

                # verify probabilisitic label quality
                precision_tr = accuracy_score(ys_tr, ys_pred_tr_hard)
                coverage_tr = len(ys_tr) / len(train_dataset)
                print('Recovery Precision: {}'.format(precision_tr))
                print('Coverage: {}'.format(coverage_tr))
                history['train_precision'].append(precision_tr)
                history['train_coverage'].append(coverage_tr)

                if args.qei:
                    lf_model.update(lf.keyword)                
            else:
                print('No LF returned')

                if args.qei:
                    keywords_rm = train_dataset.xs_token[cur_query_idxs[0]]
                    lf_model.update_none(keywords_rm)
                if args.lf_model == "learned":
                    lf_model.user_model.add(cur_query_idxs[0], [])

        average_acc = np.mean(history["test_acc"])
        average_auc = np.mean(history["test_auc"])
        average_f1 = np.mean(history["test_f1"])
        print(f'acc_test_avg: {average_acc}')
        print(f'auc_test_avg: {average_auc}')
        print(f'f1_test_avg: {average_f1}')

        wandb.finish()

        save_path = f'./feature_{args.feature}_warmup_{args.warmup_ratio}_val_{args.valid_ratio}_lf_{args.lf_acc}_simulate_{args.lf_simulate}'\
                    f'/{args.dataset}/{args.model_type}/{args.lf_method}'\
                    f'/{args.label_model}_{args.soft_training}'\
                    f'/{args.query_method}_{args.qei}_{args.use_ys_pred}_{args.aggregate}_{args.pn}'\
                    f'_{args.kw}_{args.lexicon}_{args.discard}_{args.dist_shift}/{run}'

        save_path = os.path.join(save_dir, save_path)

        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))
        with open(save_path, 'w') as f:
            json.dump(history, f)

        # calculate the shapley value of each LF
        lfs = lf_agent.lfs
        n_lfs = len(lfs)
        shap_scores = np.zeros(n_lfs)
        counts = np.zeros(n_lfs)
        for i in range(args.shap_repeat):
            history = defaultdict(list)
            lf_agent_shap = LFAgent(train_dataset, valid_dataset, sentiment_lexicon, method=args.lf_method, rand_state=rand_state,
                           keyword_dict=keyword_dict, vectorizer=lf_model.vectorizer_pos, lf_acc=args.lf_acc, lf_simulate=args.lf_simulate)
            order = np.random.permutation(n_lfs)
            classes = []
            for j in range(n_lfs):
                idx = order[j]  # this idx is the index of lf in lf_list, not the global idx
                lf = lfs[idx]
                if lf.label not in classes:
                    classes.append(lf.label)
                L_tr, _ = lf_agent_shap.update(lf)
                if len(classes) == 2:
                    (label_model, xs_tr, ys_tr, ys_pred_tr_soft,
                     ys_pred_tr_hard, xs_tr_unlabeled) = train_label_model(args, train_dataset, valid_dataset, lf_agent_shap,
                                                                           discard)
                    disc_model = train_disc_model(args, xs_tr, ys_pred_tr_soft, ys_pred_tr_hard, xs_tr_unlabeled,
                                                  valid_dataset, warmup_dataset)
                    eval_disc_model(j+1, disc_model, test_dataset, history)
                    if len(history["test_acc"]) > 1:
                        shap_scores[idx] += history["test_acc"][-1] - history["test_acc"][-2]
                        counts[idx] += 1

        shap_scores = np.divide(shap_scores, counts, out=np.zeros_like(shap_scores), where=counts!=0)
        L_tr = lf_agent.L_tr
        shap_order = np.argsort(shap_scores)[::-1]

        results = {
            "lf": [],
            "shapley": [],
            "coverage": [],
            "precision": []
        }

        for idx in shap_order:
            lf = lfs[idx]
            print('lf: {} --> {}'.format(lf.keyword, lf.label))
            coverage = compute_coverage(L_tr[:, idx])
            precision = compute_gt_precision(L_tr[:,idx], train_dataset.ys)
            print('Shapley value: {}; LF coverage: {}; LF precision: {}'.format(shap_scores[idx],coverage, precision))
            results["lf"].append('{} --> {}'.format(lf.keyword, lf.label))
            results["shapley"].append(shap_scores[idx])
            results["coverage"].append(coverage)
            results["precision"].append(precision)

        results = pd.DataFrame(results)
        results.to_csv(save_path+"_shap.csv")





