import math
from collections import defaultdict
import json
import csv
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
import tweepy
# We must use the API to determine whether the tweets are protected
from tweepy import TweepError
import os
import random

RANDOM_SEED = 42

class DataHandler:
    def __init__(self, unlabeled_data_filename, labeled_data_filename):

        # set random seed for reproducable data sets
        random.seed(RANDOM_SEED)

        # load json data
        with open(unlabeled_data_filename) as fin:
            self.unlabeled = [json.loads(line) for line in fin.readlines()]
        with open(labeled_data_filename) as fin:
            self.labeled = [json.loads(line) for line in fin.readlines()]

        # get set of labeled tweet ids
        self.labeled_tweet_ids = set()
        for tweet_json in self.labeled:
            tweet_id = tweet_json['id']
            if tweet_id in self.labeled_tweet_ids:
                raise RuntimeError('Tweet %s is duplicate in labeled data %s' % (tweet_id, labeled_data_filename))
            self.labeled_tweet_ids.add(tweet_json['id'])

        # get set of unlabeled tweet ids
        self.unlabeled_tweet_ids = set()
        for tweet_json in self.unlabeled:
            tweet_id = tweet_json['id']
            if tweet_id in self.unlabeled_tweet_ids:
                raise RuntimeError('Tweet %s is duplicate in labeled data %s' % (tweet_id, labeled_data_filename))
            self.unlabeled_tweet_ids.add(tweet_json['id'])

        # merge labeled data with unlabeled avoiding duplicates
        self.overlap = 0 # number of tweets overlapping between unlabeled and labeled
        self.merged = self.labeled.copy()
        for tweet_json in self.unlabeled:
            tweet_id = tweet_json['id']
            if tweet_id in self.labeled_tweet_ids:
                self.overlap += 1
            self.merged.append(tweet_json)

        # get dict of {user id : [labeled_tweet1_by_user, labeled_tweet2_by_user, etc]}
        self.labeled_tweets_by_user = defaultdict(list)
        for tweet_json in self.labeled:
            user_id = tweet_json['user']['id']
            self.labeled_tweets_by_user[user_id].append(tweet_json)

        # get user histories (TODO SORT BY DATE)
        self.full_history_by_user = defaultdict(list)
        for tweet_json in self.merged:
            user_id = tweet_json['user']['id']
            self.full_history_by_user[user_id].append(tweet_json)

    def get_train_test_split(self,ratio_of_test_samples = .2):

        basket_of_users = [user_id for user_id in self.labeled_tweets_by_user.keys()]
        random.shuffle(basket_of_users)

        goal_number_of_test_tweets = round(len(self.labeled) * ratio_of_test_samples)

        # grab user data until threshold
        test_labeled = []
        test_histories = []
        cur_user_idx = 0
        while(len(test_labeled) < goal_number_of_test_tweets):
            cur_user = basket_of_users[cur_user_idx]
            labeled_tweets = self.labeled_tweets_by_user[cur_user]
            test_labeled.extend(labeled_tweets)

            # put the users history in for each of their tweets (to keep test_history aligned with test_labeled)
            for _ in range(len(labeled_tweets)):
                test_histories.append(self.full_history_by_user[cur_user])
            cur_user_idx += 1

        # put the remaining data in the training set
        train_labeled = []
        train_histories = []
        for cur_user in basket_of_users[cur_user_idx:]:
            labeled_tweets = self.labeled_tweets_by_user[cur_user]
            train_labeled.extend(labeled_tweets)

            # put the users history in for each of their tweets (to keep train_history aligned with train_labeled)
            for _ in range(len(labeled_tweets)):
                train_histories.append(self.full_history_by_user[cur_user])

        return train_labeled, train_histories, test_labeled, test_histories
