
from collections import defaultdict
import json
import random
import csv

RANDOM_SEED = 42 # this ensures that you get the same labels if have seperate datahandlers for different extractions

class DataHandler:
    """
    A class to unify data
    """
    def __init__(self, unlabeled_data_filename, labeled_data_filename, classifications_filename):

        # set random seed for reproducable data sets
        random.seed(RANDOM_SEED)

        # Establish dict of annotated scores for each tweet ID
        class_encoding = {'dont_know_or_cant_judge':-1 , 'informative': 1, 'not_informative': 0}
        self.id_to_class = {}  # this is the binary informative / not informative score
        # CHANGE FILEPATH — Read in the classifications of each tweet
        with open(classifications_filename) as fin:
            reader = csv.DictReader(fin, dialect='excel-tab')
            for line in reader:
                self.id_to_class[int(line['tweet_id'])]=class_encoding[line['text_info']]

        # load json data
        with open(unlabeled_data_filename) as fin:
            self.unlabeled = [json.loads(line) for line in fin.readlines()]
        with open(labeled_data_filename) as fin:
            jsons = [json.loads(line) for line in fin.readlines()]
            # skip the don't know classifications (there are only 6)
            self.labeled = [json for json in jsons if self.id_to_class[json['id']] != -1]

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
        test_classes = []
        cur_user_idx = 0
        while(len(test_labeled) < goal_number_of_test_tweets):
            cur_user = basket_of_users[cur_user_idx]
            labeled_tweets = self.labeled_tweets_by_user[cur_user]
            test_labeled.extend(labeled_tweets)
            test_classes.extend([self.id_to_class[json['id']] for json in labeled_tweets])

            # put the users history in for each of their tweets (to keep test_history aligned with test_labeled)
            for _ in range(len(labeled_tweets)):
                test_histories.append(self.full_history_by_user[cur_user])
            cur_user_idx += 1

        # put the remaining data in the training set
        train_labeled = []
        train_histories = []
        train_classes = []
        for cur_user in basket_of_users[cur_user_idx:]:
            labeled_tweets = self.labeled_tweets_by_user[cur_user]
            train_labeled.extend(labeled_tweets)
            train_classes.extend([self.id_to_class[json['id']] for json in labeled_tweets])

            # put the users history in for each of their tweets (to keep train_history aligned with train_labeled)
            for _ in range(len(labeled_tweets)):
                train_histories.append(self.full_history_by_user[cur_user])

        # get the merged (labeled and unlabeled data without duplicates from overlap) for just the users in trainset
        training_users = set(basket_of_users[cur_user_idx:])
        train_merged = []
        for tweet_json in self.merged:
            if tweet_json['user']['id'] in training_users:
                train_merged.append(tweet_json)

        return train_labeled, train_histories, test_labeled, test_histories, train_merged, train_classes, test_classes
