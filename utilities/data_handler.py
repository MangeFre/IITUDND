from dateutil import parser
from collections import defaultdict
import json
import random
import csv

RANDOM_SEED = 42  # this ensures that you get the same labels if have seperate datahandlers for different extractions


class DataHandler:
    """
    A class to unify data
    """

    def __init__(self, unlabeled_data_filename, labeled_data_filename, classifications_filename):

        # set random seed for reproducable data sets
        random.seed(RANDOM_SEED)

        # Establish dict of annotated scores for each tweet ID
        class_encoding = {'dont_know_or_cant_judge': -1, 'informative': 1, 'not_informative': 0}
        self.id_to_class = {}  # this is the binary informative / not informative score
        # CHANGE FILEPATH — Read in the classifications of each tweet
        with open(classifications_filename) as fin:
            reader = csv.DictReader(fin, dialect='excel-tab')
            for line in reader:
                self.id_to_class[int(line['tweet_id'])] = class_encoding[line['text_info']]

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
        self.overlap = 0  # number of tweets overlapping between unlabeled and labeled
        self.merged = self.labeled.copy()
        for tweet_json in self.unlabeled:
            tweet_id = tweet_json['id']
            if tweet_id in self.labeled_tweet_ids:
                self.overlap += 1
                continue
            self.merged.append(tweet_json)

        # get dict of {user id : [labeled_tweet1_by_user, labeled_tweet2_by_user, etc]}
        self.labeled_tweets_by_user = defaultdict(list)
        for tweet_json in self.labeled:
            user_id = tweet_json['user']['id']
            self.labeled_tweets_by_user[user_id].append(tweet_json)

        # get user histories
        self.full_sorted_history_by_user = defaultdict(list)
        for tweet_json in self.merged:
            user_id = tweet_json['user']['id']
            self.full_sorted_history_by_user[user_id].append(tweet_json)
        # sort each list of tweets by 'created_at' by each user
        for key, val in self.full_sorted_history_by_user.items():
            self.full_sorted_history_by_user[key] = sorted(val, key=lambda i: parser.parse(
                i['created_at']).timestamp())

    def get_train_test_split(self, ratio_of_test_samples=.2):

        basket_of_users = [user_id for user_id in self.labeled_tweets_by_user.keys()]
        random.shuffle(basket_of_users)

        goal_number_of_test_tweets = round(len(self.labeled) * ratio_of_test_samples)

        # grab user data until threshold
        test_labeled = []
        test_histories = []
        test_classes = []
        test_histories_by_target = []
        cur_user_idx = 0
        while len(test_labeled) < goal_number_of_test_tweets:
            cur_user = basket_of_users[cur_user_idx]
            labeled_tweets = self.labeled_tweets_by_user[cur_user]
            test_labeled.extend(labeled_tweets)
            test_classes.extend([self.id_to_class[json['id']] for json in labeled_tweets])

            # put the users history in for each of their tweets (to keep test_history aligned with test_labeled)
            for labeled_tweet in labeled_tweets:
                test_histories.append(self.full_sorted_history_by_user[cur_user])
                cur_labeled_tweet = 0
                for tweet in self.full_sorted_history_by_user[cur_user]:
                    if tweet.get('id') == labeled_tweet.get('id'):
                        break
                    cur_labeled_tweet += 1
                test_histories_by_target.append(self.full_sorted_history_by_user[cur_user][:cur_labeled_tweet + 1])
            cur_user_idx += 1

        # put the remaining data in the training set
        train_labeled = []
        train_histories = []
        train_classes = []
        train_histories_by_target = []
        for cur_user in basket_of_users[cur_user_idx:]:
            labeled_tweets = self.labeled_tweets_by_user[cur_user]
            train_labeled.extend(labeled_tweets)
            train_classes.extend([self.id_to_class[json['id']] for json in labeled_tweets])

            # put the users history in for each of their tweets (to keep train_history aligned with train_labeled)
            for labeled_tweet in labeled_tweets:
                train_histories.append(self.full_sorted_history_by_user[cur_user])
                cur_labeled_tweet = 0
                for tweet in self.full_sorted_history_by_user[cur_user]:
                    if tweet.get('id') == labeled_tweet.get('id'):
                        break
                    cur_labeled_tweet += 1
                train_histories_by_target.append(self.full_sorted_history_by_user[cur_user][:cur_labeled_tweet + 1])

        # get the merged (labeled and unlabeled data without duplicates from overlap) for just the users in trainset
        training_users = set(basket_of_users[cur_user_idx:])
        train_merged = []
        for tweet_json in self.merged:
            if tweet_json['user']['id'] in training_users:
                train_merged.append(tweet_json)

        return train_labeled, train_histories, train_histories_by_target, test_labeled, test_histories, test_histories_by_target, train_merged, train_classes, test_classes

    def get_k_fold_split(self, k):
        train, train_h, train_hbt, test, test_h, test_hbt, train_m, train_c, test_c = datahandler.get_train_test_split()
        num_per_train_fold = len(train) // k
        num_per_test_fold = len(test) // k
        validation = []
        train_tweet_cursor = 0
        test_tweet_cursor = 0
        for i in range(k):
            if i == k - 1:
                # dump rest of the data
                validation.append(([train[train_tweet_cursor:],
                                    train_h[train_tweet_cursor:],
                                    train_hbt[train_tweet_cursor:],
                                    train_c[train_tweet_cursor:],
                                    train_m[train_tweet_cursor:]],
                                   [test[test_tweet_cursor:],
                                    test_h[test_tweet_cursor:],
                                    test_hbt[test_tweet_cursor:],
                                    test_c[test_tweet_cursor:], None]))
                break
            # initialize fold counter
            train_fold_counter = 0
            test_fold_counter = 0
            # get the train set split index
            cur_train_user_id = train[train_tweet_cursor].get('user').get('id')
            while train_fold_counter < num_per_train_fold or cur_train_user_id == train[
                train_tweet_cursor + train_fold_counter + 1].get('user').get('id'):
                train_fold_counter += 1
            # get the test set split index
            cur_test_user_id = test[test_tweet_cursor].get('user').get('id')
            while test_fold_counter < num_per_test_fold or cur_test_user_id == test[
                test_tweet_cursor + test_fold_counter + 1].get('user').get('id'):
                test_fold_counter += 1

            validation.append(([train[train_tweet_cursor:train_tweet_cursor + train_fold_counter],
                                train_h[train_tweet_cursor:train_tweet_cursor + train_fold_counter],
                                train_hbt[train_tweet_cursor:train_tweet_cursor + train_fold_counter],
                                train_c[train_tweet_cursor:train_tweet_cursor + train_fold_counter],
                                train_m[train_tweet_cursor:train_tweet_cursor + train_fold_counter]],
                               [test[test_tweet_cursor:test_tweet_cursor + test_fold_counter],
                                test_h[test_tweet_cursor:test_tweet_cursor + test_fold_counter],
                                test_hbt[test_tweet_cursor:test_tweet_cursor + test_fold_counter],
                                test_c[test_tweet_cursor:test_tweet_cursor + test_fold_counter], None]))
            train_tweet_cursor += train_fold_counter
            test_tweet_cursor += test_fold_counter
        # todo: do evaluation here
        evaluation = ([train, train_h, train_hbt, train_c, train_m], [test, test_h, test_hbt, test_c, None])
        return evaluation, validation


UNLABELED_DATA = '/Users/maw/PyCharmProjects/IITUDND/data/retrieved_data/calfire_extras.json'
LABELED_DATA = '/Users/maw/PyCharmProjects/IITUDND/data/CrisisMMD_v1.0/json/california_wildfires_final_data.json'
CLASSIFICATIONS = '/Users/maw/PyCharmProjects/IITUDND/data/CrisisMMD_v1.0/annotations/california_wildfires_final_data.tsv'
datahandler = DataHandler(UNLABELED_DATA, LABELED_DATA, CLASSIFICATIONS)

train_labeled, train_histories, train_histories_by_target, test_labeled, test_histories, test_histories_by_target, train_merged, train_classes, test_classes = datahandler.get_train_test_split()

# eva = tuple(  train[train_labeled, train_histories, train_histories_by_target, train_classes, train_merged],
#               test[test_labeled, test_histories, test_histories_by_target, test_classes, None])
# val = list[   tuple_1(train[train_labeled, train_histories, train_histories_by_target, train_merged, train_classes],
#                       test[test_labeled, test_histories, test_histories_by_target, None, test_classes]),
#               tuple_2()...
#               tuple_k()]
eva, val = datahandler.get_k_fold_split(10)
