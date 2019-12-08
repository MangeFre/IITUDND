import csv
from collections import Counter


def parse_annotation(file_path):
    class_list = list()
    pos_text_list = list()
    neg_text_list = list()
    unknown_text_list = list()
    i = 0
    with open(file_path) as fd:
        rd = csv.reader(fd, delimiter="\t", quotechar='"')
        for row in rd:
            if i == 0:
                i += 1
                continue
            c = row[2]
            class_list.append(c)
            if c == 'informative':
                pos_text_list.append(row[12])
            elif c == 'not_informative':
                neg_text_list.append(row[12])
            else:
                unknown_text_list.append(row[12])
    return Counter(class_list), pos_text_list, neg_text_list, unknown_text_list


HARVEY_PATH = "annotations/hurricane_harvey_final_data.tsv"
IRMA_PATH = "annotations/hurricane_irma_final_data.tsv"
MARIA_PATH = "annotations/hurricane_maria_final_data.tsv"

harvey_class_dict, harvey_pos_text_list, harvey_neg_text_list, harvey_unknown_text_list = parse_annotation(HARVEY_PATH)
irma_class_dict, irma_pos_text_list, irma_neg_text_list, irma_unknown_text_list = parse_annotation(IRMA_PATH)
maria_class_dict, maria_pos_text_list, maria_neg_text_list, maria_unknown_text_list = parse_annotation(MARIA_PATH)


def get_word_count(lst_of_s):
    res = 0
    for s in lst_of_s:
        res += len(s.split())
    return res


def get_tweet_avg(pos_text_list, neg_text_list):
    return sum(map(len, pos_text_list)) / len(pos_text_list), \
           sum(map(len, neg_text_list)) / len(neg_text_list), \
           sum(map(len, pos_text_list + neg_text_list)) / len(pos_text_list + neg_text_list), \
           get_word_count(pos_text_list) / len(pos_text_list), \
           get_word_count(neg_text_list) / len(neg_text_list), \
           get_word_count(pos_text_list + neg_text_list) / len(pos_text_list + neg_text_list)


# tweet length is the length of a whole tweet
# average word count is count the list by splitting tweet by space
harvey_pos_tweet_len_avg, harvey_neg_tweet_len_avg, harvey_tweet_len_avg, harvey_pos_word_count_avg, harvey_neg_word_count_avg, harvey_word_count_avg = get_tweet_avg(
    harvey_pos_text_list, harvey_neg_text_list)
irma_pos_tweet_len_avg, irma_neg_tweet_len_avg, irma_tweet_len_avg, irma_pos_word_count_avg, irma_neg_word_count_avg, irma_word_count_avg = get_tweet_avg(
    irma_pos_text_list, irma_neg_text_list)
maria_pos_tweet_len_avg, maria_neg_tweet_len_avg, maria_tweet_len_avg, maria_pos_word_count_avg, maria_neg_word_count_avg, maria_word_count_avg = get_tweet_avg(
    maria_pos_text_list, maria_neg_text_list)
hurricane_pos_tweet_len_avg, hurricane_neg_tweet_len_avg, hurricane_tweet_len_avg, hurricane_pos_word_count_avg, hurricane_neg_word_count_avg, hurricane_word_count_avg = get_tweet_avg(
    harvey_pos_text_list + irma_pos_text_list + maria_pos_text_list,
    harvey_neg_text_list + irma_neg_text_list + maria_neg_text_list)

print('' * 50)
print('Harvey has', harvey_class_dict.get('informative'), 'informative tweets and', harvey_class_dict.get(
    'not_informative'), 'not_informative tweets. The total tweets amount is',
      harvey_class_dict.get('informative') + harvey_class_dict.get('not_informative'))
print('Harvey has an average tweet length of', harvey_pos_tweet_len_avg, 'for informative tweets and',
      harvey_neg_tweet_len_avg, 'for not_informative tweets. The total average tweet length is',
      harvey_tweet_len_avg)
print('Harvey has an average word counts per tweet of', harvey_pos_word_count_avg, 'for informative tweets and',
      harvey_neg_word_count_avg, 'for not_informative tweets. The total average word counts per tweet is',
      harvey_word_count_avg)

print('Irma has', irma_class_dict.get('informative'), 'informative tweets and', irma_class_dict.get(
    'not_informative'), 'not_informative tweets. The total tweets amount is',
      irma_class_dict.get('informative') + irma_class_dict.get('not_informative'))
print('Irma is the only one that has tweets with \'dont_know_or_cant_judge\' class. Total number of such class is',
      len(irma_unknown_text_list))
print('Irma has an average tweet length of', irma_pos_tweet_len_avg, 'for informative tweets and',
      irma_neg_tweet_len_avg, 'for not_informative tweets. The total average tweet length is',
      irma_tweet_len_avg)
print('Irma has an average word counts per tweet of', irma_pos_word_count_avg, 'for informative tweets and',
      irma_neg_word_count_avg, 'for not_informative tweets. The total average word counts per tweet is',
      irma_word_count_avg)

print('Maria has', maria_class_dict.get('informative'), 'informative tweets and', maria_class_dict.get(
    'not_informative'), 'not_informative tweets. The total tweets amount is',
      maria_class_dict.get('informative') + maria_class_dict.get('not_informative'))
print('Maria has an average tweet length of', maria_pos_tweet_len_avg, 'for informative tweets and',
      maria_neg_tweet_len_avg, 'for not_informative tweets. The total average tweet length is',
      maria_tweet_len_avg)
print('Maria has an average word counts per tweet of', maria_pos_word_count_avg, 'for informative tweets and',
      maria_neg_word_count_avg, 'for not_informative tweets. The total average word counts per tweet is',
      maria_word_count_avg)

print('Hurricane Disaster (Harvey, Irma, Maria) has',
      harvey_class_dict.get('informative') + irma_class_dict.get('informative') + maria_class_dict.get('informative'),
      'informative tweets and',
      harvey_class_dict.get('not_informative') + irma_class_dict.get('not_informative') + maria_class_dict.get(
          'not_informative'), 'not_informative tweets. The total tweets amount is',
      harvey_class_dict.get('informative') + irma_class_dict.get('informative') +
      maria_class_dict.get('informative') + harvey_class_dict.get('not_informative') +
      irma_class_dict.get('not_informative') + maria_class_dict.get('not_informative'))
print('Hurricane Disaster has an average tweet length of', hurricane_pos_tweet_len_avg,
      'for informative tweets and', hurricane_neg_tweet_len_avg,
      'for not_informative tweets. The total average tweet length is', hurricane_tweet_len_avg)
print('Hurricane Disaster has an average word counts per tweet of', hurricane_pos_word_count_avg,
      'for informative tweets and', hurricane_neg_word_count_avg,
      'for not_informative tweets. The total average word counts per tweet is', hurricane_word_count_avg)
