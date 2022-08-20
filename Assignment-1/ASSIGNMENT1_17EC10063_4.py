import os
import sys
import re
import json
import copy
import string

with open('./inverted_index.json', 'r') as f:
    inverted_index = json.load(f)

query_file = ''
tokens_prefix = []
tokens_suffix = []

''' Precomputes the sorted list of tokens both for prefix and suffix matching '''


def precompute():
    tokens_prefix = list(inverted_index.keys())
    tokens_suffix = list([key[::-1] for key in inverted_index.keys()])

    tokens_prefix.sort()
    tokens_suffix.sort()

    return tokens_prefix, tokens_suffix


''' Returns the list of matching tokens which have 'query' as their prefix '''


def solve_prefix(query):

    left = 0
    right = len(tokens_prefix) - 1
    position = len(tokens_prefix)

    while left <= right:
        mid = (left + right) // 2
        if tokens_prefix[mid] >= query:
            position = mid
            right = mid - 1
        else:
            left = mid + 1

    if position == len(tokens_prefix):
        return []

    matchings = []

    for itr in range(position, len(tokens_prefix)):
        if not tokens_prefix[itr].startswith(query):
            break
        matchings.append(tokens_prefix[itr])
    return matchings


''' Returns the list of matching tokens which have 'query' as their suffix '''


def solve_suffix(query):
    left = 0
    right = len(tokens_suffix) - 1
    position = len(tokens_suffix)

    while left <= right:
        mid = (left + right) // 2
        # print(left, right, mid, tokens_suffix[mid])
        if tokens_suffix[mid] >= query:
            position = mid
            right = mid - 1
        else:
            left = mid + 1

    if position == len(tokens_suffix):
        return []

    matchings = []

    for itr in range(position, len(tokens_suffix)):
        if not tokens_suffix[itr].startswith(query):
            break
        matchings.append(tokens_suffix[itr][::-1])
    return matchings


''' Intersect the list of prefix and suffix tokens to return the common ones '''


def intersect(a, b):
    return list(set(a).intersection(b))


''' Output the answers (<docID, posID>) for the given 'matching' '''


def write_answer(file, matching, result):

    file.write(matching + ':')
    count = 0
    for post in result:
        file.write('<{},{}>'.format(post[0], post[1]))
        count += 1
        if count != len(result):
            file.write(',')
    return


''' Prepares the results for all the matchings to the 'input_query' '''


def get_postings(result, out_file, input_query):

    postings = []

    count = 0
    result.sort()
    for key in result:
        # handle cases like wo*ow which can match to wow
        if len(key) >= len(input_query) - 1:
            if count:
                out_file.write('; ')
            posting = inverted_index[key]
            posting.sort(key=lambda x: (x[0], x[1]))
            write_answer(out_file, key, posting)
            count += 1
    return


''' Driver function which takes computes the result for all the queries in the 'query_file' '''


def answer_queries():

    out_file = open('RESULTS1_17EC10063.txt', 'w')

    with open(query_file, 'r') as f:
        queries = f.readlines()
        for query in queries:
            if query.endswith('\n'):
                query = query[:-1]
            if query[0] == '*':
                result = solve_suffix(query[1:][::-1])
            elif query[-1] == '*':
                result = solve_prefix(query[:-1])
            else:
                left, right = query.split('*')
                result_left = solve_prefix(left)
                result_right = solve_suffix(right[::-1])
                result = intersect(result_left, result_right)
            get_postings(result, out_file, query)
            out_file.write('\n')
    out_file.close()


if __name__ == "__main__":

    try:
        assert len(sys.argv) == 2
        tokens_prefix, tokens_suffix = precompute()
        query_file = sys.argv[1]
        answer_queries()
    except AssertionError:
        print('Please enter the query file name')
