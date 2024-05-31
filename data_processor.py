import json
import re
import random
from os import listdir
import csv

test_data = []


def realtimeQA_reader():
    inpath = "/shared/xdyu/msr/container/data/realtimeqa_public/past/"
    query_counter = len(test_data)
    for year in sorted(listdir(inpath)):
        for f in sorted(listdir(inpath+year)):
            if 'qa.jsonl' not in f:
                continue 
            infile = open(inpath + year + '/' + f, 'r')
            for line in infile:
                question = json.loads(line.strip())
                if len(question['evidence']) == 0:
                    print(question)
                    continue
                query = {
                    'q_id': query_counter,
                    'question': question['question_sentence'],
                    'choices': question['choices'],
                    'answer': question['choices'][int(question['answer'][0])],
                    'evidence': question['evidence'],
                    'source': 'realtimeqa'
                }
                test_data.append(query)
                query_counter += 1
    print("RealtimeQA query size: ", query_counter)
    return


def nq_reader():
    infile = open('/shared/xdyu/msr/container/data/MRQA-Shared-Task-2019/data/NaturalQuestions.jsonl', 'r')

    html_pattern = '<[^<]+?>'

    query_counter = 0
    for line in infile:
        question = json.loads(line.strip())
        if 'header' in question:
            continue
        query = {
            'q_id': question['qas'][0]['qid'],
            'question': question['qas'][0]['question'],
            'answer': question['qas'][0]['answers'][0],
            'evidence': re.sub(html_pattern, '', question['context']).strip(),
            'source': 'nq'
        }
        test_data.append(query)
        query_counter += 1
    print(query_counter)
    return


def remove_duplicates(data):
    questions = {}
    duplicate_counter = 0
    new_data = []
    for case in data:
        if case['question'] + case['evidence'] + case['answer'] in questions:
            duplicate_counter += 1
            continue
        else: 
            questions[case['question'] + case['evidence'] + case['answer']] = True
        new_data.append(case)
    
    print(duplicate_counter)

    return new_data


def main():
    
    # realtimeQA_reader()
    nq_reader()
    new_data = remove_duplicates(test_data)
    outfile = open('/shared/xdyu/msr/container/data/realtime.json', 'w')
    json.dump(new_data, outfile, indent=4)



if __name__ == "__main__":
    main()