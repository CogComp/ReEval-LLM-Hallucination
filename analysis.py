import json
import string
import regex
import unicodedata
from collections import Counter
from sentence_transformers import CrossEncoder
import random
# from gpt_inference import generate_prompt


nli_model = CrossEncoder('cross-encoder/nli-deberta-v3-base')

class SimpleTokenizer(object):
    ALPHA_NUM = r'[\p{L}\p{N}\p{M}]+'
    NON_WS = r'[^\p{Z}\p{C}]'

    def __init__(self):
        """
        Args:
            annotators: None or empty set (only tokenizes).
        """
        self._regexp = regex.compile(
            '(%s)|(%s)' % (self.ALPHA_NUM, self.NON_WS),
            flags=regex.IGNORECASE + regex.UNICODE + regex.MULTILINE
        )

    def tokenize(self, text, uncased=False):
        # print(text)
        matches = [m for m in self._regexp.finditer(text)]
        if uncased:
            tokens = [m.group().lower() for m in matches]
        else:
            tokens = [m.group() for m in matches]
        # print(tokens)
        return tokens

tokenizer = SimpleTokenizer()

def has_answer(answer, text, tokenizer) -> bool:
    """Check if a document contains an answer string."""
    text = _normalize(text)
    text = tokenizer.tokenize(text, uncased=True)

    answer = _normalize(answer)
    answer = tokenizer.tokenize(answer, uncased=True)
    for i in range(0, len(text) - len(answer) + 1):
        if answer == text[i: i + len(answer)]:
            return True
    return False


def _normalize(text):
    return unicodedata.normalize('NFD', text)


def normalize_answer(s):
    def remove_articles(text):
        return regex.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction, ground_truth):
    if prediction is None:
        return  0
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def entailment_verifier(evi, ans):
    scores = nli_model.predict([(evi, ans)])
    label_mapping = ['contradiction', 'entailment', 'neutral']
    labels = [label_mapping[score_max] for score_max in scores.argmax(axis=1)]
    if labels[0] == 'entailment':
        return True
    return False


def check_string_match(answer, output, case=None, hit=False, EM=False):

    if output is None:
        return False

    norm_ans = normalize_answer(answer)
    if type(output) is str:
        norm_out = normalize_answer(output)
        if EM:
            if norm_ans == norm_out:
                return True
            else:
                return False

        if hit:
            if norm_ans in norm_out:
                return True
            if norm_out in norm_ans:
                return True
            if has_answer(answer, output, tokenizer):
                return True
            return False

        if entailment_verifier(case['question'] + '? ' + output, case['question'] + '? ' + answer):
            return True
        return False


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

    return new_data


def remove_multi_answer_querys(data, data2=None):
    ori_file = open('out/nq_dev_gpt35_short.json', 'r')
    ori_data = json.load(ori_file)
    questions = {}

    query_counter = 0
    for case in ori_data:
        if case['status'] not in ['mem_wrong_evi_correct']:
            continue
        if case['question'] in questions:
            questions[case['question']].append({'q_id': case['q_id'], 'answer': case['answer'], 'evidence': case['evidence']})
        else: 
            questions[case['question']] = [{'q_id': case['q_id'], 'answer': case['answer'], 'evidence': case['evidence']}]
        query_counter += 1

    exclude_qid = {}

    multi_answer_counter = 0
    nest_answer_counter = 0
    diff_answer_counter = 0
    for q in questions:
        if len(questions[q]) > 1:
            multi_answer_counter += 1
            sorted_pairs = sorted(questions[q],  key=lambda i: len(i['answer']))
            nest_flag = True
            for i in range(0, len(sorted_pairs)-1):
                if check_string_match(sorted_pairs[i]['answer'], sorted_pairs[i+1]['answer'], hit=True):
                    continue
                else:
                    nest_flag = False
            if nest_flag:
                nest_answer_counter += 1
                unique_evi = {}
                for case in questions[q]:
                    if case['evidence'] in unique_evi:
                        unique_evi[case['evidence']].append(case['q_id'])
                    else:
                        unique_evi[case['evidence']] = [case['q_id']]
                for evi in unique_evi:
                    samples = unique_evi[evi][1:]
                    for i in samples:
                        exclude_qid[i] = True
            else:
                diff_answer_counter += 1
                for case in questions[q]:
                    exclude_qid[case['q_id']] = True

    new_data = []
    query_size = 0
    for case in data:
        if case['status'] not in ['mem_wrong_evi_correct']:
            continue
        if case['q_id'] in exclude_qid:
            case['duplicate'] = True
        else:
            case['duplicate'] = False
            query_size += 1
        new_data.append(case)

    if data2 is not None:
        new_data2 = []
        for case in data2:
            if case['q_id'] in exclude_qid:
                case['duplicate'] = True
            else:
                case['duplicate'] = False
                query_size += 1
            new_data2.append(case)
        return new_data, new_data2
    
    return new_data


def cat1_result_analysis(path, dump=False, bob=False, EM=False):
    infile = open(path, 'r')
    data = json.load(infile)

    status = {}

    valid_counter = 0

    errors = []

    mem_token_F1 = 0
    evi_token_F1 = 0
    for case in data:
        if 'duplicate' in case and case['duplicate'] == True:
            continue

        if case['source'] not in status:
            status[case['source']] = {
                'mem_wrong_evi_correct': 0,
                'mem_wrong_evi_wrong': 0,
                'mem_correct_evi_correct': 0,
                'mem_correct_evi_wrong': 0
            }
        valid_counter += 1

        status_key = ''
        if bob:
            mem_token_F1 += f1_score(case['memory_output'], case['answer'])
            evi_token_F1 += f1_score(case['bob_output'], case['answer'])
            mem_correct = check_string_match(case['answer'], case['memory_output'], case, EM=EM)
            evi_correct = check_string_match(case['answer'], case['bob_output'], case, EM=EM)
            status_key = 'bob_status'
        else:
            mem_token_F1 += f1_score(case['memory_output'], case['answer'])
            evi_token_F1 += f1_score(case['evidence_output'], case['answer'])
            mem_correct = check_string_match(case['answer'], case['memory_output'], case, EM=EM)
            evi_correct = check_string_match(case['answer'], case['evidence_output'], case, EM=EM)
            status_key = 'status'
        if mem_correct:
            if evi_correct:
                status[case['source']]['mem_correct_evi_correct'] += 1
                case[status_key] = 'mem_correct_evi_correct'
            else:
                status[case['source']]['mem_correct_evi_wrong'] += 1
                case[status_key] = 'mem_correct_evi_wrong'
        else:
            if evi_correct:
                status[case['source']]['mem_wrong_evi_correct'] += 1
                case[status_key] = 'mem_wrong_evi_correct'
            else:
                status[case['source']]['mem_wrong_evi_wrong'] += 1
                case[status_key] = 'mem_wrong_evi_wrong'

    print("Total: ", valid_counter)

    for dataset in status:
        print(dataset)
        for key in status[dataset]:
            print(key, status[dataset][key], status[dataset][key] / valid_counter)

    print("Mem Token F1: ", mem_token_F1/valid_counter)
    print("Evi Token F1: ", evi_token_F1/valid_counter)
    if dump:
        outfile = open(path, 'w')
        json.dump(data, outfile, indent=4)


def cat2_result_analysis(path1, path2, dump=False, bob=False, EM=False):
    infile = open(path1, 'r')
    query_data = json.load(infile)

    infile = open(path2, 'r')
    evi_data = json.load(infile)

    status = {}

    valid_counter = 0

    if not dump:
        query_data, evi_data = remove_multi_answer_querys(query_data, evi_data)

    for idx, case in enumerate(query_data):
        if case['duplicate'] == True:
            continue

        if case['source'] not in status:
            status[case['source']] = {
                'mem_wrong_evi_correct': 0,
                'mem_wrong_evi_wrong': 0,
                'mem_correct_evi_correct': 0,
                'mem_correct_evi_wrong': 0
            }

        valid_counter += 1
        status_key = ''
        if bob:
            mem_correct = check_string_match(case['answer'], case['memory_output'], case, EM=EM)
            query_evi_correct = check_string_match(case['answer'], case['bob_output'], case, EM=EM)
            evi_evi_correct = check_string_match(case['answer'], evi_data[idx]['bob_output'], case, EM=EM)
            status_key = 'bob_status'
        else:
            mem_correct = check_string_match(case['answer'], case['memory_output'], case, EM=EM)
            query_evi_correct = check_string_match(case['answer'], case['evidence_output'], case, EM=EM)
            evi_evi_correct = check_string_match(case['answer'], evi_data[idx]['evidence_output'], case, EM=EM)
            status_key = 'status'
        if mem_correct:
            if query_evi_correct and evi_evi_correct:
                status[case['source']]['mem_correct_evi_correct'] += 1
                case[status_key] = 'mem_correct_evi_correct'
            else:
                status[case['source']]['mem_correct_evi_wrong'] += 1
                case[status_key] = 'mem_correct_evi_wrong'
        else:
            if query_evi_correct and evi_evi_correct:
                status[case['source']]['mem_wrong_evi_correct'] += 1
                case[status_key] = 'mem_wrong_evi_correct'                    
            else:
                status[case['source']]['mem_wrong_evi_wrong'] += 1
                case[status_key] = 'mem_wrong_evi_wrong'

    print("Total: ", valid_counter)

    for dataset in status:
        print(dataset)
        for key in status[dataset]:
            print(key, status[dataset][key], status[dataset][key] / valid_counter)
            
    if dump:
        outfile = open(path1, 'w')
        json.dump(query_data, outfile, indent=4)

        outfile = open(path2, 'w')
        json.dump(evi_data, outfile, indent=4)


def cat1_evaluate(path, filter_id=None):
    infile = open(path, 'r')
    data = json.load(infile)

    mem_em = 0
    evi_em = 0
    bob_em = 0

    mem_token_F1 = 0
    evi_token_F1 = 0
    bob_token_F1 = 0

    mem_entail = 0
    evi_entail = 0
    bob_entail = 0

    counter = 0

    for case in data:

        if filter_id is not None:
            if str(case['q_id']) not in filter_id:
                continue

        if case['evidence'].count(case['answer']) != 1:
            continue

        counter += 1

        mem_token_F1 += f1_score(case['memory_output'], case['answer'])
        evi_token_F1 += f1_score(case['evidence_output'], case['answer'])
        bob_token_F1 += f1_score(case['bob_output'], case['answer'])

        mem_em_correct = check_string_match(case['answer'], case['memory_output'], case, EM=True)
        evi_em_correct = check_string_match(case['answer'], case['evidence_output'], case, EM=True)
        bob_em_correct = check_string_match(case['answer'], case['bob_output'], case, EM=True)

        mem_entail_correct = check_string_match(case['answer'], case['memory_output'], case, EM=False)
        evi_entail_correct = check_string_match(case['answer'], case['evidence_output'], case, EM=False)
        bob_entail_correct = check_string_match(case['answer'], case['bob_output'], case, EM=False)

        if mem_em_correct:
            mem_em += 1
        if evi_em_correct:
            evi_em += 1
        if bob_em_correct:
            bob_em += 1

        if mem_entail_correct:
            mem_entail += 1
        if evi_entail_correct:
            evi_entail += 1
        if bob_entail_correct:
            bob_entail += 1

    print("Total: ", counter)

    print("Mem EM: ", mem_em / counter)
    print("Evi EM: ", evi_em / counter)
    print("Bob EM: ", bob_em / counter)

    print("Mem Token F1: ", mem_token_F1 / counter)
    print("Evi Token F1: ", evi_token_F1 / counter)
    print("Bob Token F1: ", bob_token_F1 / counter)

    print("Mem Entailment Accuracy: ", mem_entail / counter)
    print("Evi Entailment Accuracy: ", evi_entail / counter)
    print("Bob Entailment Accuracy: ", bob_entail / counter)


def cat2_evaluate(path1, path2, filter_id=None):
    infile = open(path1, 'r')
    query_data = json.load(infile)

    infile = open(path2, 'r')
    evi_data = json.load(infile)

    mem_em = 0
    evi_em = 0
    bob_em = 0

    mem_token_F1 = 0
    evi_token_F1 = 0
    bob_token_F1 = 0

    mem_entail = 0
    evi_entail = 0
    bob_entail = 0

    counter = 0

    for idx, case in enumerate(query_data):

        if filter_id is not None:
            if str(case['q_id']) not in filter_id:
                continue

        counter += 1

        mem_token_F1 += f1_score(case['memory_output'], case['answer'])

        query_evi_token_F1 = f1_score(case['evidence_output'], case['answer'])
        evi_evi_token_F1 = f1_score(evi_data[idx]['evidence_output'], evi_data[idx]['answer'])
        evi_token_F1 += (query_evi_token_F1 + evi_evi_token_F1) / 2

        query_bob_token_F1 = f1_score(case['bob_output'], case['answer'])
        evi_bob_token_F1 = f1_score(evi_data[idx]['bob_output'], evi_data[idx]['answer'])
        bob_token_F1 += (query_bob_token_F1 + evi_bob_token_F1) / 2

        mem_em_correct = check_string_match(case['answer'], case['memory_output'], EM=True)
        query_evi_em_correct = check_string_match(case['answer'], case['evidence_output'], EM=True)
        evi_evi_em_correct = check_string_match(evi_data[idx]['answer'], evi_data[idx]['evidence_output'], EM=True)
        query_bob_em_correct = check_string_match(case['answer'], case['bob_output'], EM=True)
        evi_bob_em_correct = check_string_match(evi_data[idx]['answer'], evi_data[idx]['bob_output'], EM=True)

        if mem_em_correct:
            mem_em += 1
        if query_evi_em_correct and evi_evi_em_correct:
            evi_em += 1
        if query_bob_em_correct and evi_bob_em_correct:
            bob_em += 1

        mem_entail_correct = check_string_match(case['answer'], case['memory_output'], case, EM=False)
        query_evi_entail_correct = check_string_match(case['answer'], case['evidence_output'], case, EM=False)
        evi_evi_entail_correct = check_string_match(evi_data[idx]['answer'], evi_data[idx]['evidence_output'], evi_data[idx], EM=False)
        query_bob_entail_correct = check_string_match(case['answer'], case['bob_output'], case, EM=False)
        evi_bob_entail_correct = check_string_match(evi_data[idx]['answer'], evi_data[idx]['bob_output'], evi_data[idx], EM=False)

        if mem_entail_correct:
            mem_entail += 1
        if query_evi_entail_correct and evi_evi_entail_correct:
            evi_entail += 1
        if query_bob_entail_correct and evi_bob_entail_correct:
            bob_entail += 1

    print("Total: ", counter)

    print("Mem EM: ", mem_em / counter)
    print("Evi EM: ", evi_em / counter)
    print("Bob EM: ", bob_em / counter)

    print("Mem Token F1: ", mem_token_F1 / counter)
    print("Evi Token F1: ", evi_token_F1 / counter)
    print("Bob Token F1: ", bob_token_F1 / counter)

    print("Mem Entailment Accuracy: ", mem_entail / counter)
    print("Evi Entailment Accuracy: ", evi_entail / counter)
    print("Bob Entailment Accuracy: ", bob_entail / counter)


def main():
    cat1_evaluate("out/nq_cat1_counterfact_gpt4.json")
    cat1_result_analysis("out/nq_cat1_counterfact_gpt4.json")

    cat2_result_analysis("out/nq_cat1_counterfact_gpt4_evi.json", "out/nq_cat1_counterfact_gpt4_query.json")
    cat2_result_analysis("out/nq_cat1_counterfact_gpt4.json")


if __name__ == "__main__":
    main()