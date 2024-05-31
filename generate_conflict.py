import anthropic
import json
import openai
import time

import transformers
import torch
from data_processor import remove_duplicates


PROMPT_DICT = {
    "prompt_input": (
        "A question and its correct answer is below. Generate a wrong answer to the question that is different from the correct answer. Make sure the wrong answer is short, and has the same type as the correct answer.\n\n"
        "Question:\n{instruction}\n\nAnswer:\n{answer}\n\nWrong Answer:"
    ),
    "prompt_rewrite_claude": (
        "A passage and a text span inside the passage is shown below. Replace the text span with the new span and rewrite the passage so that the context is coherent to the new span.\n\n"
        "Passage:\n{evidence}\n\nSpan:\n{answer}\n\nNew Span:\n{fake_answer}\n\nNew Passage:"
    ),
    "prompt_rewrite_chatgpt": (
        "A passage and a text span inside the passage is shown below. Rewrite the passage to replace all the occurrences of the text span with the new span.\n\n"
        "Passage:\n{evidence}\n\nText Span:\n{answer}\n\nNew Span:\n{fake_answer}\n\nNew Passage:"
    )
}


openai.api_key = "OPENAI_KEY"


def generate_fake_answer(alpaca=False):

    infile = open('out/cat1_by_chatgpt.json', 'r')
    test_data = json.load(infile)

    prompt_input = PROMPT_DICT["prompt_input"]

    for idx, case in enumerate(test_data):

        if 'fake_answer' in case:
            continue

        if 'duplicate' in case and case['duplicate']:
            continue
        
        if case['status'] not in ['mem_wrong_evi_correct', 'mem_correct_evi_correct']:
            continue

        prompts = []
        prompts.append(prompt_input.format_map({
            "instruction": case['question'],
            "answer": case['answer']
        }))

        responses = []
        try:
            for p in prompts:
                completion = openai.ChatCompletion.create(
                            model="gpt-3.5-turbo",
                            messages=[
                                {"role": "system", "content": "You are a helpful assistant."},
                                {"role": "user", "content": p}
                            ],
                            temperature=1,
                            max_tokens=100
                            )
                responses.append(completion.choices[0].message['content'])
            case['fake_answer'] = responses[0]
        except Exception as e:
            print(e)

        if idx % 100 == 0:
            outfile = open('out/cat1_by_chatgpt_counterfact.json', 'w')
            json.dump(test_data, outfile, indent=4)
    outfile = open('out/cat1_by_chatgpt_counterfact.json', 'w')
    json.dump(test_data, outfile, indent=4)


def clean_answer(ans):
    if '\n\n' in ans:
        ans = ans.split('\n\n')[-1]
    if 'Wrong Answer:' in ans:
        ans = ans.split('Wrong Answer:')[-1]
    if 'New Passage:' in ans:
        ans = ans.split('New Passage:')[-1]
    if ans[-1] == '.':
        ans = ans[0:-1]
    ans = ans.strip()
    return ans

def rewrite_context(path):
    infile = open(path, 'r')
    data = json.load(infile)
    data = remove_duplicates(data)

    prompt_rewrite = PROMPT_DICT["prompt_rewrite_chatgpt"]

    new_data = []
    for idx, case in enumerate(data):
        if len(case['evidence']) == 0 or len(case['evidence'].split(' ')) < 10:
            continue
        if 'fake_answer' not in case:
            continue

        if 'duplicate' in case and case['duplicate']:
            continue

        print(idx, len(data))
        fake_answer = clean_answer(case['fake_answer'].strip())
        
        prompts = []        
        prompts.append(prompt_rewrite.format_map({
            "evidence": case['evidence'],
            "answer": case['answer'],
            "fake_answer": fake_answer
        }))

        responses = []
        try:
            for p in prompts:
                completion = openai.ChatCompletion.create(
                                model="gpt-3.5-turbo",
                                messages=[
                                    {"role": "system", "content": "You are a helpful assistant."},
                                    {"role": "user", "content": p}
                                ],
                                temperature=1,
                                max_tokens=3000
                                )

                responses.append(completion.choices[0].message['content'])
            new_case = {
                'q_id': case['q_id'],
                'question': case['question'],
                'answer': fake_answer,
                'evidence': clean_answer(responses[0]),
                'source': case['source'],
            }
            new_data.append(new_case)
            time.sleep(1)
        except Exception as e:
            print(e)
            print("ERROR AT ", idx)

        if idx % 100 == 0:
            outfile = open(path.replace(".json", "_rewrite.json"), 'w')
            json.dump(new_data, outfile, indent=4)
    outfile = open(path.replace(".json", "_rewrite.json"), 'w')
    json.dump(new_data, outfile, indent=4)
    print(len(new_data))


def main():
    generate_fake_answer(alpaca=False)
    rewrite_context('out/cat1_by_chatgpt_counterfact.json')


if __name__ == "__main__":
    main()