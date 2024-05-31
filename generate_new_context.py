import json
import openai

import time

import unidecode

from analysis import check_string_match

from sentence_transformers import SentenceTransformer, util
model = SentenceTransformer('all-MiniLM-L6-v2')



PROMPT_DICT = {
    "prompt_condense": (
        "Three relevant passages are shown below. Please condense the three passages into one passage. \n\n"
        "Relevant Passages:\n[1] {passage1}\n\n[2] {passage2}\n\n[3] {passage3}\n\nRelevant New Information:"
    ),
    "prompt_select": (
        "A question, the answer and a passage are shown below. Please select the sentence in the passage that support to answer the question correctly.\n\n"
        "Question:\n{question}\n\nAnswer:\n{answer}\n\nPassage:\n{evidence}\n\nSentence:"
    ),
    "prompt_merge": (
        "Two passages and a span are shown below. Please merge the two passages, and make sure to keep the span in the new passage.\n\n"
        "Passages:\n[1] {evidence}\n\n[2] {passage}\n\nSpan:\n{answer}\n\nNew Passage:"
    )
}


openai.api_key = "OPENAI_KEY"


def clean_answer(ans):
    if '\n\n' in ans:
        ans = ans.split('\n\n')[-1]
    if 'Wrong Answer:' in ans:
        ans = ans.split('Wrong Answer:')[-1]
    ans = ans.strip()
    return ans


def generate_condensed_context(inpath, outpath, use_query=False):

    infile = open(inpath, 'r')
    test_data = json.load(infile)

    retrieval_file = open(outpath, 'r')

    retrieval_data = json.load(retrieval_file)
    retrieval_results = {}
    for case in retrieval_data:

        titles = {}
        gold_evi = case['question']
        retrieval_results[unidecode.unidecode(case['question'])] = []
        evi_emb = model.encode(gold_evi)

        for ctxt in case['ctxs']:
            if ctxt['title'] not in titles:

                if check_string_match(gold_evi, ctxt['text'], hit=True):
                    continue

                ctxt_emb = model.encode(ctxt['text'])
                cos_sim = util.cos_sim(evi_emb, ctxt_emb)
                if cos_sim > 0.85:
                    continue

                titles[ctxt['title']] = True
                retrieval_results[unidecode.unidecode(case['question'])].append(ctxt)
            if len(retrieval_results[unidecode.unidecode(case['question'])]) == 3:
                break

    prompt_condense = PROMPT_DICT["prompt_condense"]
    prompt_merge = PROMPT_DICT['prompt_merge']
    prompt_select = PROMPT_DICT['prompt_select']

    new_data = []
    for idx, case in enumerate(test_data):
        if "condensed_evidence" in case and 'new_retrieval_info' in case and 'supporting_sen' in case:
            continue

        if 'duplicate' in case and case['duplicate']:
            continue

        prompts = []

        if use_query:
            q = unidecode.unidecode(case['question'])
            prompts.append(prompt_condense.format_map({
                "evidence": case['evidence'],
                "passage1": retrieval_results[q][0]['text'],
                "passage2": retrieval_results[q][1]['text'],
                "passage3": retrieval_results[q][2]['text'],
            }))
        else:
            evi = unidecode.unidecode(case['evidence'])
            prompts.append(prompt_condense.format_map({
                "evidence": case['evidence'],
                "passage1": retrieval_results[evi][0]['text'],
                "passage2": retrieval_results[evi][1]['text'],
                "passage3": retrieval_results[evi][2]['text'],
            }))

        responses = []
        patience = 100
        while patience > 0:
            try:
                for p in prompts:
                    completion = openai.ChatCompletion.create(
                                    model="gpt-3.5-turbo",
                                    messages=[
                                        {"role": "system", "content": "You are a helpful assistant."},
                                        {"role": "user", "content": p}
                                    ],
                                    temperature=1,
                                    max_tokens=1000
                                    )
                    responses.append(completion.choices[0].message['content'])

                    p_select = prompt_select.format_map({
                                "question": case['question'],
                                "answer": case['answer'],
                                "evidence": case['evidence']
                            })
                    completion = openai.ChatCompletion.create(
                                    model="gpt-3.5-turbo",
                                    messages=[
                                        {"role": "system", "content": "You are a helpful assistant."},
                                        {"role": "user", "content": p_select}
                                    ],
                                    temperature=1,
                                    max_tokens=1000
                                    )
                    responses.append(completion.choices[0].message['content'])
                    p_merge = prompt_merge.format_map({
                                "evidence": responses[1],
                                "passage": responses[0],
                                "answer": case['answer']
                            })
                    completion = openai.ChatCompletion.create(
                                    model="gpt-3.5-turbo",
                                    messages=[
                                        {"role": "system", "content": "You are a helpful assistant."},
                                        {"role": "user", "content": p_merge}
                                    ],
                                    temperature=1,
                                    max_tokens=1000
                                    )
                    responses.append(completion.choices[0].message['content'])

                new_case = {'q_id': case['q_id'], 'question': case['question'], 'answer': case['answer'],
                            'evidence': case['evidence'], 'source': case['source'], 'new_retrieval_info': responses[0],
                            'supporting_sen': responses[1], 'condensed_evidence': responses[2]}
                new_data.append(new_case)
                break

            except Exception as e:
                patience -= 1
                if patience == 0:
                    print(e)
                time.sleep(1)
                continue

        if idx % 100 == 0:
            outfile = open(outpath, 'w')
            json.dump(new_data, outfile, indent=4)

    outfile = open(outpath, 'w')
    json.dump(new_data, outfile, indent=4)


def main():
    generate_condensed_context(inpath='data/nq_cat2_question.json', outpath='data/nq_cat2_retrieve_by_query.json', use_query=True)
    generate_condensed_context(inpath='data/nq_cat2_evidence.json', outpath='data/nq_cat2_retrieve_by_evidence.json', use_query=False)



if __name__ == "__main__":
    main()