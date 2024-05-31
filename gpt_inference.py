import os
import openai
import json
import time
import requests

openai.api_key = "INPUT YOUR KEY"

PROMPT_DICT = {
    "prompt_input": (
        "Answer the question below, paired with a context that provides background knowledge. Only output the answer without other context words.\n\n"
        "Context:\n{input}\n\nQuestion:\n{instruction}\n\nAnswer:"
    ),
    "prompt_no_input": (
        "Answer the question below. Only output the answer without other context words.\n\n"
        "Question:\n{instruction}\n\nAnswer:"
    ),
    "prompt_bob": (
        "Instruction: read the given information and answer the corresponding question. Only output the answer without other context words.\n\n"
        "Bob said, \"{input}\"\nQ: {instruction} in Bob's opinion based on the given text?"
    )
}

few_shot_prompt_instruction_input = "Answer the question below, paired with a context that provides background knowledge. Only output the answer without other context words."
few_shot_prompt_instruction_no_input = "Answer the question below. Only output the answer without other context words."
few_shot_prompt_instruction_bob = "Instruction: read the given information and answer the corresponding question. Only output the answer without other context words."

few_shot_prompt_query_input_demo = "\n\n\nContext:\n{input}\n\nQuestion:\n{instruction}\n\nAnswer:\n{answer}"
few_shot_prompt_query_input_query = "\n\n\nContext:\n{input}\n\nQuestion:\n{instruction}\n\nAnswer:"

few_shot_prompt_query_no_input_demo = "\n\n\nQuestion:\n{instruction}\n\nAnswe r:\n{answer}"
few_shot_prompt_query_no_input_query = "\n\n\nQuestion:\n{instruction}\n\nAnswer:"

few_shot_prompt_query_bob_demo = "\n\n\nBob said, \"{input}\"\nQ: {instruction} in Bob's opinion based on the given text?\n\nAnswer:\n{answer}"
few_shot_prompt_query_bob_query = "\n\n\nBob said, \"{input}\"\nQ: {instruction} in Bob's opinion based on the given text?\n\nAnswer:"


def generate_prompt(demo, query, few_shot=True, use_condensed_evi=False):

    prompt_input, prompt_no_input = PROMPT_DICT["prompt_input"], PROMPT_DICT["prompt_no_input"]
    prompt_bob = PROMPT_DICT['prompt_bob']
    prompts = []

    if use_condensed_evi:
        evi_key = 'condensed_evidence'
    else:
        evi_key = 'evidence'
    if few_shot:
        p = ''
        p += few_shot_prompt_instruction_no_input
        for q in demo:
            p += few_shot_prompt_query_no_input_demo.format_map({
                "instruction": q['question'],
                "answer": q['answer']
            })
        p += few_shot_prompt_query_no_input_query.format_map({
                "instruction": query['question']
            })
        prompts.append(p)

        p = ''
        p += few_shot_prompt_instruction_input
        for q in demo:
            p += few_shot_prompt_query_input_demo.format_map({
                "instruction": q['question'],
                "input": q['evidence'],
                "answer": q['answer']
            })
        p += few_shot_prompt_query_input_query.format_map({
                "instruction": query['question'],
                "input": query[evi_key]
            })
        prompts.append(p)

        p = ''
        p += few_shot_prompt_instruction_bob
        for q in demo:
            p += few_shot_prompt_query_bob_demo.format_map({
                "instruction": q['question'],
                "input": q['evidence'],
                "answer": q['answer']
            })
        p += few_shot_prompt_query_bob_query.format_map({
                "instruction": query['question'],
                "input": query[evi_key]
            })
        prompts.append(p)
    else:
        prompts.append(prompt_no_input.format_map({
            "instruction": query['question']
        }))
        prompts.append(prompt_input.format_map({
            "instruction": query['question'],
            "input": query[evi_key]
        }))
        prompts.append(prompt_bob.format_map({
            "instruction": query['question'],
            "input": query[evi_key]
        }))
    return prompts



def evaluate():
    infile = open('out/nq_cat1_counterfact.json', 'r')
    test_data = json.load(infile)

    demo_infile = open('out/nq_demo.json', 'r')
    demo_data = json.load(demo_infile)

    for idx, case in enumerate(test_data):
        prompts = generate_prompt(demo_data, case, few_shot=True)
        responses = []
        patience = 100
        while patience > 0:
            try:
                for p in prompts:
                    completion = openai.ChatCompletion.create(
                                    model="gpt-4",
                                    messages=[
                                        {"role": "system", "content": "You are a helpful assistant."},
                                        {"role": "user", "content": p}
                                    ],
                                    temperature=0,
                                    max_tokens=100,
                                    top_p=1
                                    )
                    responses.append(completion.choices[0].message['content'])
                case['memory_output'] = responses[0]
                case['evidence_output'] = responses[1]
                case['bob_output'] = responses[2]
                break
            except Exception as e:
                patience -= 1
                if patience % 10 == 0:
                    print(e)
                time.sleep(1)
                continue

        if idx % 100 == 0:
            outfile = open('out/nq_cat1_counterfact_gpt4.json', 'w')
            json.dump(test_data, outfile, indent=4)

    outfile = open('out/nq_cat1_counterfact_gpt4.json', 'w')
    json.dump(test_data, outfile, indent=4)


def main():
    evaluate()


if __name__ == "__main__":
    main()