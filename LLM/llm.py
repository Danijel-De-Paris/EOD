# import plac
import torch
import transformers

from datasets import Features, Sequence, Value, load_dataset, load_from_disk
from transformers import (
    DPRContextEncoder,
    DPRContextEncoderTokenizerFast,
    RagRetriever,
    RagSequenceForGeneration,
    RagTokenForGeneration,
    RagTokenizer,
    AutoTokenizer,
    AutoModelForCausalLM
)

MODEL_PATH='./model'
PASSAGES_PATH='./knowledge_base' 
INDEX_PATH='./indices.faiss'



# def init():
#     """
#     init process of chat
#     """
#     chat = ""

#     init_text = "Hi, how can I help you today?"
#     process_chat(init_text, chat)



def main(
    model_path=MODEL_PATH, 
    passages_path=PASSAGES_PATH, 
    index_path=INDEX_PATH,
    online=False,
    gpu=False
):
    print("Hi, how can I help you today?")
    while True:
        input_raw = input("Type in your question. Otherwise, type 'DONE' to end the conversation.")
        if len(input_raw.lower().strip()) == 0:
            # use a sample input for the demo
            input_raw = "Why are aluminum additives used in explosives?"
        elif (input_raw.lower().strip()) == "done":
            break
        gen(model_path, passages_path, index_path, input_raw, online, gpu)
    print("If you have any questions, please re-run the task. END.")

def gen(
    model_path=MODEL_PATH,
    passages_path=PASSAGES_PATH, 
    index_path=INDEX_PATH,
    input_raw="Why are aluminum additives used in explosives?",
    online=False,
    gpu=False
):
    dataset = load_from_disk(passages_path)  # to reload the dataset
    dataset.load_faiss_index("embeddings", index_path)  # to reload the index

    if online:
        tokenizer = AutoTokenizer.from_pretrained("facebook/rag-sequence-nq")
        retriever = RagRetriever.from_pretrained(
            "facebook/rag-sequence-nq", index_name="custom", indexed_dataset=dataset
        )
        # initialize with RagRetriever to do everything in one forward call
        model = RagSequenceForGeneration.from_pretrained("facebook/rag-sequence-nq", retriever=retriever)
    else:
        tokenizer = RagTokenizer.from_pretrained(model_path)
        retriever = RagRetriever.from_pretrained(model_path, index_name="custom", indexed_dataset=dataset)
        model = RagSequenceForGeneration.from_pretrained(model_path, retriever=retriever)

    inputs = tokenizer(input_raw, return_tensors="pt")
    input_ids = inputs["input_ids"]

    # Use retriever separately
    # 1. Encode
    question_hidden_states = model.question_encoder(input_ids)[0]
    # 2. Retrieve
    docs_dict = retriever(input_ids.numpy(), question_hidden_states.detach().numpy(), return_tensors="pt")
    doc_scores = torch.bmm(
        question_hidden_states.unsqueeze(1), docs_dict["retrieved_doc_embeds"].float().transpose(1, 2)
    ).squeeze(1)

    doc_ids = docs_dict["doc_ids"]
    original_docs = [dataset[int(doc_id)] for doc_id in doc_ids[0]]

    satisfied = False
    out_of_index = False
    index = 0
    
    if gpu:
        model_paths = [
            "gradientai/Llama-3-8B-Instruct-Gradient-1048k",
            "unsloth/llama-3-8b-Instruct",
            "princeton-nlp/Llama-3-Instruct-8B-SimPO",
            "nbeerbower/llama-3-Daredevil-Mahou-8B"
        ]
        model_path = model_paths[0]
        model_path = "./EOD_train_big_sequence_nq/llama_hf"
        device="cuda"

        resp_tokenizer = AutoTokenizer.from_pretrained(model_path)
        resp_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="cuda",
        )

        while not satisfied and not out_of_index:
            context = original_docs[index]['text']
            messages = [
                {"role": "system", "content": f"You have access to the following information: {context}. Try to paraphrase that information to answer the following question from the user."},
                {"role": "user", "content": input_raw}
            ]

            resp_input_ids = resp_tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt"
            ).to(device)

            terminators = [
                resp_tokenizer.eos_token_id,
                resp_tokenizer.convert_tokens_to_ids("<|eot_id|>")
            ]

            resp_outputs = resp_model.generate(
                resp_input_ids,
                max_new_tokens=256,
                eos_token_id=terminators,
                do_sample=True,
                temperature=0.1,
                top_p=0.9,
            )
            response = resp_outputs[0][resp_input_ids.shape[-1]:]
            response_cleaned = resp_tokenizer.decode(response, skip_special_tokens=True).split(".")[0].strip()
            print(response_cleaned)
            user_check = ""
            while user_check.lower().strip() not in ["done", "more"]:
                user_check = input("Would you like to see the next answer? If yes, say 'MORE'; if not, say 'DONE'.")
                if user_check.lower().strip() == "done":
                    satisfied = True
                elif user_check.lower().strip() == "more":
                    index += 1
                    if index >= len(original_docs):
                        out_of_index = True
                else:
                    print("That's an invalid response.")
    else:
        while not satisfied and not out_of_index:
            context = original_docs[index]['text']
            title = original_docs[index]['title']
            score = doc_scores[0][index].item()

            print(f"According to {title} with a match score of {str(score)}: {context}")
            user_check = ""
            while user_check.lower().strip() not in ["done", "more"]:
                user_check = input("Would you like to see the next answer? If yes, say 'MORE'; if not, say 'DONE'.")
                if user_check.lower().strip() == "done":
                    satisfied = True
                elif user_check.lower().strip() == "more":
                    index += 1
                    if index >= len(original_docs):
                        out_of_index = True
                else:
                    print("That's an invalid response.")

    if out_of_index:
        print("Unfortunately, we couldn't find other matches to your question. Try asking again after rewording your question.")


# if __name__ == '__main__':
#     plac.call(main)