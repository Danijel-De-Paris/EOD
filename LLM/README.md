## OK Boomer repository

How to use the demo:

1. Install the necessary dependencies:
```
!pip3 install -r requirements.txt
```
NB! Recommended Python version is 3.10.12

2. Put the indexed knowledge base and the LLM into the same directory as demo.py.

3. Parameters:
```
usage: demo.py [-h] [-on] [-g] model_path passages_path index_path

positional arguments:
  model_path     Path to model checkpoint
  passages_path  Path to knowledge base
  index_path     Path to document indices

options:
  -h, --help     show this help message and exit
  -on, --online  Set this flag if you have online access
  -g, --gpu      Set this flag if you have gpu

```

If your model path is `./model` and your indexed knowledge base (it'll be in a directory form) is `./knowledge_base` and the indices are located in `./knowledge_base/indices.faiss`, then you can run the following command:

```
!demo.py ./model ./knowledge_base ./knowledge_base/indices.faiss
```

If you are running the script from a device that has CUDA-enabled gpu, you can set the -g flag above:
```
!demo.py ./model ./knowledge_base ./knowledge_base/indices.faiss -g
```
NB! Keep in mind this only works if you have a CUDA-enabled gpu. Setting the -g option on a machine that does not have a CUDA-enabled gpu will cause the script to crash half-way.

NB! The `-on` flag is an experimental feature and may be removed in future builds.

4. The script will let you ask any questions. If you wish to end the conversation without asking any questions, type `DONE` on the console.

5. The retriever will retrieve up to 5 relevant documents related to the question asked. After each result is displayed, type `MORE` to see the next result, or type `DONE` to ask the next question or exit the script.
