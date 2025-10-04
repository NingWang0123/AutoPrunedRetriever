from AutoPrunedRetriever import get_code_book,merging_codebook
from langchain_community.embeddings import HuggingFaceEmbeddings

prompt1 = 'dog likes human and human likes dogs'

fact_cb_str = get_code_book(
    prompt1,
    type='facts',
    rule="Store factual statements.",
    batch_size=1,
)



prompt2 = ['dog likes human','human likes dogs']



fact_cb_lst = get_code_book(
    prompt2,
    type='facts',
    rule="Store factual statements.",
    batch_size=1,
)

ini_meta = {}


word_emb = HuggingFaceEmbeddings(
    model_name="BAAI/bge-large-en-v1.5"
)

fbs = [fact_cb_str,fact_cb_lst]

for cb in fbs:
    if cb:
        ini_meta = merging_codebook(
                                ini_meta, cb, 'facts', word_emb, False
                            )
        
        print(len(ini_meta['facts_lst']))


# print('fact_cb_str',fact_cb_str)

# print('fact_cb_lst',fact_cb_lst)




# python test_file_codebook.py