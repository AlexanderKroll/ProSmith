from transformers import AutoTokenizer, AutoModelForMaskedLM
import pickle
from os.path import join
import numpy as np
from util_embeddings import create_empty_path



SMILES_BERT = "DeepChem/ChemBERTa-77M-MTR"
smiles_reprs = {}
smiles_tokenizer = AutoTokenizer.from_pretrained(SMILES_BERT)
smiles_bert = AutoModelForMaskedLM.from_pretrained(SMILES_BERT)


def calculate_smiles_embeddings(all_smiles, outpath, no_of_embeddings = 1000):
	create_empty_path(join(outpath, "SMILES"))

	n = len(all_smiles)
	parts = int(np.ceil(n/no_of_embeddings))

	for part in range(parts):
	    smiles_reprs = {}
	    smiles_list = all_smiles[part*no_of_embeddings: (part+1)*no_of_embeddings]

	    for k, smiles in enumerate(smiles_list):
	        smiles_rep = get_last_layer_repr(smiles)
	        #smiles_rep.requires_grad = False
	        smiles_reprs[smiles] = smiles_rep
	    
	    with open(join(outpath, "SMILES", "SMILES_repr_" + str(part)+".pkl"), 'wb') as handle:
	        pickle.dump(smiles_reprs, handle, protocol=pickle.HIGHEST_PROTOCOL)



def get_last_layer_repr(smiles):
    tokenizer = smiles_tokenizer
    model = smiles_bert
    key = "logits"
    max_length = 256

    tokens = tokenizer(
            smiles, 
            max_length=500, 
            padding=True, 
            truncation=True, 
            return_tensors="pt")
    
    tokens["input_ids"] = tokens["input_ids"]
    tokens["attention_mask"] = tokens["attention_mask"]
    last_layer_repr = model(**tokens)[key]
    return last_layer_repr