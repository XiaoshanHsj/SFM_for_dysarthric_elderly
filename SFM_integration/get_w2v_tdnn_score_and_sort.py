import sys
import jiwer

w2v_score = sys.argv[1]
tdnn_score = sys.argv[2]
w2v_weight = sys.argv[3]
tdnn_weight = sys.argv[4]
output = sys.argv[5]
text_path = sys.argv[6]

print("=====================================================")
print("w2v_weight: ", w2v_weight, " tdnn_weight: ", tdnn_weight)
with open(w2v_score, "r") as f:
    w2v_score_lines = f.readlines()
    
with open(tdnn_score, "r") as f:
    tdnn_score_lines = f.readlines()
    
w2v_score_dict = {}
tdnn_score_dict = {}
for line in w2v_score_lines:
    utt_id_index, score = line.rstrip("\n").split()
    w2v_score_dict[utt_id_index] = float(score)
    
for line in tdnn_score_lines:
    utt_id_index, _, _, score, word = line.rstrip("\n").split("\t")
    tdnn_score_dict[utt_id_index] = (float(score), word)
    
both_score_dict = {}
for k in w2v_score_dict.keys():
    k_w2v_score = w2v_score_dict[k]
    k_tdnn_score, k_word = tdnn_score_dict[k]
    utt_id = "-".join(k.split("-")[:-1])
    if both_score_dict.get(utt_id) is None:
        both_score_dict[utt_id] = [(k_w2v_score, k_tdnn_score, k_word)]
    else:
        both_score_dict[utt_id].append((k_w2v_score, k_tdnn_score, k_word))

results = ""
for k in both_score_dict:
    utt_id = k
    score_list = both_score_dict[utt_id]
    final_score = []
    word_list = []
    for k_score in score_list:
        k_w2v_score = k_score[0]
        k_tdnn_score = k_score[1]
        k_word = k_score[2]
        k_final_score = -float(w2v_weight) * k_w2v_score - float(tdnn_weight) * k_tdnn_score
        final_score.append(k_final_score)
        word_list.append(k_word)
        
    max_index = final_score.index(max(final_score))
    word = word_list[max_index]
    # result = k+"-"+str(max_index+1)+" " + word
    result = k+ " " + word
    results = results + result + "\n"
    
output_path = output + "_w2v_" + w2v_weight + "_tdnn_" + tdnn_weight
with open(output_path, "w") as f:
    f.write(results)

# compute WER
with open(text_path, "r") as f:
    text_lines = f.readlines()

text_dict = {}
for line in text_lines:
    terms = line.rstrip("\n").split()
    utt_id = terms[0]
    word = " ".join(terms[1:])
    spk = utt_id.split("-")[1]
    
    text_dict[utt_id] = word
    
with open(output_path, "r") as f:
    pred_lines = f.readlines()

def compute_measures_by_group(text_dict):
    hy = []
    pr = []
    for line in pred_lines:
        terms = line.rstrip("\n").split()
        utt_id = terms[0]
        pred_word = " ".join(terms[1:])
        if text_dict.get(utt_id) is not None:
            word = text_dict[utt_id]
            hy.append(word)
            pr.append(pred_word)
        else:
            continue

    measures = jiwer.compute_measures(hy, pr)
    return measures['wer']

overall_wer = compute_measures_by_group(text_dict)
print("overall: ", str(round(overall_wer, 4)))
