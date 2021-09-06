from tensorflow.keras.models import load_model
from Protein_Encoding import PC_6
import numpy as np

model = load_model('./ACPvenv/PC_6_model_ACP4db_final_best_weights.h5')
test = PC_6('./ACPvenv/example_seq.fasta', length=50)
array_test= np.array(list(test.values()))
labels_score = model.predict(array_test)
print("Sequence", "Score", "Prediction")
for i in range(len(labels_score)):
    if labels_score[i] > 0.472:
        print(list(test.keys())[i], labels_score[i], "YES")
    else:
        print(list(test.keys())[i], labels_score[i], "NO")