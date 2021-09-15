# AI4ACP
AI4ACP is a sequence-based anticancer peptides (ACPs) predictor based on the combination of the [PC6 protein-encoding method](https://github.com/LinTzuTang/PC6-protein-encoding-method "link") and deep learning model.

AI4ACP (web-server) is freely accessible at <https://axp.iis.sinica.edu.tw/AI4ACP/>

### 1. Quick demo of AI4ACP model
For qick demo of our model, run the command below:

``` bash
bash AI4ACP/test/example.sh
```
  * The input file of this demo is a FASTA file (__`example_seq.fasta`__) with 10 peptide sequences.
  
  * The output file of this demo is a CSV file (__`test/example_output.csv`__), which is composed of the sequence identities, the prediction scores, and the prediction results.
  ![image](https://user-images.githubusercontent.com/68101604/133358145-81aab6b1-de06-4675-acd9-68736eb35aa2.png)

  

### 2. Common usage of AI4ACP
  1. Make sure your working directory access to __`code/AI4ACP_predictor.py`__
  2. excute command like the example
  ``` bash
  python AI4ACP_predictor.py -f [input.fasta] -o [output.csv]
  ```
  > -f: input peptides in FASTA format
  > 
  > -o: output results in CSV

## AI4ACP deep neural network model architecture
The figure below shows the model architechure of AI4ACP. After PC6 encoding, petide sequences would pass through three convolution blocks, which are composed of convolution layers, batch normalization, max pooling, and dropout layers, and two dense layers.
![Figure5_model arch](https://user-images.githubusercontent.com/68101604/133357566-1a2d9874-6b9a-4f27-88b1-f01df278d0f2.jpg)
