# Query reformulation for VQA
Query reformulation based on hyponym-hypernym relations.

## Installation 

### Requirement 
Python 3.6  
Pytorch 1.4  

```bash
conda create -n qr_vqa python=3.6
conda activate qr_vqa 
conda install pytorch==1.4.0 torchvision==0.5.0 cudatoolkit=10.1 -c pytorch
```

## Files

### HypoSelector
To extract the correct hyponym needed for replacement.

### SpanSelector
To extract the span/ hyponym to be replaced.

### data
For preprocessing of data

### prepare_dataset
Scripts for preparing data

## Training 
```bash
python run.py
```
## Test
### hyponym prediction
```bash
python test_hypo_scorer.py
```

### span prediction
```bash 
python test_span_scorer.py
```
