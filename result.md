======================================================================
RGCN for Drug Indication Prediction using PrimeKG
======================================================================

[Step 2] Loading PrimeKG knowledge graph...
  - Total edges in PrimeKG: 8,100,498
  - Columns: ['relation', 'display_relation', 'x_index', 'x_id', 'x_type', 'x_name', 'x_source', 'y_index', 'y_id', 'y_type', 'y_name', 'y_source']

[Step 3] Filtering relevant relations...
  - Edges after filtering: 873,054

  Relation distribution:
    - drug_protein: 51,306
    - protein_protein: 642,150
    - disease_protein: 160,822
    - indication: 18,776

  Message passing relations (indication excluded):
    - drug_protein: 51,306
    - protein_protein: 642,150
    - disease_protein: 160,822

[Step 4] Creating unified node indices...
  - Total unique nodes: 31,704

  Node type distribution:
    - gene/protein: 19,093
    - drug: 6,593
    - disease: 6,018

  - Unified node count: 31,704

[Step 5] Creating edge lists and relation types...
  - Number of relation types: 3
    0: drug_protein
    1: protein_protein
    2: disease_protein

  - Edge index shape: torch.Size([2, 1708556])
  - Edge type shape: torch.Size([1708556])
  - Indication edges (for prediction): 18,776

[Step 6] Creating node features...
  - Using learnable embeddings with dimension: 64

[Step 7] Preparing data for link prediction task...
  - Number of drug nodes: 6,593
  - Number of disease nodes: 6,018
  - Positive samples (real indications): 18,776
  - Train/Val positive edges: 15,020
  - Test positive edges (held out): 3,756
  - Note: Negative samples will be generated *within each fold/epoch* to avoid leakage.

[Step 8] Defining RGCN model architecture...
  - Embedding dimension: 64
  - Hidden dimension: 64
  - Number of relations: 3
  - Number of nodes: 31,704

[Step 9] Setting up K-Fold Cross-Validation...
  - Number of folds: 5
  - Number of epochs per fold: 100
  - Learning rate: 0.01
  - Early stopping patience: 15
  - Using device: cuda

======================================================================
Starting K-Fold Cross-Validation Training
======================================================================

============================== Fold 1/5 ==============================
  Training positives: 12,016 | negatives/epoch: 120,160
  Validation positives: 3,004 | negatives: 30,040
  Epoch   1/100: Train Loss = 0.7128, Val Loss = 0.2520
  Epoch  20/100: Train Loss = 0.1275, Val Loss = 0.1084
  Epoch  40/100: Train Loss = 0.0744, Val Loss = 0.0725
  Epoch  60/100: Train Loss = 0.0574, Val Loss = 0.0662
  Epoch  80/100: Train Loss = 0.0492, Val Loss = 0.0662
  Early stopping at epoch 89

  Fold 1 Results:
    - AUC-ROC:   0.9901
    - AP:        0.9423
    - Accuracy:  0.9762
    - Precision: 0.8788
    - Recall:    0.8569
    - F1-Score:  0.8677

============================== Fold 2/5 ==============================
  Training positives: 12,016 | negatives/epoch: 120,160
  Validation positives: 3,004 | negatives: 30,040
  Epoch   1/100: Train Loss = 0.7588, Val Loss = 0.2744
  Epoch  20/100: Train Loss = 0.1208, Val Loss = 0.1156
  Epoch  40/100: Train Loss = 0.0776, Val Loss = 0.0736
  Epoch  60/100: Train Loss = 0.0604, Val Loss = 0.0635
  Epoch  80/100: Train Loss = 0.0517, Val Loss = 0.0611
  Early stopping at epoch 95

  Fold 2 Results:
    - AUC-ROC:   0.9919
    - AP:        0.9461
    - Accuracy:  0.9772
    - Precision: 0.8976
    - Recall:    0.8459
    - F1-Score:  0.8710

============================== Fold 3/5 ==============================
  Training positives: 12,016 | negatives/epoch: 120,160
  Validation positives: 3,004 | negatives: 30,040
  Epoch   1/100: Train Loss = 0.7320, Val Loss = 0.2750
  Epoch  20/100: Train Loss = 0.1413, Val Loss = 0.1310
  Epoch  40/100: Train Loss = 0.0892, Val Loss = 0.0870
  Epoch  60/100: Train Loss = 0.0617, Val Loss = 0.0675
  Epoch  80/100: Train Loss = 0.0515, Val Loss = 0.0648
  Early stopping at epoch 99

  Fold 3 Results:
    - AUC-ROC:   0.9906
    - AP:        0.9431
    - Accuracy:  0.9765
    - Precision: 0.8823
    - Recall:    0.8559
    - F1-Score:  0.8689

============================== Fold 4/5 ==============================
  Training positives: 12,016 | negatives/epoch: 120,160
  Validation positives: 3,004 | negatives: 30,040
  Epoch   1/100: Train Loss = 0.5698, Val Loss = 0.2800
  Epoch  20/100: Train Loss = 0.1365, Val Loss = 0.1156
  Epoch  40/100: Train Loss = 0.0750, Val Loss = 0.0710
  Epoch  60/100: Train Loss = 0.0598, Val Loss = 0.0628
  Epoch  80/100: Train Loss = 0.0503, Val Loss = 0.0610
  Early stopping at epoch 88

  Fold 4 Results:
    - AUC-ROC:   0.9920
    - AP:        0.9462
    - Accuracy:  0.9767
    - Precision: 0.8892
    - Recall:    0.8492
    - F1-Score:  0.8687

============================== Fold 5/5 ==============================
  Training positives: 12,016 | negatives/epoch: 120,160
  Validation positives: 3,004 | negatives: 30,040
  Epoch   1/100: Train Loss = 0.7490, Val Loss = 0.2365
  Epoch  20/100: Train Loss = 0.1373, Val Loss = 0.1320
  Epoch  40/100: Train Loss = 0.0834, Val Loss = 0.0856
  Epoch  60/100: Train Loss = 0.0599, Val Loss = 0.0699
  Epoch  80/100: Train Loss = 0.0510, Val Loss = 0.0685
  Early stopping at epoch 90

  Fold 5 Results:
    - AUC-ROC:   0.9891
    - AP:        0.9412
    - Accuracy:  0.9764
    - Precision: 0.9091
    - Recall:    0.8222
    - F1-Score:  0.8635

======================================================================
[Step 10] Evaluating Best CV Model on Independent Test Set
======================================================================
  Best model from Fold 4 (val loss = 0.0604)

  Independent Test Set Results:
    - Test positives: 3,756 | Test negatives: 37,560
    - AUC-ROC:   0.9904
    - AP:        0.9430
    - Accuracy:  0.9762
    - Precision: 0.8845
    - Recall:    0.8485
    - F1-Score:  0.8662

======================================================================
Cross-Validation Results Summary
======================================================================

--------------------------------------------------
Metric                Mean        Std
--------------------------------------------------
AUC                 0.9907     0.0011
AP                  0.9438     0.0020
ACCURACY            0.9766     0.0003
PRECISION           0.8914     0.0109
RECALL              0.8460     0.0126
F1                  0.8679     0.0025
--------------------------------------------------

Per-Fold Results:
--------------------------------------------------------------------------------
Fold          AUC         AP        Acc       Prec     Recall         F1
--------------------------------------------------------------------------------
1          0.9901     0.9423     0.9762     0.8788     0.8569     0.8677
2          0.9919     0.9461     0.9772     0.8976     0.8459     0.8710
3          0.9906     0.9431     0.9765     0.8823     0.8559     0.8689
4          0.9920     0.9462     0.9767     0.8892     0.8492     0.8687
5          0.9891     0.9412     0.9764     0.9091     0.8222     0.8635
--------------------------------------------------------------------------------

[Step 12] Generating visualizations...
  - Results plot saved to 'rgcn_results.png'

======================================================================
Dataset Statistics Summary
======================================================================

1. Graph Structure:
   - Total nodes: 31,704
   - Total edges: 1,708,556
   - Relation types: 3

2. Node Types:
   - Drug nodes: 6,593
   - Disease nodes: 6,018

3. Prediction Task:
   - Total positive samples: 18,776
   - Train/Val positives: 15,020
   - Test positives (held out): 3,756
   - Negative sampling (train): 10 per positive (re-sampled each epoch)
   - Negative sampling (val):   10 per positive
   - Negative sampling (test):  10 per positive

4. Model Configuration:
   - Node embedding dim: 64
   - Hidden dim: 64
   - Number of RGCN layers: 2
   - Learning rate: 0.01

5. Cross-Validation Performance (Mean ± Std):
   - AUC-ROC: 0.9907 ± 0.0011
   - Average Precision: 0.9438 ± 0.0020
   - Accuracy: 0.9766 ± 0.0003
   - F1 Score: 0.8679 ± 0.0025

6. Independent Test Set Performance:
   - AUC-ROC: 0.9904
   - Average Precision: 0.9430
   - Accuracy: 0.9762
   - Precision: 0.8845
   - Recall: 0.8485
   - F1 Score: 0.8662

======================================================================
Training Complete!
======================================================================