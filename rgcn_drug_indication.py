# =============================================================================
# Step 1: Import all required libraries
# =============================================================================
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv
from torch_geometric.data import Data
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import (
    roc_auc_score, 
    average_precision_score, 
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
import matplotlib.pyplot as plt
import warnings
import copy
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)

print("=" * 70)
print("RGCN for Drug Indication Prediction using PrimeKG")
print("=" * 70)

# =============================================================================
# Step 2: Load and preprocess the PrimeKG dataset
# =============================================================================
print("\n[Step 2] Loading PrimeKG knowledge graph...")

# Load the knowledge graph CSV file
# The kg.csv file contains all relations in PrimeKG
kg_path = "kg.csv"
df = pd.read_csv(kg_path, low_memory=False)

print(f"  - Total edges in PrimeKG: {len(df):,}")
print(f"  - Columns: {list(df.columns)}")

# =============================================================================
# Step 3: Filter relevant relations
# =============================================================================
print("\n[Step 3] Filtering relevant relations...")

# We only keep these relation types:
# 1. drug_protein: drug-gene interactions (drug targets proteins/genes)
# 2. protein_protein: gene-gene interactions (protein-protein interactions)
# 3. disease_protein: gene-disease associations
# 4. indication: drug-disease treatment relations (our prediction target!)

# Define relation types to include in the dataset
relation_types = [
    'drug_protein',      # Drug-Gene relations
    'protein_protein',   # Gene-Gene relations
    'disease_protein',   # Gene-Disease relations
    'indication'         # Drug-Disease relations (target for prediction)
]

# Define relation types used for message passing (exclude target)
message_passing_relations = [
    'drug_protein',
    'protein_protein',
    'disease_protein'
]

# Filter the dataframe to only include these relations
df_filtered = df[df['relation'].isin(relation_types)].copy()
print(f"  - Edges after filtering: {len(df_filtered):,}")

# Show relation distribution
print("\n  Relation distribution:")
for rel in relation_types:
    count = len(df_filtered[df_filtered['relation'] == rel])
    print(f"    - {rel}: {count:,}")

print("\n  Message passing relations (indication excluded):")
for rel in message_passing_relations:
    count = len(df_filtered[df_filtered['relation'] == rel])
    print(f"    - {rel}: {count:,}")

# =============================================================================
# Step 4: Create unified node indices
# =============================================================================
print("\n[Step 4] Creating unified node indices...")

# Each node has a type (drug, gene/protein, disease) and an index
# We need to create a single unified index for all nodes

# Extract all unique nodes with their types
# Node format: (node_index, node_type)
nodes_x = df_filtered[['x_index', 'x_type']].drop_duplicates()
nodes_x.columns = ['index', 'type']
nodes_y = df_filtered[['y_index', 'y_type']].drop_duplicates()
nodes_y.columns = ['index', 'type']

# Combine all nodes
all_nodes = pd.concat([nodes_x, nodes_y]).drop_duplicates()
print(f"  - Total unique nodes: {len(all_nodes):,}")

# Count nodes by type
print("\n  Node type distribution:")
for node_type in all_nodes['type'].unique():
    count = len(all_nodes[all_nodes['type'] == node_type])
    print(f"    - {node_type}: {count:,}")

# Create a mapping from (original_index, type) to new unified index
# This is necessary because different node types may have overlapping indices
node_to_idx = {}
idx = 0
for _, row in all_nodes.iterrows():
    key = (row['index'], row['type'])
    if key not in node_to_idx:
        node_to_idx[key] = idx
        idx += 1

num_nodes = len(node_to_idx)
print(f"\n  - Unified node count: {num_nodes:,}")

# =============================================================================
# Step 5: Create edge lists and edge types
# =============================================================================
print("\n[Step 5] Creating edge lists and relation types...")

# Create mapping for message passing relation types to indices
relation_to_idx = {rel: i for i, rel in enumerate(message_passing_relations)}
print(f"  - Number of relation types: {len(relation_to_idx)}")
for rel, idx in relation_to_idx.items():
    print(f"    {idx}: {rel}")

# Lists to store edges
edge_index_list = []  # [source, target] pairs
edge_type_list = []   # relation type for each edge

# Separate indication edges for our prediction task
indication_edges = []

# Keep track of indication edges separately (prediction target)
for _, row in df_filtered[df_filtered['relation'] == 'indication'].iterrows():
    src_key = (row['x_index'], row['x_type'])
    tgt_key = (row['y_index'], row['y_type'])
    src_idx = node_to_idx[src_key]
    tgt_idx = node_to_idx[tgt_key]
    indication_edges.append((src_idx, tgt_idx))

# Process message passing edges only (exclude indication)
df_mp = df_filtered[df_filtered['relation'].isin(message_passing_relations)]
for _, row in df_mp.iterrows():
    # Get source and target node unified indices
    src_key = (row['x_index'], row['x_type'])
    tgt_key = (row['y_index'], row['y_type'])
    
    src_idx = node_to_idx[src_key]
    tgt_idx = node_to_idx[tgt_key]
    
    # Get relation type index
    rel_idx = relation_to_idx[row['relation']]
    
    # Store the edge
    edge_index_list.append([src_idx, tgt_idx])
    edge_type_list.append(rel_idx)
    
    # Also add reverse edge (undirected graph)
    edge_index_list.append([tgt_idx, src_idx])
    edge_type_list.append(rel_idx)

# Convert to tensors
edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()
edge_type = torch.tensor(edge_type_list, dtype=torch.long)

print(f"\n  - Edge index shape: {edge_index.shape}")
print(f"  - Edge type shape: {edge_type.shape}")
print(f"  - Indication edges (for prediction): {len(indication_edges):,}")

# =============================================================================
# Step 6: Create node features
# =============================================================================
print("\n[Step 6] Creating node features...")

# For simplicity, we use learnable node embeddings
# In practice, you could use pre-trained embeddings or molecular features
# Here we initialize with random features that will be learned during training

embedding_dim = 64  # Dimension of node embeddings
print(f"  - Using learnable embeddings with dimension: {embedding_dim}")

# =============================================================================
# Step 7: Prepare data for link prediction
# =============================================================================
print("\n[Step 7] Preparing data for link prediction task...")

# Our task: Predict drug-disease indication relationships
# Here we only use positive samples (real indication edges)

# Get all drug nodes and disease nodes
drug_nodes = set()
disease_nodes = set()

for _, row in df_filtered.iterrows():
    if row['x_type'] == 'drug':
        drug_nodes.add(node_to_idx[(row['x_index'], row['x_type'])])
    if row['y_type'] == 'drug':
        drug_nodes.add(node_to_idx[(row['y_index'], row['y_type'])])
    if row['x_type'] == 'disease':
        disease_nodes.add(node_to_idx[(row['x_index'], row['x_type'])])
    if row['y_type'] == 'disease':
        disease_nodes.add(node_to_idx[(row['y_index'], row['y_type'])])

drug_nodes = list(drug_nodes)
disease_nodes = list(disease_nodes)

print(f"  - Number of drug nodes: {len(drug_nodes):,}")
print(f"  - Number of disease nodes: {len(disease_nodes):,}")

# Create positive samples (existing indication edges)
positive_edges = list(set(indication_edges))
pos_edges = np.array(positive_edges, dtype=np.int64)
print(f"  - Positive samples (real indications): {len(pos_edges):,}")

# Hold out 20% as independent test set BEFORE cross-validation
train_val_edges, test_edges = train_test_split(
    pos_edges, test_size=0.2, random_state=SEED
)
test_pos_set = set(map(tuple, test_edges.tolist()))
print(f"  - Train/Val positive edges: {len(train_val_edges):,}")
print(f"  - Test positive edges (held out): {len(test_edges):,}")
print("  - Note: Negative samples will be generated *within each fold/epoch* to avoid leakage.")


# -----------------------------------------------------------------------------
# Helper: Negative sampling (fold-local)
# -----------------------------------------------------------------------------
def sample_negatives(drug_nodes, disease_nodes, avoid_set, n_samples, rng, max_tries=5_000_000):
    """
    Sample negative (drug, disease) pairs.

    Important:
    - 'avoid_set' is the set of edges that must NOT be sampled.
      For training, pass only train_pos to avoid leakage from val/test.
      For validation, you may pass train_pos ∪ val_pos for cleaner evaluation.
    """
    drug_nodes = np.asarray(drug_nodes, dtype=np.int64)
    disease_nodes = np.asarray(disease_nodes, dtype=np.int64)

    if n_samples <= 0:
        return np.empty((0, 2), dtype=np.int64)

    neg = []
    tries = 0
    while len(neg) < n_samples:
        tries += 1
        if tries > max_tries:
            raise RuntimeError(
                f"Negative sampling failed (requested={n_samples}, obtained={len(neg)}). "
                "Try lowering NEG_RATIO_TRAIN/NEG_RATIO_VAL or expanding candidate nodes."
            )

        d = int(drug_nodes[rng.integers(0, len(drug_nodes))])
        dis = int(disease_nodes[rng.integers(0, len(disease_nodes))])
        e = (d, dis)

        if e in avoid_set:
            continue

        avoid_set.add(e)  # prevent duplicates
        neg.append(e)

    return np.array(neg, dtype=np.int64)

# =============================================================================
# Step 8: Define the RGCN Model
# =============================================================================
print("\n[Step 8] Defining RGCN model architecture...")


class RGCNLinkPredictor(nn.Module):
    """
    RGCN-based model for link prediction.
    
    Architecture:
    1. Node Embedding Layer: Converts node indices to dense vectors
    2. RGCN Layer 1: First relational graph convolution
    3. RGCN Layer 2: Second relational graph convolution
    4. Link Predictor: MLP that scores node pairs
    
    The model learns node representations that capture the graph structure,
    then uses these representations to predict if a link exists between nodes.
    """
    
    def __init__(self, num_nodes, num_relations, embedding_dim=64, hidden_dim=64):
        """
        Initialize the RGCN model.
        
        Args:
            num_nodes: Total number of nodes in the graph
            num_relations: Number of different relation types
            embedding_dim: Dimension of node embeddings
            hidden_dim: Dimension of hidden layers
        """
        super(RGCNLinkPredictor, self).__init__()
        
        # Node embedding layer - learns a vector for each node
        self.node_embedding = nn.Embedding(num_nodes, embedding_dim)
        
        # First RGCN layer
        # Transforms embedding_dim -> hidden_dim
        # num_bases is for regularization (decomposition of weight matrices)
        self.conv1 = RGCNConv(
            in_channels=embedding_dim,
            out_channels=hidden_dim,
            num_relations=num_relations,
            num_bases=min(num_relations, 4)  # Use basis decomposition
        )
        
        # Second RGCN layer
        # Transforms hidden_dim -> hidden_dim
        self.conv2 = RGCNConv(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            num_relations=num_relations,
            num_bases=min(num_relations, 4)
        )
        
        # Link prediction MLP
        # Takes concatenated embeddings of two nodes and predicts link probability
        self.link_predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 1)
        )
        
    def encode(self, edge_index, edge_type):
        """
        Encode all nodes using RGCN layers.
        
        Args:
            edge_index: Graph connectivity [2, num_edges]
            edge_type: Relation type for each edge [num_edges]
            
        Returns:
            Node embeddings after RGCN [num_nodes, hidden_dim]
        """
        # Get initial node embeddings
        x = self.node_embedding.weight
        
        # First RGCN layer with ReLU activation
        x = self.conv1(x, edge_index, edge_type)
        x = F.relu(x)
        x = F.dropout(x, p=0.3, training=self.training)
        
        # Second RGCN layer
        x = self.conv2(x, edge_index, edge_type)
        
        return x
    
    def decode(self, z, edge_pairs):
        """
        Predict link probability for given node pairs.
        
        Args:
            z: Node embeddings [num_nodes, hidden_dim]
            edge_pairs: Pairs of nodes to predict [num_pairs, 2]
            
        Returns:
            Link probabilities [num_pairs]
        """
        # Get embeddings for source and target nodes
        src = z[edge_pairs[:, 0]]
        dst = z[edge_pairs[:, 1]]
        
        # Concatenate source and target embeddings
        edge_features = torch.cat([src, dst], dim=1)
        
        # Predict link probability
        logits = self.link_predictor(edge_features).squeeze()
        
        return logits
    
    def forward(self, edge_index, edge_type, edge_pairs):
        """
        Full forward pass: encode nodes and predict links.
        
        Args:
            edge_index: Graph connectivity
            edge_type: Edge relation types
            edge_pairs: Node pairs to predict
            
        Returns:
            Link prediction logits
        """
        # Encode all nodes
        z = self.encode(edge_index, edge_type)
        
        # Predict links
        logits = self.decode(z, edge_pairs)
        
        return logits


# Print model architecture
print(f"  - Embedding dimension: {embedding_dim}")
print(f"  - Hidden dimension: {embedding_dim}")
print(f"  - Number of relations: {len(message_passing_relations)}")
print(f"  - Number of nodes: {num_nodes:,}")

# =============================================================================
# Step 9: K-Fold Cross-Validation Training
# =============================================================================
print("\n[Step 9] Setting up K-Fold Cross-Validation...")

# Hyperparameters
K_FOLDS = 5
NUM_EPOCHS = 100
LEARNING_RATE = 0.01
HIDDEN_DIM = 64
PATIENCE = 15  # Early stopping patience

# Negative sampling ratios (negatives per positive)
NEG_RATIO_TRAIN = 10
NEG_RATIO_VAL = 10
NEG_RATIO_TEST = 10

print(f"  - Number of folds: {K_FOLDS}")
print(f"  - Number of epochs per fold: {NUM_EPOCHS}")
print(f"  - Learning rate: {LEARNING_RATE}")
print(f"  - Early stopping patience: {PATIENCE}")

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"  - Using device: {device}")

# Move graph data to device
edge_index = edge_index.to(device)
edge_type = edge_type.to(device)

# Initialize K-Fold
kfold = KFold(n_splits=K_FOLDS, shuffle=True, random_state=SEED)

# Storage for results
all_fold_results = []
all_train_losses = []
all_val_losses = []

# Track the best model across all folds (lowest val loss)
global_best_val_loss = float('inf')
global_best_model_state = None
global_best_fold = -1

print("\n" + "=" * 70)
print("Starting K-Fold Cross-Validation Training")
print("=" * 70)

# K-Fold Cross-Validation Loop
for fold, (train_idx, val_idx) in enumerate(kfold.split(train_val_edges)):
    print(f"\n{'='*30} Fold {fold + 1}/{K_FOLDS} {'='*30}")
    
    # Split positive edges into train/val
    train_pos = train_val_edges[train_idx]
    val_pos = train_val_edges[val_idx]

    # Fold-local RNG (reproducible)
    rng = np.random.default_rng(SEED + fold)

    # Build sets for fast membership checks
    train_pos_set = set(map(tuple, train_pos.tolist()))
    val_pos_set = set(map(tuple, val_pos.tolist()))
    all_pos_set = train_pos_set | val_pos_set | test_pos_set

    # Validation negatives: sample once per fold (fixed for consistent evaluation)
    # Exclude ALL known positives (train + val + test) to ensure true negatives
    val_neg = sample_negatives(
        drug_nodes, disease_nodes,
        avoid_set=set(all_pos_set),
        n_samples=len(val_pos) * NEG_RATIO_VAL,
        rng=rng
    )

    # Assemble validation dataset (fixed across epochs)
    val_edges_np = np.vstack([val_pos, val_neg])
    val_labels_np = np.concatenate([
        np.ones(len(val_pos), dtype=np.float32),
        np.zeros(len(val_neg), dtype=np.float32)
    ])
    perm_va = rng.permutation(len(val_edges_np))
    val_edges_np = val_edges_np[perm_va]
    val_labels_np = val_labels_np[perm_va]

    val_edges = torch.tensor(val_edges_np, dtype=torch.long, device=device)
    val_labels = torch.tensor(val_labels_np, dtype=torch.float, device=device)

    print(f"  Training positives: {len(train_pos):,} | negatives/epoch: {len(train_pos) * NEG_RATIO_TRAIN:,}")
    print(f"  Validation positives: {len(val_pos):,} | negatives: {len(val_neg):,}")


    # Initialize model for this fold
    model = RGCNLinkPredictor(
        num_nodes=num_nodes,
        num_relations=len(message_passing_relations),
        embedding_dim=embedding_dim,
        hidden_dim=HIDDEN_DIM
    ).to(device)
    
    # Optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    criterion = nn.BCEWithLogitsLoss()
    
    # Training history for this fold
    fold_train_losses = []
    fold_val_losses = []
    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0
    
    # Training loop
    for epoch in range(NUM_EPOCHS):
        # ============ Re-sample training negatives each epoch ============
        rng_epoch = np.random.default_rng(SEED + fold * NUM_EPOCHS + epoch)
        train_neg = sample_negatives(
            drug_nodes, disease_nodes,
            avoid_set=set(all_pos_set),  # fresh copy; exclude ALL known positives
            n_samples=len(train_pos) * NEG_RATIO_TRAIN,
            rng=rng_epoch
        )
        train_edges_np = np.vstack([train_pos, train_neg])
        train_labels_np = np.concatenate([
            np.ones(len(train_pos), dtype=np.float32),
            np.zeros(len(train_neg), dtype=np.float32)
        ])
        perm_tr = rng_epoch.permutation(len(train_edges_np))
        train_edges_np = train_edges_np[perm_tr]
        train_labels_np = train_labels_np[perm_tr]
        train_edges = torch.tensor(train_edges_np, dtype=torch.long, device=device)
        train_labels = torch.tensor(train_labels_np, dtype=torch.float, device=device)

        # ============ Training Phase ============
        model.train()
        optimizer.zero_grad()
        
        # Forward pass
        logits = model(edge_index, edge_type, train_edges)
        
        # Calculate loss
        loss = criterion(logits, train_labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        train_loss = loss.item()
        fold_train_losses.append(train_loss)
        
        # ============ Validation Phase ============
        model.eval()
        with torch.no_grad():
            val_logits = model(edge_index, edge_type, val_edges)
            val_loss = criterion(val_logits, val_labels).item()
            fold_val_losses.append(val_loss)
        
        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Print progress every 20 epochs
        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f"  Epoch {epoch + 1:3d}/{NUM_EPOCHS}: "
                  f"Train Loss = {train_loss:.4f}, "
                  f"Val Loss = {val_loss:.4f}")
        
        # Early stopping
        if patience_counter >= PATIENCE:
            print(f"  Early stopping at epoch {epoch + 1}")
            break
    
    # Load best model
    model.load_state_dict(best_model_state)
    
    # ============ Final Evaluation ============
    model.eval()
    with torch.no_grad():
        val_logits = model(edge_index, edge_type, val_edges)
        val_probs = torch.sigmoid(val_logits).cpu().numpy()
        val_preds = (val_probs > 0.5).astype(int)
        val_true = val_labels.cpu().numpy()
    
    # Calculate metrics
    if len(np.unique(val_true)) < 2:
        # AUC/AP are undefined for single-class labels
        auc = np.nan
        ap = np.nan
    else:
        auc = roc_auc_score(val_true, val_probs)
        ap = average_precision_score(val_true, val_probs)
    acc = accuracy_score(val_true, val_preds)
    precision = precision_score(val_true, val_preds, zero_division=0)
    recall = recall_score(val_true, val_preds, zero_division=0)
    f1 = f1_score(val_true, val_preds, zero_division=0)
    
    # Store results
    fold_results = {
        'fold': fold + 1,
        'auc': auc,
        'ap': ap,
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'best_val_loss': best_val_loss
    }
    all_fold_results.append(fold_results)
    all_train_losses.append(fold_train_losses)
    all_val_losses.append(fold_val_losses)

    # Track best model across all folds
    if best_val_loss < global_best_val_loss:
        global_best_val_loss = best_val_loss
        global_best_model_state = copy.deepcopy(best_model_state)
        global_best_fold = fold + 1
    
    # Print fold results
    print(f"\n  Fold {fold + 1} Results:")
    print(f"    - AUC-ROC:   {auc:.4f}")
    print(f"    - AP:        {ap:.4f}")
    print(f"    - Accuracy:  {acc:.4f}")
    print(f"    - Precision: {precision:.4f}")
    print(f"    - Recall:    {recall:.4f}")
    print(f"    - F1-Score:  {f1:.4f}")

# =============================================================================
# Step 10: Final Evaluation on Independent Test Set
# =============================================================================
print("\n" + "=" * 70)
print("[Step 10] Evaluating Best CV Model on Independent Test Set")
print("=" * 70)

# Load the best model from cross-validation (lowest validation loss)
print(f"  Best model from Fold {global_best_fold} (val loss = {global_best_val_loss:.4f})")

final_model = RGCNLinkPredictor(
    num_nodes=num_nodes,
    num_relations=len(message_passing_relations),
    embedding_dim=embedding_dim,
    hidden_dim=HIDDEN_DIM
).to(device)
final_model.load_state_dict(global_best_model_state)

all_known_pos_set = set(map(tuple, pos_edges.tolist()))

# Generate test negatives
rng_test = np.random.default_rng(SEED + 8888)
test_neg = sample_negatives(
    drug_nodes, disease_nodes,
    avoid_set=set(all_known_pos_set),
    n_samples=len(test_edges) * NEG_RATIO_TEST,
    rng=rng_test
)

test_edges_np = np.vstack([test_edges, test_neg])
test_labels_np = np.concatenate([
    np.ones(len(test_edges), dtype=np.float32),
    np.zeros(len(test_neg), dtype=np.float32)
])
test_edge_tensor = torch.tensor(test_edges_np, dtype=torch.long, device=device)
test_label_tensor = torch.tensor(test_labels_np, dtype=torch.float, device=device)

# Evaluate on test set
final_model.eval()
with torch.no_grad():
    test_logits = final_model(edge_index, edge_type, test_edge_tensor)
    test_probs = torch.sigmoid(test_logits).cpu().numpy()
    test_preds = (test_probs > 0.5).astype(int)
    test_true = test_label_tensor.cpu().numpy()

test_auc = roc_auc_score(test_true, test_probs)
test_ap = average_precision_score(test_true, test_probs)
test_acc = accuracy_score(test_true, test_preds)
test_prec = precision_score(test_true, test_preds, zero_division=0)
test_rec = recall_score(test_true, test_preds, zero_division=0)
test_f1 = f1_score(test_true, test_preds, zero_division=0)

test_results = {
    'auc': test_auc, 'ap': test_ap, 'accuracy': test_acc,
    'precision': test_prec, 'recall': test_rec, 'f1': test_f1
}

print(f"\n  Independent Test Set Results:")
print(f"    - Test positives: {len(test_edges):,} | Test negatives: {len(test_neg):,}")
print(f"    - AUC-ROC:   {test_auc:.4f}")
print(f"    - AP:        {test_ap:.4f}")
print(f"    - Accuracy:  {test_acc:.4f}")
print(f"    - Precision: {test_prec:.4f}")
print(f"    - Recall:    {test_rec:.4f}")
print(f"    - F1-Score:  {test_f1:.4f}")

# =============================================================================
# Step 11: Summarize Results
# =============================================================================
print("\n" + "=" * 70)
print("Cross-Validation Results Summary")
print("=" * 70)

# Calculate mean and std for each metric
metrics = ['auc', 'ap', 'accuracy', 'precision', 'recall', 'f1']
summary = {}

for metric in metrics:
    values = [r[metric] for r in all_fold_results]
    summary[metric] = {
        'mean': np.nanmean(values),
        'std': np.nanstd(values),
        'values': values
    }

# Print summary table
print("\n" + "-" * 50)
print(f"{'Metric':<15} {'Mean':>10} {'Std':>10}")
print("-" * 50)
for metric in metrics:
    print(f"{metric.upper():<15} {summary[metric]['mean']:>10.4f} {summary[metric]['std']:>10.4f}")
print("-" * 50)

# Print individual fold results
print("\nPer-Fold Results:")
print("-" * 80)
print(f"{'Fold':<6} {'AUC':>10} {'AP':>10} {'Acc':>10} {'Prec':>10} {'Recall':>10} {'F1':>10}")
print("-" * 80)
for r in all_fold_results:
    print(f"{r['fold']:<6} {r['auc']:>10.4f} {r['ap']:>10.4f} {r['accuracy']:>10.4f} "
          f"{r['precision']:>10.4f} {r['recall']:>10.4f} {r['f1']:>10.4f}")
print("-" * 80)

# =============================================================================
# Step 12: Plot Results
# =============================================================================
print("\n[Step 12] Generating visualizations...")

# Create figure with multiple subplots
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('RGCN Drug Indication Prediction - K-Fold Cross-Validation Results', 
             fontsize=14, fontweight='bold')

# Plot 1: Training and Validation Loss Curves (all folds)
ax1 = axes[0, 0]
colors = plt.cm.tab10(np.linspace(0, 1, K_FOLDS))
for i in range(K_FOLDS):
    epochs = range(1, len(all_train_losses[i]) + 1)
    ax1.plot(epochs, all_train_losses[i], '--', color=colors[i], 
             alpha=0.7, label=f'Fold {i+1} Train')
    ax1.plot(epochs, all_val_losses[i], '-', color=colors[i], 
             alpha=0.7, label=f'Fold {i+1} Val')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.set_title('Training and Validation Loss by Fold')
ax1.legend(loc='upper right', fontsize=8, ncol=2)
ax1.grid(True, alpha=0.3)

# Plot 2: Bar chart of metrics across folds
ax2 = axes[0, 1]
x = np.arange(K_FOLDS)
width = 0.15
metric_names = ['AUC', 'AP', 'Accuracy', 'F1']
metric_keys = ['auc', 'ap', 'accuracy', 'f1']
colors_bar = ['#2ecc71', '#3498db', '#9b59b6', '#e74c3c']

for i, (name, key) in enumerate(zip(metric_names, metric_keys)):
    values = [r[key] for r in all_fold_results]
    ax2.bar(x + i * width, values, width, label=name, color=colors_bar[i])

ax2.set_xlabel('Fold')
ax2.set_ylabel('Score')
ax2.set_title('Performance Metrics by Fold')
ax2.set_xticks(x + width * 1.5)
ax2.set_xticklabels([f'Fold {i+1}' for i in range(K_FOLDS)])
ax2.legend()
ax2.set_ylim(0, 1.1)
ax2.grid(True, alpha=0.3, axis='y')

# Plot 3: Box plot of metrics
ax3 = axes[1, 0]
metric_data = [[r[key] for r in all_fold_results] for key in metric_keys]
bp = ax3.boxplot(metric_data, labels=metric_names, patch_artist=True)
for patch, color in zip(bp['boxes'], colors_bar):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
ax3.set_ylabel('Score')
ax3.set_title('Metric Distribution Across Folds')
ax3.grid(True, alpha=0.3, axis='y')

# Add mean values as text
for i, (name, key) in enumerate(zip(metric_names, metric_keys)):
    mean_val = summary[key]['mean']
    ax3.annotate(f'{mean_val:.3f}', 
                 xy=(i + 1, mean_val), 
                 xytext=(i + 1.3, mean_val + 0.02),
                 fontsize=9)

# Plot 4: Summary bar chart with error bars
ax4 = axes[1, 1]
x = np.arange(len(metric_names))
means = [summary[key]['mean'] for key in metric_keys]
stds = [summary[key]['std'] for key in metric_keys]

bars = ax4.bar(x, means, yerr=stds, capsize=5, color=colors_bar, alpha=0.8)
ax4.set_ylabel('Score')
ax4.set_title('Average Performance with Standard Deviation')
ax4.set_xticks(x)
ax4.set_xticklabels(metric_names)
ax4.set_ylim(0, 1.1)
ax4.grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for bar, mean, std in zip(bars, means, stds):
    ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.02,
             f'{mean:.3f}±{std:.3f}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('rgcn_results.png', dpi=150, bbox_inches='tight')
print("  - Results plot saved to 'rgcn_results.png'")
plt.show()

# =============================================================================
# Step 13: Additional Analysis - Relation Statistics
# =============================================================================
print("\n" + "=" * 70)
print("Dataset Statistics Summary")
print("=" * 70)

print(f"\n1. Graph Structure:")
print(f"   - Total nodes: {num_nodes:,}")
print(f"   - Total edges: {edge_index.shape[1]:,}")
print(f"   - Relation types: {len(message_passing_relations)}")

print(f"\n2. Node Types:")
print(f"   - Drug nodes: {len(drug_nodes):,}")
print(f"   - Disease nodes: {len(disease_nodes):,}")

print(f"\n3. Prediction Task:")
print(f"   - Total positive samples: {len(pos_edges):,}")
print(f"   - Train/Val positives: {len(train_val_edges):,}")
print(f"   - Test positives (held out): {len(test_edges):,}")
print(f"   - Negative sampling (train): {NEG_RATIO_TRAIN} per positive (re-sampled each epoch)")
print(f"   - Negative sampling (val):   {NEG_RATIO_VAL} per positive")
print(f"   - Negative sampling (test):  {NEG_RATIO_TEST} per positive")

print(f"\n4. Model Configuration:")
print(f"   - Node embedding dim: {embedding_dim}")
print(f"   - Hidden dim: {HIDDEN_DIM}")
print(f"   - Number of RGCN layers: 2")
print(f"   - Learning rate: {LEARNING_RATE}")

print(f"\n5. Cross-Validation Performance (Mean ± Std):")
print(f"   - AUC-ROC: {summary['auc']['mean']:.4f} ± {summary['auc']['std']:.4f}")
print(f"   - Average Precision: {summary['ap']['mean']:.4f} ± {summary['ap']['std']:.4f}")
print(f"   - Accuracy: {summary['accuracy']['mean']:.4f} ± {summary['accuracy']['std']:.4f}")
print(f"   - F1 Score: {summary['f1']['mean']:.4f} ± {summary['f1']['std']:.4f}")

print(f"\n6. Independent Test Set Performance:")
print(f"   - AUC-ROC: {test_results['auc']:.4f}")
print(f"   - Average Precision: {test_results['ap']:.4f}")
print(f"   - Accuracy: {test_results['accuracy']:.4f}")
print(f"   - Precision: {test_results['precision']:.4f}")
print(f"   - Recall: {test_results['recall']:.4f}")
print(f"   - F1 Score: {test_results['f1']:.4f}")

print("\n" + "=" * 70)
print("Training Complete!")
print("=" * 70)
