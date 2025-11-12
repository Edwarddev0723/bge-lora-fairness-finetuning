"""
Fair Data Loader for Resume-Job Matching
Handles data loading, preprocessing, and sensitive attribute masking
"""

import torch
from torch.utils.data import Dataset as TorchDataset, DataLoader, WeightedRandomSampler, BatchSampler
from datasets import load_from_disk, Dataset as HFDataset, DatasetDict
from transformers import AutoTokenizer
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from collections import Counter
import re
from typing import Dict, List, Tuple, Optional, Any, Iterable

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from fair_lora_config import *


class FairResumeDataset(TorchDataset):
    """Dataset for fair resume-job matching"""
    
    def __init__(
        self,
        data,
        tokenizer,
        max_length: int = MAX_LENGTH,
        mask_sensitive: bool = MASK_SENSITIVE_ATTRS,
        replace_with_neutral: bool = REPLACE_WITH_NEUTRAL
    ):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mask_sensitive = mask_sensitive
        self.replace_with_neutral = replace_with_neutral
        
    def __len__(self):
        return len(self.data)
    
    def _mask_school_names(self, text: str) -> str:
        """Replace school names with neutral token [SCHOOL]"""
        if not self.replace_with_neutral:
            return text
        
        # List of school keywords to replace
        school_patterns = [
            r'\b(university|college|institute|school)\b',
            r'\b(harvard|yale|princeton|stanford|mit|caltech|columbia|penn)\b',
            r'\b(berkeley|ucla|duke|cornell|dartmouth|brown)\b',
            r'\b(northwestern|johns hopkins|vanderbilt|rice)\b',
            r'\b(oxford|cambridge|imperial|lse|ucl)\b',
            r'\b(toronto|mcgill|ubc|peking|tsinghua)\b',
        ]
        
        masked_text = text
        for pattern in school_patterns:
            masked_text = re.sub(pattern, '[SCHOOL]', masked_text, flags=re.IGNORECASE)
        
        return masked_text
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Get text inputs
        resume_text = item['resume_text']
        job_desc_text = item['job_description_text']
        
        # Mask sensitive attributes if enabled
        if self.mask_sensitive:
            resume_text = self._mask_school_names(resume_text)
        
        # Combine resume and job description
        # Format: [CLS] resume [SEP] job_desc [SEP]
        combined_text = f"{resume_text} [SEP] {job_desc_text}"
        
        # Tokenize
        encoding = self.tokenizer(
            combined_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Get label robustly (supports 'fit'/'not fit', '1'/'0', booleans, etc.)
        raw_label = item['label']
        label = 0
        try:
            # Numeric-like
            if isinstance(raw_label, (int, np.integer)):
                label = int(raw_label)
            elif isinstance(raw_label, (float, np.floating)):
                label = int(raw_label > 0.5)
            elif isinstance(raw_label, (bool, np.bool_)):
                label = int(raw_label)
            elif isinstance(raw_label, (str, np.str_)):
                s = raw_label.strip().lower()
                pos_set = {"1", "true", "yes", "y", "fit", "good fit", "potential fit", "positive", "pos", "match"}
                neg_set = {"0", "false", "no", "n", "not fit", "no fit", "poor fit", "negative", "neg", "mismatch"}
                if s in pos_set:
                    label = 1
                elif s in neg_set:
                    label = 0
                else:
                    # Try integer/float cast fallback
                    try:
                        label = int(float(s))
                    except Exception:
                        label = 0
            else:
                label = 0
        except Exception:
            label = 0
        label = 1 if label >= 1 else 0
        
        # Get sensitive attributes
        is_top_school = int(item['is_top_school'])
        school_category = item['school_category']
        if school_category in SCHOOL_CATEGORIES:
            school_category_id = SCHOOL_CATEGORIES.index(school_category)
        else:
            # Fallback: map unknown to last category if exists, else 0
            school_category_id = len(SCHOOL_CATEGORIES) - 1 if len(SCHOOL_CATEGORIES) > 0 else 0
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'label': torch.tensor(label, dtype=torch.long),
            'is_top_school': torch.tensor(is_top_school, dtype=torch.long),
            'school_category': torch.tensor(school_category_id, dtype=torch.long),
            'resume_text': resume_text[:200],  # For debugging
            'job_desc_text': job_desc_text[:200]
        }


class GroupBatchSampler(BatchSampler):
    """Batch sampler ensuring each batch contains at least one minority-group sample.

    Minority groups are determined from joint (school_category, is_top_school) counts
    using a simple rule: groups with count <= median group count are considered minority.

    This sampler works without replacement over the epoch. If a batch initially lacks
    a minority example, it swaps in a minority index from the remainder. If insufficient
    minorities exist to cover all batches, some batches may still be majority-only.
    """

    def __init__(self,
                 dataset_size: int,
                 group_keys: List[Tuple[Any, Any]],
                 batch_size: int,
                 drop_last: bool = False,
                 seed: int = 42):
        self.dataset_size = int(dataset_size)
        self.group_keys = list(group_keys)
        self.batch_size = int(batch_size)
        self.drop_last = drop_last
        self.seed = int(seed)

        # Identify minority groups (<= median count)
        cnt = Counter(self.group_keys)
        counts = np.array(list(cnt.values())) if cnt else np.array([self.dataset_size])
        med = np.median(counts)
        self.minority_groups = {g for g, c in cnt.items() if c <= med}
        self.minority_indices = [i for i, g in enumerate(self.group_keys) if g in self.minority_groups]
        self.minority_set = set(self.minority_indices)

    def __len__(self) -> int:
        n = self.dataset_size // self.batch_size
        if not self.drop_last and self.dataset_size % self.batch_size != 0:
            n += 1
        return n

    def __iter__(self) -> Iterable[List[int]]:
        rng = np.random.default_rng(self.seed)
        indices = np.arange(self.dataset_size)
        rng.shuffle(indices)

        num_batches = len(self)
        for b in range(num_batches):
            start = b * self.batch_size
            end = min((b + 1) * self.batch_size, self.dataset_size)
            if start >= self.dataset_size:
                break
            batch = indices[start:end].tolist()
            # Ensure at least one minority if possible
            if not any(idx in self.minority_set for idx in batch) and len(self.minority_indices) > 0:
                # Look ahead within remainder to find a minority to swap in
                swapped = False
                for j in range(end, self.dataset_size):
                    if indices[j] in self.minority_set:
                        # swap with first position in batch
                        swap_target = start  # first element in current batch
                        indices[swap_target], indices[j] = indices[j], indices[swap_target]
                        batch[0] = indices[swap_target]
                        swapped = True
                        break
                # If none in remainder, sample a minority index (with replacement) and replace first
                if not swapped:
                    m_idx = int(rng.choice(self.minority_indices))
                    batch[0] = m_idx
            # If drop_last and final small batch
            if self.drop_last and len(batch) < self.batch_size:
                continue
            yield batch


def get_sample_weights(dataset) -> List[float]:
    """Group-aware weighted sampling.
    Weight = inverse frequency for joint group (school_category, is_top_school)
             multiplied by optional category prior from SAMPLE_WEIGHTS.
    """
    if hasattr(dataset, "__getitem__") and hasattr(dataset, "column_names"):
        school_categories = list(dataset["school_category"])  # HF Dataset column access
        is_top = list(dataset["is_top_school"])
    else:
        school_categories = [item['school_category'] for item in dataset]
        is_top = [item['is_top_school'] for item in dataset]

    # Joint group key
    joint = [(str(sc), int(it)) for sc, it in zip(school_categories, is_top)]
    joint_counts = Counter(joint)

    total = len(joint)
    weights: List[float] = []
    for (sc, it) in joint:
        base = total / max(1, joint_counts[(sc, it)])
        prior = SAMPLE_WEIGHTS.get(sc, 1.0)  # allow overriding by school_category name
        # Slight boost for top_school minority when present
        if int(it) == 1:
            prior *= 1.25
        weights.append(base * prior)
    return weights


def create_data_loaders(
    dataset_path: str = str(DATASET_PATH),
    batch_size: int = BATCH_SIZE,
    max_seq_length: int = MAX_LENGTH,
    use_reweighting: bool = USE_REWEIGHTING,
    use_group_batch_sampler: bool = False,
    create_balanced_val: bool = True,
    num_workers: int = NUM_WORKERS,
    seed: int = SEED
) -> Tuple[DataLoader, DataLoader, DataLoader, Any, Optional[DataLoader]]:
    """
    Create train, validation, and test data loaders
    
    Returns:
        train_loader, val_loader, test_loader
    """
    print(f"\n📦 Loading dataset from {dataset_path}...")
    
    # Load dataset; if re-split path missing but original exists, auto-create re-split
    ds_path = Path(dataset_path)
    if not ds_path.exists() and str(ds_path).endswith('processed_resume_dataset_resplit') and ORIGINAL_DATASET_PATH.exists():
        print("⚠️ Re-split dataset not found. Creating from original using combined stratification (label + school_category)...")
        orig = load_from_disk(str(ORIGINAL_DATASET_PATH))
        # Build DataFrame
        import pandas as pd
        all_records = []
        for split_name in ['train','test']:
            split_ds = orig[split_name]
            for i in range(len(split_ds)):
                all_records.append({
                    'resume_text': split_ds[i]['resume_text'],
                    'job_description_text': split_ds[i]['job_description_text'],
                    'label': split_ds[i]['label'],
                    'is_top_school': split_ds[i]['is_top_school'],
                    'school_category': split_ds[i]['school_category']
                })
        df_all = pd.DataFrame(all_records)
        # Label bin
        def to_bin(v):
            if isinstance(v, str):
                s = v.strip().lower()
                return 1 if s in {'fit','1','true','yes'} else 0
            try:
                return int(float(v))
            except Exception:
                return 0
        df_all['label_bin'] = df_all['label'].apply(to_bin).clip(0,1)
        df_all['strat_key'] = df_all['label_bin'].astype(str) + '_' + df_all['school_category'].astype(str)
        # Merge rare strata
        strata_counts = df_all['strat_key'].value_counts()
        rare = strata_counts[strata_counts < 2].index.tolist()
        if rare:
            df_all.loc[df_all['strat_key'].isin(rare),'strat_key'] = df_all.loc[df_all['strat_key'].isin(rare),'label_bin'].astype(str)
        from sklearn.model_selection import StratifiedShuffleSplit
        sss_outer = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=SEED)
        idx_all = df_all.index.values
        y_strat = df_all['strat_key'].to_numpy()
        train_val_idx, test_new_idx = next(sss_outer.split(idx_all, y_strat))
        df_tv = df_all.loc[train_val_idx].reset_index(drop=True)
        df_test_new = df_all.loc[test_new_idx].reset_index(drop=True)

        sss_inner = StratifiedShuffleSplit(n_splits=1, test_size=0.1111, random_state=SEED)
        idx_tv = df_tv.index.values
        y_tv = df_tv['strat_key'].to_numpy()
        train_idx_new, val_idx_new = next(sss_inner.split(idx_tv, y_tv))
        df_train_new = df_tv.loc[train_idx_new].reset_index(drop=True)
        df_val_new = df_tv.loc[val_idx_new].reset_index(drop=True)
        # Convert to HF and save
        def to_dataset(df):
            return HFDataset.from_pandas(df[['resume_text','job_description_text','label','is_top_school','school_category']])
        new_ds = DatasetDict({'train': to_dataset(df_train_new), 'validation': to_dataset(df_val_new), 'test': to_dataset(df_test_new)})
        new_ds.save_to_disk(str(ds_path))
        print(f"✅ Re-split dataset saved to {ds_path}")

    dataset = load_from_disk(str(ds_path))
    # Support validation split name
    train_data = dataset['train']
    test_data = dataset['test']
    val_data_existing = dataset.get('validation', None) if isinstance(dataset, DatasetDict) else None
    
    # Split train into train + val (stratified by school_category). If HF stratify unsupported, fallback to sklearn.
    if val_data_existing is not None:
        val_data = val_data_existing
    else:
        try:
            train_val_split = train_data.train_test_split(
                test_size=VAL_SPLIT,
                seed=seed,
                stratify_by_column='school_category'
            )
            train_data = train_val_split['train']
            val_data = train_val_split['test']
        except ValueError:
            categories = list(train_data['school_category'])
            sss = StratifiedShuffleSplit(n_splits=1, test_size=VAL_SPLIT, random_state=seed)
            indices = np.arange(len(categories))
            train_idx, val_idx = next(sss.split(np.zeros(len(categories)), categories))
            # Select from the original (pre-split) train_data
            train_data = train_data.select(train_idx.tolist())
            val_data = load_from_disk(str(ds_path))['train'].select(val_idx.tolist())
            print("⚠️ HuggingFace stratify_by_column unsupported for 'school_category'; used sklearn StratifiedShuffleSplit.")
    
    print(f"   Train: {len(train_data):,} samples")
    print(f"   Val:   {len(val_data):,} samples")
    print(f"   Test:  {len(test_data):,} samples")
    
    # Print distribution
    print(f"\n📊 School Distribution:")
    for split_name, split_data in [("Train", train_data), ("Val", val_data), ("Test", test_data)]:
        school_dist = Counter(split_data['school_category'])
        print(f"\n   {split_name}:")
        for category, count in school_dist.items():
            pct = count / len(split_data) * 100
            print(f"     {category:.<30} {count:>5} ({pct:>5.2f}%)")
    
    # Load tokenizer
    print(f"\n🔤 Loading tokenizer: {BASE_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    
    # Create datasets
    train_dataset = FairResumeDataset(train_data, tokenizer, max_length=max_seq_length)
    val_dataset = FairResumeDataset(val_data, tokenizer, max_length=max_seq_length)
    test_dataset = FairResumeDataset(test_data, tokenizer, max_length=max_seq_length)
    
    # Create samplers for balanced training
    sampler = None
    shuffle = True
    if use_group_batch_sampler:
        print(f"\n👥 Using GroupBatchSampler (ensure ≥1 minority per batch)...")
        # Build joint group keys from HF train_data
        sc_col = list(train_data['school_category'])
        ts_col = list(train_data['is_top_school'])
        group_keys = [(sc_col[i], int(ts_col[i])) for i in range(len(train_data))]
        batch_sampler = GroupBatchSampler(dataset_size=len(train_data), group_keys=group_keys, batch_size=batch_size, drop_last=False, seed=seed)
        train_loader = DataLoader(
            train_dataset,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            pin_memory=True if DEVICE.type == 'cuda' else False
        )
        shuffle = False
    elif use_reweighting:
        print(f"\n⚖️  Using weighted sampling for fairness...")
        sample_weights = get_sample_weights(train_data)
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )
        shuffle = False
    
    # Create data loaders
    if not use_group_batch_sampler:
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=sampler,
            shuffle=shuffle if sampler is None else False,
            num_workers=num_workers,
            pin_memory=True if DEVICE.type == 'cuda' else False
        )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if DEVICE.type == 'cuda' else False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if DEVICE.type == 'cuda' else False
    )

    # Balanced validation loader (downsample each joint group to min count)
    balanced_val_loader: Optional[DataLoader] = None
    if create_balanced_val:
        try:
            sc_v = list(val_data['school_category'])
            ts_v = list(val_data['is_top_school'])
            from collections import defaultdict
            groups: Dict[Tuple[Any, Any], List[int]] = defaultdict(list)
            for i in range(len(val_data)):
                groups[(sc_v[i], int(ts_v[i]))].append(i)
            if len(groups) >= 2:
                min_c = min(len(v) for v in groups.values())
                if min_c >= 1:
                    rng = np.random.default_rng(seed)
                    sel_idx = []
                    for idxs in groups.values():
                        if len(idxs) >= min_c:
                            sel = rng.choice(idxs, size=min_c, replace=False)
                        else:
                            sel = np.array(idxs, dtype=int)
                        sel_idx.extend(sel.tolist())
                    val_balanced_hf = val_data.select(sel_idx)
                    val_balanced_dataset = FairResumeDataset(val_balanced_hf, tokenizer, max_length=max_seq_length)
                    balanced_val_loader = DataLoader(
                        val_balanced_dataset,
                        batch_size=batch_size,
                        shuffle=False,
                        num_workers=num_workers,
                        pin_memory=True if DEVICE.type == 'cuda' else False
                    )
                    print(f"\n🧪 Balanced Val created (joint groups): {len(val_balanced_hf)} samples; per-group ~{min_c}")
        except Exception as e:
            print(f"⚠️ Balanced Val creation failed: {e}")
    
    print(f"\n✅ Data loaders created successfully!")
    print(f"   Train batches: {len(train_loader)}")
    print(f"   Val batches:   {len(val_loader)}")
    print(f"   Test batches:  {len(test_loader)}")
    if balanced_val_loader is not None:
        print(f"   Balanced Val batches: {len(balanced_val_loader)}")

    return train_loader, val_loader, test_loader, tokenizer, balanced_val_loader


if __name__ == "__main__":
    # Test data loading with new sampler & balanced val
    print("="*80)
    print("Testing Fair Data Loader")
    print("="*80)
    
    train_loader, val_loader, test_loader, tokenizer, balanced_val_loader = create_data_loaders(
        use_group_batch_sampler=True,
        create_balanced_val=True
    )
    
    print(f"\n🧪 Testing one train batch...")
    batch = next(iter(train_loader))
    print(f"   Batch keys: {list(batch.keys())}")
    print(f"   Input IDs shape: {batch['input_ids'].shape}")
    print(f"   School category in batch (counts):")
    sc_vals = batch['school_category'].tolist()
    sc_cnt = Counter(sc_vals)
    print(f"     {sc_cnt}")
    print(f"   is_top_school in batch (counts):")
    ts_vals = batch['is_top_school'].tolist()
    ts_cnt = Counter(ts_vals)
    print(f"     {ts_cnt}")
    if balanced_val_loader is not None:
        bal_batch = next(iter(balanced_val_loader))
        print(f"\n🧪 Testing one balanced val batch (counts):")
        bal_sc_cnt = Counter(bal_batch['school_category'].tolist())
        bal_ts_cnt = Counter(bal_batch['is_top_school'].tolist())
        print(f"   school_category: {bal_sc_cnt}")
        print(f"   is_top_school:   {bal_ts_cnt}")
    print(f"\n✅ Data loader test completed!")
