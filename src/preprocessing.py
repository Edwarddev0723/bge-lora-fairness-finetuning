"""
Preprocessing module for Fair AI Classification
Handles:
1. Gender word removal (pre-processing debiasing)
2. Balanced sampling by profession × gender
3. Text normalization
"""

import re
import json
import random
from collections import defaultdict
from typing import List, Dict, Tuple, Optional
import numpy as np
from datasets import Dataset, DatasetDict


class TextDebiaser:
    """Remove gender-indicative words from text"""
    
    def __init__(self, gender_words: List[str]):
        """
        Args:
            gender_words: List of gender-indicative words to remove
        """
        self.gender_words = set(w.lower() for w in gender_words)
        # Create regex pattern for word boundaries
        self.pattern = re.compile(
            r'\b(' + '|'.join(map(re.escape, self.gender_words)) + r')\b',
            re.IGNORECASE
        )
    
    def debias_text(self, text: str) -> str:
        """
        Remove gender words from text
        
        Args:
            text: Input biography text
            
        Returns:
            Debiased text with gender words removed
        """
        # Replace gender words with empty string
        debiased = self.pattern.sub('', text)
        # Clean up multiple spaces
        debiased = re.sub(r'\s+', ' ', debiased)
        return debiased.strip()
    
    def debias_dataset(self, dataset: Dataset, text_field: str = "hard_text") -> Dataset:
        """
        Apply debiasing to entire dataset
        
        Args:
            dataset: HuggingFace Dataset
            text_field: Name of text field to debias
            
        Returns:
            Dataset with debiased text
        """
        def debias_example(example):
            example[text_field] = self.debias_text(example[text_field])
            return example
        
        return dataset.map(debias_example, desc="Debiasing text")


class BalancedSampler:
    """
    Create balanced samples by profession × gender
    Ensures equal representation of male/female for each profession
    """
    
    def __init__(
        self,
        min_samples_per_group: int = 50,
        random_seed: int = 42
    ):
        """
        Args:
            min_samples_per_group: Minimum samples required for a group
            random_seed: Random seed for reproducibility
        """
        self.min_samples_per_group = min_samples_per_group
        self.random_seed = random_seed
        random.seed(random_seed)
        np.random.seed(random_seed)
    
    def analyze_distribution(
        self,
        dataset: Dataset,
        profession_field: str = "profession",
        gender_field: str = "gender"
    ) -> Dict:
        """
        Analyze profession × gender distribution
        
        Returns:
            Dictionary with statistics about the distribution
        """
        distribution = defaultdict(lambda: defaultdict(int))
        
        for example in dataset:
            prof = example[profession_field]
            gender = example[gender_field]
            distribution[prof][gender] += 1
        
        stats = {
            "total_professions": len(distribution),
            "profession_gender_counts": dict(distribution),
            "imbalanced_professions": []
        }
        
        # Find imbalanced professions
        for prof, gender_counts in distribution.items():
            male_count = gender_counts.get(0, 0)
            female_count = gender_counts.get(1, 0)
            ratio = min(male_count, female_count) / max(male_count, female_count) if max(male_count, female_count) > 0 else 0
            
            if ratio < 0.8:  # More than 20% imbalance
                stats["imbalanced_professions"].append({
                    "profession": prof,
                    "male_count": male_count,
                    "female_count": female_count,
                    "ratio": ratio
                })
        
        return stats
    
    def create_balanced_dataset(
        self,
        dataset: Dataset,
        profession_field: str = "profession",
        gender_field: str = "gender",
        strategy: str = "undersample"
    ) -> Dataset:
        """
        Create balanced dataset by profession × gender
        
        Args:
            dataset: Input dataset
            profession_field: Field name for profession labels
            gender_field: Field name for gender labels
            strategy: 'undersample' or 'oversample'
                - undersample: Sample min(male, female) from each
                - oversample: Sample max(male, female) from each
        
        Returns:
            Balanced dataset
        """
        # Group indices by profession × gender
        groups = defaultdict(lambda: defaultdict(list))
        
        for idx, example in enumerate(dataset):
            prof = example[profession_field]
            gender = example[gender_field]
            groups[prof][gender].append(idx)
        
        # Select balanced indices
        selected_indices = []
        
        for prof, gender_indices in groups.items():
            male_indices = gender_indices.get(0, [])
            female_indices = gender_indices.get(1, [])
            
            # Skip if either gender has too few samples
            if len(male_indices) < self.min_samples_per_group or \
               len(female_indices) < self.min_samples_per_group:
                print(f"Warning: Profession {prof} has insufficient samples "
                      f"(M:{len(male_indices)}, F:{len(female_indices)}), skipping")
                continue
            
            if strategy == "undersample":
                target_count = min(len(male_indices), len(female_indices))
            elif strategy == "oversample":
                target_count = max(len(male_indices), len(female_indices))
            else:
                raise ValueError(f"Unknown strategy: {strategy}")
            
            # Sample from each gender
            selected_male = random.sample(male_indices, min(target_count, len(male_indices)))
            selected_female = random.sample(female_indices, min(target_count, len(female_indices)))
            
            # If oversampling and need more samples, sample with replacement
            if strategy == "oversample":
                if len(selected_male) < target_count:
                    additional = random.choices(male_indices, k=target_count - len(selected_male))
                    selected_male.extend(additional)
                if len(selected_female) < target_count:
                    additional = random.choices(female_indices, k=target_count - len(selected_female))
                    selected_female.extend(additional)
            
            selected_indices.extend(selected_male)
            selected_indices.extend(selected_female)
        
        # Shuffle indices
        random.shuffle(selected_indices)
        
        # Create new dataset
        balanced_dataset = dataset.select(selected_indices)
        
        print(f"Balanced dataset created: {len(balanced_dataset)} samples "
              f"(from {len(dataset)} original samples)")
        
        return balanced_dataset


class DataPreprocessor:
    """
    Main preprocessing pipeline
    Combines debiasing and balanced sampling
    """
    
    def __init__(
        self,
        gender_words: List[str],
        remove_gender_words: bool = True,
        balance_by_gender: bool = True,
        min_samples_per_group: int = 50,
        random_seed: int = 42
    ):
        """
        Args:
            gender_words: List of gender words to remove
            remove_gender_words: Whether to remove gender words
            balance_by_gender: Whether to balance by profession × gender
            min_samples_per_group: Minimum samples per group for balancing
            random_seed: Random seed
        """
        self.remove_gender_words = remove_gender_words
        self.balance_by_gender = balance_by_gender
        
        if remove_gender_words:
            self.debiaser = TextDebiaser(gender_words)
        
        if balance_by_gender:
            self.sampler = BalancedSampler(
                min_samples_per_group=min_samples_per_group,
                random_seed=random_seed
            )
    
    def preprocess(
        self,
        dataset: Dataset,
        text_field: str = "hard_text",
        profession_field: str = "profession",
        gender_field: str = "gender",
        save_stats: bool = True,
        stats_path: Optional[str] = None
    ) -> Tuple[Dataset, Dict]:
        """
        Apply full preprocessing pipeline
        
        Args:
            dataset: Input dataset
            text_field: Field name for text
            profession_field: Field name for profession
            gender_field: Field name for gender
            save_stats: Whether to save statistics
            stats_path: Path to save statistics JSON
        
        Returns:
            Tuple of (preprocessed_dataset, statistics)
        """
        stats = {
            "original_size": len(dataset),
            "preprocessing_steps": []
        }
        
        processed_dataset = dataset
        
        # Step 1: Remove gender words
        if self.remove_gender_words:
            print("Step 1: Removing gender words...")
            processed_dataset = self.debiaser.debias_dataset(
                processed_dataset,
                text_field=text_field
            )
            stats["preprocessing_steps"].append("gender_word_removal")
        
        # Step 2: Balance by profession × gender
        if self.balance_by_gender:
            print("Step 2: Creating balanced dataset...")
            
            # Analyze before balancing
            pre_balance_stats = self.sampler.analyze_distribution(
                processed_dataset,
                profession_field=profession_field,
                gender_field=gender_field
            )
            stats["pre_balance_distribution"] = pre_balance_stats
            
            # Balance
            processed_dataset = self.sampler.create_balanced_dataset(
                processed_dataset,
                profession_field=profession_field,
                gender_field=gender_field,
                strategy="undersample"
            )
            
            # Analyze after balancing
            post_balance_stats = self.sampler.analyze_distribution(
                processed_dataset,
                profession_field=profession_field,
                gender_field=gender_field
            )
            stats["post_balance_distribution"] = post_balance_stats
            stats["preprocessing_steps"].append("balanced_sampling")
        
        stats["final_size"] = len(processed_dataset)
        stats["reduction_ratio"] = len(processed_dataset) / len(dataset)
        
        # Save statistics
        if save_stats and stats_path:
            with open(stats_path, 'w') as f:
                json.dump(stats, f, indent=2)
            print(f"Preprocessing statistics saved to {stats_path}")
        
        return processed_dataset, stats


def preprocess_pipeline(
    train_ds: Dataset,
    dev_ds: Dataset,
    test_ds: Dataset,
    config
) -> Tuple[Dataset, Dataset, Dataset, Dict]:
    """
    Apply preprocessing to train/dev/test splits
    
    Args:
        train_ds: Training dataset
        dev_ds: Development dataset
        test_ds: Test dataset
        config: Configuration object with preprocessing settings
    
    Returns:
        Tuple of (processed_train, processed_dev, processed_test, stats)
    """
    from config import GENDER_WORDS
    
    # Initialize preprocessor
    preprocessor = DataPreprocessor(
        gender_words=GENDER_WORDS,
        remove_gender_words=config.REMOVE_GENDER_WORDS,
        balance_by_gender=config.BALANCE_BY_GENDER,
        min_samples_per_group=config.MIN_SAMPLES_PER_GROUP,
        random_seed=config.RANDOM_SEED
    )
    
    # Process training set (full preprocessing)
    print("\n" + "="*80)
    print("Preprocessing TRAIN set")
    print("="*80)
    processed_train, train_stats = preprocessor.preprocess(
        train_ds,
        save_stats=True,
        stats_path=str(config.PROCESSED_DATA_DIR / "train_preprocessing_stats.json")
    )
    
    # Process dev set (only debiasing, no balancing)
    print("\n" + "="*80)
    print("Preprocessing DEV set")
    print("="*80)
    if config.REMOVE_GENDER_WORDS:
        debiaser = TextDebiaser(GENDER_WORDS)
        processed_dev = debiaser.debias_dataset(dev_ds)
    else:
        processed_dev = dev_ds
    
    # Process test set (only debiasing, no balancing)
    print("\n" + "="*80)
    print("Preprocessing TEST set")
    print("="*80)
    if config.REMOVE_GENDER_WORDS:
        debiaser = TextDebiaser(GENDER_WORDS)
        processed_test = debiaser.debias_dataset(test_ds)
    else:
        processed_test = test_ds
    
    all_stats = {
        "train": train_stats,
        "dev": {"original_size": len(dev_ds), "final_size": len(processed_dev)},
        "test": {"original_size": len(test_ds), "final_size": len(processed_test)}
    }
    
    return processed_train, processed_dev, processed_test, all_stats


if __name__ == "__main__":
    # Test preprocessing
    from datasets import load_dataset
    import sys
    sys.path.append('..')
    import config
    
    print("Loading dataset...")
    train_ds = load_dataset("LabHC/bias_in_bios", split="train[:1000]")
    
    print("\nTesting text debiasing...")
    debiaser = TextDebiaser(config.GENDER_WORDS)
    sample_text = "She is a great professor. He worked as a nurse. Mrs. Smith is an engineer."
    debiased = debiaser.debias_text(sample_text)
    print(f"Original: {sample_text}")
    print(f"Debiased: {debiased}")
    
    print("\nTesting balanced sampling...")
    sampler = BalancedSampler(min_samples_per_group=10)
    stats = sampler.analyze_distribution(train_ds)
    print(f"Distribution analysis: {json.dumps(stats, indent=2)}")
