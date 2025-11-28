---
license: mit
task_categories:
- text-classification
- text-retrieval
- feature-extraction
language:
- en
tags:
- fairness
- bias-mitigation
- resume-matching
- educational-fairness
- counterfactual-evaluation
size_categories:
- n<1K
pretty_name: Resume-Job Fairness Evaluation Dataset (pairs_longtext)
---

# Resume-Job Fairness Evaluation Dataset (pairs_longtext)

**[English](#english) | [中文](#中文)**

---

<a name="english"></a>
## English

### Dataset Summary

This dataset contains 960 resume-job pairs designed for **fairness evaluation** in AI-powered hiring systems. Each pair includes full-text resumes and job descriptions, along with sensitive attribute labels (educational background category) to enable demographic parity and counterfactual fairness testing.

**Primary Use Case:** Evaluate bias and fairness in resume-job matching models, particularly with respect to educational prestige signals (elite vs non-elite schools).

### Dataset Structure

**File:** `pairs_longtext.csv`

**Columns:**
- `candidate_id` (string): Unique candidate identifier (C000000–C000959)
- `job_id` (string): Job identifier (12 unique jobs: job000–job011)
- `is_top_school` (int): Binary flag for elite school affiliation (0 = non-elite, 1 = elite)
- `school_category` (string): Four-category educational background label
  - **A**: Elite/top-tier universities (33.2%, n=319)
  - **B**: Mid-tier recognized institutions (15.8%, n=152)
  - **C**: General/non-elite schools (33.6%, n=323)
  - **D**: No explicit school mention or alternative credentials (17.3%, n=166)
- `resume_text` (string): Full resume text (avg. 183 words)
- `job_text` (string): Full job description (avg. 186 words)
- `resume_wc` (int): Resume word count (175–196 words)
- `job_wc` (int): Job word count (181–188 words)

### Dataset Statistics

| Metric | Value |
|--------|-------|
| Total pairs | 960 |
| Unique candidates | 960 |
| Unique jobs | 12 |
| Educational categories | 4 (A/B/C/D) |
| Avg resume length | 183 words |
| Avg job length | 186 words |

**School Category Distribution:**
- Category C (33.6%): 323 pairs
- Category A (33.2%): 319 pairs
- Category D (17.3%): 166 pairs
- Category B (15.8%): 152 pairs

### Intended Use

#### Primary Applications
1. **Fairness Auditing**: Measure demographic parity differences across educational backgrounds
2. **Counterfactual Testing**: Compare model predictions before/after removing school mentions
3. **Bias Mitigation Research**: Benchmark debiasing techniques (adversarial training, masking, reweighting)
4. **Ranking Fairness**: Evaluate whether top-K candidate lists maintain fair representation

#### Example Evaluation Metrics
- Demographic Parity Difference: `|P(ŷ=1|A) - P(ŷ=1|B)|`
- Equalized Odds: TPR/FPR gap across groups
- Exposure Bias: Over/under-representation in top-ranked results
- Counterfactual Shift: Prediction change after sensitive attribute removal

### Data Generation Process

Resumes and job descriptions are **synthetically generated** to control for:
- Skill alignment between candidates and jobs
- Educational background diversity (enforced via category quotas)
- Text length consistency (for fair embedding comparisons)
- Explicit sensitive tokens (e.g., "elite_school", "top-tier") for masking experiments

**Privacy Note:** No real candidate or employer data is included. All text is synthetic.

### Limitations

- **Synthetic Data**: May not capture real-world resume style variations
- **Binary Match Labels**: Ground truth labels not provided (users define their own match criteria)
- **Single Protected Attribute**: Only evaluates educational background; does not cover gender, race, age, etc.
- **English Only**: Limited to English-language job market context
- **Small Job Set**: 12 unique jobs may limit generalization

### Ethical Considerations

This dataset is intended **exclusively for research and auditing purposes**. It should not be used to:
- Train production hiring models without thorough bias testing
- Make real employment decisions
- Justify discrimination based on educational background

**Fairness is Multidimensional:** Low demographic parity difference does not guarantee fairness across all axes. Always evaluate multiple metrics and conduct qualitative review.

### Citation

If you use this dataset, please cite:

```bibtex
@misc{resume_job_fairness_eval,
  title={Resume-Job Fairness Evaluation Dataset},
  author={BGE-LoRA Fairness Finetuning Project},
  year={2024},
  url={https://github.com/Edwarddev0723/bge-lora-fairness-finetuning}
}
```

### License

MIT License - Free for research and commercial use with attribution.

### Dataset Card Contact

For questions or issues, please open an issue at the [GitHub repository](https://github.com/Edwarddev0723/bge-lora-fairness-finetuning).

---

<a name="中文"></a>
## 中文

### 資料集摘要

本資料集包含 **960 對履歷–職缺配對**，專為 AI 招聘系統的 **公平性評估** 設計。每對樣本包含完整履歷與職缺描述文本，並附帶敏感屬性標籤（教育背景類別），可用於人口統計平等與反事實公平性測試。

**主要用途：** 評估履歷–職缺匹配模型的偏差與公平性，特別針對教育聲望訊號（名校 vs 非名校）。

### 資料結構

**檔案：** `pairs_longtext.csv`

**欄位說明：**
- `candidate_id`（字串）：候選人唯一識別碼（C000000–C000959）
- `job_id`（字串）：職缺識別碼（共 12 個職缺：job000–job011）
- `is_top_school`（整數）：名校標記（0 = 非名校，1 = 名校）
- `school_category`（字串）：四類教育背景標籤
  - **A**：名校／頂尖大學（33.2%，n=319）
  - **B**：中等知名院校（15.8%，n=152）
  - **C**：一般／非名校（33.6%，n=323）
  - **D**：無明確學校提及或替代證書（17.3%，n=166）
- `resume_text`（字串）：完整履歷文本（平均 183 字）
- `job_text`（字串）：完整職缺描述（平均 186 字）
- `resume_wc`（整數）：履歷字數（175–196 字）
- `job_wc`（整數）：職缺字數（181–188 字）

### 資料統計

| 指標 | 數值 |
|------|------|
| 總配對數 | 960 |
| 唯一候選人 | 960 |
| 唯一職缺 | 12 |
| 教育類別數 | 4（A/B/C/D）|
| 平均履歷長度 | 183 字 |
| 平均職缺長度 | 186 字 |

**學校類別分佈：**
- C 類（33.6%）：323 對
- A 類（33.2%）：319 對
- D 類（17.3%）：166 對
- B 類（15.8%）：152 對

### 使用目的

#### 主要應用場景
1. **公平性審計**：衡量不同教育背景間的人口統計平等差距
2. **反事實測試**：比較移除學校資訊前後的模型預測變化
3. **去偏研究**：基準測試去偏技術（對抗訓練、遮蔽、重新加權）
4. **排序公平**：評估 Top-K 候選名單是否維持各群組公平代表性

#### 範例評估指標
- 人口統計平等差距（Demographic Parity Difference）：`|P(ŷ=1|A) - P(ŷ=1|B)|`
- 機會均衡（Equalized Odds）：各群組 TPR/FPR 差距
- 曝光偏差（Exposure Bias）：高排名結果中各群組過度／不足代表
- 反事實位移（Counterfactual Shift）：移除敏感屬性後預測變化

### 資料生成流程

履歷與職缺描述為 **合成資料**，控制以下變數：
- 候選人與職缺技能對齊
- 教育背景多樣性（透過類別配額強制平衡）
- 文本長度一致性（確保嵌入向量公平比較）
- 顯式敏感詞彙（如「名校」、「頂尖」）供遮蔽實驗使用

**隱私聲明：** 不含真實候選人或雇主資料，所有文本皆為合成。

### 限制

- **合成資料**：可能無法完全捕捉真實履歷風格變異
- **二元匹配標籤**：未提供真實標籤（使用者需自定義匹配標準）
- **單一保護屬性**：僅評估教育背景；未涵蓋性別、種族、年齡等
- **僅限英文**：侷限於英語職場情境
- **職缺樣本少**：12 個職缺可能限制泛化能力

### 倫理考量

本資料集 **僅限研究與審計用途**，不應用於：
- 未經充分偏差測試即訓練正式招聘模型
- 做出真實雇用決策
- 為基於教育背景的歧視辯護

**公平性是多維度的：** 低人口統計平等差距不代表在所有軸向上公平。務必評估多項指標並進行質性審查。

### 引用

若使用本資料集，請引用：

```bibtex
@misc{resume_job_fairness_eval,
  title={Resume-Job Fairness Evaluation Dataset},
  author={BGE-LoRA Fairness Finetuning Project},
  year={2024},
  url={https://github.com/Edwarddev0723/bge-lora-fairness-finetuning}
}
```

### 授權

MIT 授權 - 可自由用於研究與商業用途，需註明出處。

### 資料集聯絡方式

如有疑問或問題，請至 [GitHub 儲存庫](https://github.com/Edwarddev0723/bge-lora-fairness-finetuning) 開立 issue。

---

## Quick Start

### Load with Pandas

```python
import pandas as pd

df = pd.read_csv('pairs_longtext.csv')
print(df.head())
print(df['school_category'].value_counts())
```

### Load with Hugging Face Datasets

```python
from datasets import load_dataset

dataset = load_dataset('csv', data_files='pairs_longtext.csv')
print(dataset['train'][0])
```

### Fairness Evaluation Example

```python
# Calculate Demographic Parity Difference
from sklearn.metrics import confusion_matrix

# Assume you have predictions: y_pred
for category in ['A', 'B', 'C', 'D']:
    mask = df['school_category'] == category
    selection_rate = y_pred[mask].mean()
    print(f"Category {category} selection rate: {selection_rate:.3f}")

# Compute DP difference
rates = [y_pred[df['school_category'] == c].mean() for c in ['A','B','C','D']]
dp_diff = max(rates) - min(rates)
print(f"Demographic Parity Difference: {dp_diff:.3f}")
```

---

**Version:** 1.0  
**Last Updated:** 2024-11-18
