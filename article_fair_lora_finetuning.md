# 使用 LoRA、對抗去偏與多任務學習打造公平的履歷職缺匹配模型

> 本文介紹如何以 BGE-large 向量模型為基底，透過 LoRA 參數高效微調、引入對抗式與多任務公平目標、在文字層直接遮蔽敏感屬性，並結合排序與反事實（Counterfactual）公平評估，打造兼具效能與公平性的履歷匹配系統。

---
## 1. 動機 (Motivation)
履歷→職缺匹配系統在招募流程中愈來愈常見。僅看 AUC、Accuracy 等表面指標不足以評估風險：模型可能隱性利用社經訊號（例如名校聲望），加劇不公平。本專案追求：
1. 高匹配品質（AUC / PR-AUC 佳）
2. 降低不同學校群體之間的選擇率與機會差距
3. 反事實擾動（移除學校資訊）後行為穩定
4. 可解釋性：注意力 / 梯度標記歸因 + 敘事式說明

---
## 2. 問題定義 (Problem Framing)
輸入 `(resume_text, job_text)`，預測是否匹配（二分類）。公平維度：教育背景（名校 / 非名校 / 未提及）。目標：
- 最小化各群組選擇率與 TPR/FPR 差距（人口統計平等 Demographic Parity、機會均衡 Equalized Odds）
- 降低模型內部對敏感屬性可恢復性（對抗去偏）
- 在多職缺排名場景維持良好 Hit@K、NDCG

---
## 3. 資料集與切分 (Dataset & Splits)
使用 `data/processed/processed_resume_dataset_resplit/`，重新切分確保各 split 皆含正例。
```
processed_resume_dataset_resplit/
  dataset_dict.json
  train/
  validation/
  test/
```
延伸評估資料：
- `data/eval/pairs_longtext.csv`：較大型多職缺對比
- `data/eval/match_eval.json`：精選小集合（已修復格式）快速公平探測
- 反事實樣本：移除 / 替換敏感詞之履歷版本

重新切分後再建立 *平衡驗證載入器*（balanced validation loader）以穩定公平指標。

---
## 4. 公平策略總覽 (Fairness Strategy Overview)
| 元件 | 目的 | 作法 |
|------|------|------|
| 文字遮蔽 | 阻斷名校顯式資訊外洩 | Regex + 詞典替換為 `[SCHOOL]` |
| 對抗判別器 | 懲罰敏感屬性可恢復性 | Gradient Reversal + 判別頭 |
| 多任務分類器 | 輔助穩定公平嵌入 | 額外 head 預測學校類別 |
| 公平正規項 | 降低群組差距 (DP + EO) | 以選擇率 / TPR/FPR 變異數為 proxy |
| 預測差距融合 | L2 懲罰群組機率差距 | 加入總 loss |
| 溫度校正 | 提升機率門檻調參穩定性 | 驗證集一次 LBFGS 校正 |
| 視窗檢查點 | 公平門檻下選最優效能 | Epoch 後公平篩選 |

---
## 5. 模型架構 (FairLoRAModel)
基底：`BAAI/bge-large-en-v1.5`。額外：
- 動態探索 LoRA 目標層（query/key/value + dense 類型）
- 凍結前段層，僅保留最後 4 個 encoder 注意力權重 + LoRA 可訓練
- 共用（凍結）投影 + 主分類 / 對抗 / 屬性三個專用投影
- 主分類頭：履歷–職缺匹配
- 對抗判別器：多層 MLP + BN + Dropout
- 屬性分類器：做多任務輔助

精簡程式片段：
```python
self.base_model = AutoModel.from_pretrained(BASE_MODEL)
peft_config = LoraConfig(
    r=16, lora_alpha=32,
    target_modules=target_modules,
    task_type=TaskType.FEATURE_EXTRACTION,
)
self.base_model = get_peft_model(self.base_model, peft_config)
# 凍結早期層；僅最後4層注意力權重與 LoRA 參數可訓練
```
取得 CLS 嵌入後 L2 normalize，以利後續相似度與排序評估。

---
## 6. 敏感屬性文字遮蔽 (Sensitive Attribute Masking)
在公平模型前處理階段以 Regex 遮蔽學校相關詞（名校 / 排名描述）。
```python
PATTERN = re.compile(r"\b(" + "|".join(SCHOOL_DICT) + r")\b", re.IGNORECASE)

def mask_sensitive_text(txt):
    masked = PATTERN.sub('[SCHOOL]', txt)
    masked = re.sub(r'(\[SCHOOL\]\s*){2,}', '[SCHOOL] ', masked)
    return masked.strip()
```
Baseline 與 JobBERT 保持原文，作為有 / 無遮蔽之公平差異對照。

---
## 7. 訓練迴圈強化 (FairTrainer)
Loss 組成：
```
Total = CE(class)
      + fairness_lambda * (DP_loss + EO_loss)
      + multitask_lambda * CE(attribute)
      + adversarial_lambda * AdvLoss (GRL)
      + fairness_reg_lambda * gap_linear
      + gap_l2_lambda * gap^2
```
重點：
- 對抗 / 多任務 Lambda 線性暖身（避免早期崩塌）
- 對抗兩階段：先訓練判別器（detach），再 GRL 反向梯度擾動主模型
- 公平達標但效能略低時動態收斂對抗 Lambda 上限
- 溫度校正強化門檻搜尋 F1 穩定度
- 多目標早停：正規化 Loss + 公平指標自適應權重
- 檢查點雙軌：
  - `best_fairness_model.pt` 最佳公平
  - `best_util_model.pt` 公平門檻下最高 AUC
  - `window_best_model.pt` 在指定 epoch 視窗內兼顧公平 < 0.12

---
## 8. 公平評估指標 (Fairness Metrics)
`src/fairness_metrics.py` 實作：
- 人口統計平等差距 (Demographic Parity Difference)
- 機會均衡 (Equalized Odds：TPR/FPR 差距平均)
- 平等機會 (Equal Opportunity：TPR 差距)
- 預測公平 (Predictive Parity：PPV 差距)
- 分組校正 (每組 ECE + disparity)
- 綜合公平分數：|DP diff|、|EO avg diff|、|Predictive Parity diff| 之平均

```python
metrics['overall_fairness_score'] = (
    abs(metrics['demographic_parity_difference']) +
    abs(metrics['equalized_odds_avg_difference']) +
    abs(metrics['predictive_parity_difference'])
) / 3
```
Balanced Validation：以各組等量抽樣降低偏斜造成的波動。

---
## 9. 排序 / 檢索評估 (Ranking & Retrieval)
多職缺比較流程：
- 以 (履歷, 職缺) 形成配對；視需要採用遮蔽文本
- 相似度：cosine，可再 sigmoid 映射成 pseudo-prob
- 指標：Hit@K、NDCG、AUC、PR-AUC
- 排序公平：Top-K 內不同群組之選擇率

反事實檢查：移除學校資訊後重新計算分數差異，取平均衡量穩定性。

---
## 10. 可解釋性與敘事 (Explainability & Narrative)
XAI：
- 注意力熱圖（頭平均）
- 梯度 Token 歸因（針對輸入嵌入反向傳播）
- 敏感 vs 技能關鍵詞注意力比例對比
敘事生成：透過本地 LLM（Ollama）將分數 + 重要 Token 組合成 JSON 敘事說明模型決策與公平考量。

---
## 11. 整體流程 (Workflow)
1. 載入資料並進行文本敏感遮蔽（僅公平模型）
2. 初始化 `FairLoRAModel`（動態 LoRA 目標 + 層凍結）
3. 使用 `FairTrainer.train()` 進行訓練（暖身 + 溫度校正）
4. 儲存多種檢查點 (`best_model.pt`, `best_fairness_model.pt`, `best_util_model.pt`)
5. 在保留集進行公平 / 排序 / 反事實穩定性評估
6. 產生 XAI 與敘事結果供質性審查
7. 匯出 LoRA Adapter 部署推論

---
## 12. 主要成果示例 (Illustrative Results)
| 模型 | AUC | PR-AUC | 公平分數 (Balanced) | DP 差距 | EO 平均差距 |
|------|-----|--------|---------------------|---------|-------------|
| Baseline BGE LoRA | 0.978 | 0.981 | 0.21 | 0.12 | 0.10 |
| Fair LoRA (本模型) | 0.991 | 0.993 | 0.11 | 0.06 | 0.05 |
| JobBERT 參考 | 1.000 | 1.000 | 0.19 | 0.11 | 0.09 |
*上述為開發過程代表性結果（參見 `reports/metrics/`），實際可能因種子與切分變動。*

觀察：
- Fair 模型在維持效能下縮小群組差距
- 對抗 + 遮蔽組合優於單一策略
- 溫度校正提升門檻 F1 穩定度

---
## 13. 經驗摘要 (Lessons Learned)
| 經驗 | 說明 |
|------|------|
| 多目標早停 | 正規化後的綜合分數減少公平與損失尺度不一致問題 |
| 遮蔽位置 | 僅於模型管線內遮蔽，推論不依賴外部敏感標記 |
| 對抗排程 | 線性暖身避免資料不平衡時表徵崩塌 |
| 平衡驗證 | 降低少數群體樣本不足引致的指標跳動 |
| 多檢查點 | 協助利害關係人取捨合規 vs 效能 |
| 可解釋性 | 梯度歸因顯示去偏後技能權重更突出 |

---
## 14. 可重現性 (Reproducibility)
環境：
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
訓練：
```python
from src.fair_lora_model import FairLoRAModel
from src.fair_trainer import FairTrainer
model = FairLoRAModel()
trainer = FairTrainer(model, train_loader, val_loader,
                      balanced_val_loader=balanced_val_loader,
                      adversarial_lambda=0.5, fairness_lambda=0.3,
                      multitask_lambda=0.2, fairness_reg_lambda=0.05)
trainer.train(num_epochs=15, warmup_steps=500,
              window_selection_start=7, window_selection_end=12,
              window_fairness_threshold=0.12)
```
驗證：
```python
val_metrics = trainer.validate(epoch=trainer.history['best_epoch'])
print(val_metrics['demographic_parity_diff'], val_metrics['auc'])
```
匯出 LoRA：
```bash
python scripts/export_fair_lora_to_hf.py --checkpoint models/fair_adversarial/best_fairness_model.pt --out_dir models/lora_adapters/fairness_lora/
```
反事實位移：
```python

print('Mean shift after masking:', delta)
```

---
## 15. 發佈與延伸 (Publication & Extension)
發佈清單：
- 道德聲明：公平指標不保證於未觀察族群上無偏差
- Model Card：偏差、限制、建議使用情境
- 提供遮蔽 vs 未遮蔽評估腳本
後續延伸：
- 加入因果分析做敏感性檢驗
- 以凸優化替換變異損失（率約束代理）
- 群組條件溫度校正提升校正公平
- 將公平教師蒸餾至輕量模型

---
## 16. 風險與緩解 (Risks & Mitigations)
| 風險 | 緩解 |
|------|------|
| 過度遮蔽造成技能語意流失 | 維護技能白名單，不遮蔽關鍵專業詞 |
| 對抗不穩定 | 暖身 + 梯度裁剪 (1.0) |
| 公平過度擬合於單一屬性 | 週期性監控其他敏感軸（地理等） |
| 部署後校正漂移 | 定期重新溫度校正 |

---
## 17. 結論 (Conclusion)
本管線將公平視為 **一級優化目標**：結合文字遮蔽、對抗去偏與顯式群組差距正規化，在不犧牲效能下降低偏差。多檢查點策略為利害關係人提供權衡選項；可解釋層加強信任。

> 公平是持續過程——監控、衡量、迭代。

---
## 18. 附錄：核心參數 (Appendix)
摘自 `fair_lora_config.py`：
```python
LORA_R = 8
LORA_ALPHA = 16
ADVERSARIAL_LAMBDA = 0.5
FAIRNESS_LAMBDA = 0.3
MULTITASK_LAMBDA = 0.2
FAIRNESS_REG_LAMBDA = 0.05
MAX_GRAD_NORM = 1.0
WINDOW_SELECTION_START = 7
WINDOW_SELECTION_END = 12
WINDOW_FAIRNESS_THRESHOLD = 0.12
```
可調整以探索公平–效能邊界。

---
## 19. 快速 FAQ
**為何用 LoRA 而非全量微調？** 參數效率高，匯出適配器方便。
**為何用變異作為 DP/EO 損失代理？** 實作簡單、可微分、對小批次穩定。
**為何僅驗證階段平衡抽樣？** 避免過度約束訓練分佈，保持真實學習信號。
**可擴展多類敏感屬性嗎？** 可以；多任務分類器已支援多類別。

---
## 20. 後續建議 (Next Steps)
1. Clone 專案並重現訓練
2. 嘗試更新 BASE_MODEL（如 bge-m3）重新評估公平
3. 對嵌入向量使用 SHAP 探索特徵重要度
4. 在 CI 中加入公平回歸檢查（指標惡化即失敗）

祝實驗順利！
Happy experimenting!
