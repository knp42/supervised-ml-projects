# Unsupervised Machine Learning — Practice Projects

This repository documents my hands-on learning journey in unsupervised machine learning. Unlike supervised learning, there is no target variable (`y`) — no observation is pre-labeled. The goal is to discover structure (groups, patterns) directly from the features, and evaluation relies on internal metrics (e.g. Silhouette Score) rather than comparison to a ground truth.

## Projects

### 1. Mall Customer Segmentation (Clustering)
K-Means clustering to segment mall customers by annual income and spending score, with no target variable — the goal is to discover natural groupings rather than predict a known label.

- **Feature selection:** excluded `CustomerID` (an arbitrary identifier with no distance-based meaning — including it would actively distort cluster assignments, unlike in supervised learning where an uninformative feature is simply down-weighted during training)
- **Scaling:** `StandardScaler` applied before clustering — distance-based algorithms are even more sensitive to feature scale than supervised models, since there's no target to compensate for a dominant-scale feature
- **Choosing k:** validated with two independent, converging methods — the Elbow Method (inertia) and the Silhouette Score (cohesion vs. separation) — both pointed to **k = 5**, with a peak Silhouette Score of 0.5547
- **Cluster interpretation:** 5 clearly separated, business-interpretable segments (e.g., high income/high spending "target customers", low income/high spending "overspenders", high income/low spending "cautious affluent")
- **PCA investigation:** tested PCA (with `Age` and one-hot encoded `Genre` added) to explore dimensionality reduction for visualization purposes. Result: only ~60% of variance was preserved in 2 components — well below the typical 80–95% seen when features are correlated. A correlation matrix confirmed why: none of the four features (`Age`, `Annual Income`, `Spending Score`, `Genre`) were meaningfully correlated with each other (strongest was Age–Spending Score at only -0.33), so there was little redundancy for PCA to compress.
- **t-SNE follow-up:** since the underlying business need (visualizing all 4 features together) was still unmet, tested t-SNE as a non-linear alternative to PCA. Unlike PCA, t-SNE produced a visually well-separated 2D projection, confirming that a real non-linear group structure exists across all 4 features. However, silhouette scores computed directly on the t-SNE embedding were unstable and non-monotonic across k (no clear peak, best score at k=2 contradicting the known 5-segment structure) — a known limitation, since t-SNE deliberately distorts global inter-cluster distances to preserve local neighborhoods, making distance-based metrics unreliable on its output.
- **Final decision:** kept K-Means on the original 2 scaled features (`Annual Income`, `Spending Score`) as the production clustering, since it's the only version with a metric (Silhouette Score = 0.5547) that can be trusted at face value. t-SNE was used only as a visual sanity check confirming that a genuine cluster structure exists even when `Age` and `Genre` are included — not as a substitute for the validated clustering decision. This mirrors a common real-world pattern: a technique can be genuinely useful for one purpose (visual exploration) while being inappropriate for another (quantitative model selection), and knowing the difference matters more than defaulting to whichever number looks best.

### 2. Credit Card Fraud Detection (Anomaly Detection)
Isolation Forest trained purely on transaction features (`Time`, `Amount`, and 28 PCA-anonymized features `V1`–`V28`) to detect fraudulent transactions — the true `Class` label was withheld during training and used only afterward, to evaluate whether the anomalies found actually corresponded to real fraud.

- **Why anomaly detection instead of straight classification:** fraud patterns evolve constantly. A model trained only on past fraud examples (supervised) can miss entirely new fraud techniques it has never seen. An anomaly detector learns what "normal" looks like and flags deviations from it, which can in principle generalize to unseen fraud patterns — at the cost of precision.
- **Extreme class imbalance:** fraud represents only ~0.17% of transactions (492 out of 284,807) — a more severe imbalance than any supervised project in this repo.
- **Isolation Forest concept:** repeatedly splits the data randomly; anomalies, being few and different, get isolated in fewer splits than normal points, which cluster together and require many splits to separate.
- **`contamination` parameter:** set to the known historical fraud rate (0.0017) to guide how many points the model should flag — in a real production setting without any historical label, this would need to be estimated differently (domain knowledge, business tolerance for false positives, or testing multiple values).
- **Result:** Isolation Forest flagged 485 transactions as anomalies (in line with the expected ~485 given the contamination rate), but only caught 25% of actual fraud (recall = 0.25, precision = 0.26) — most flagged anomalies were false alarms, and most real fraud went undetected.
- **Supervised comparison:** trained a `RandomForestClassifier(class_weight='balanced')` on the same data, this time using the true labels during training. Result: recall = 0.76, precision = 0.99 — a large improvement over the unsupervised approach, since the model could directly learn the signature of historically known fraud patterns.
- **Key conclusion:** the supervised model's clear superiority here doesn't make the unsupervised approach useless — it reflects that all fraud in this fixed historical dataset resembles patterns the supervised model could learn. In production, fraud techniques evolve, and a purely supervised model has no mechanism to catch a genuinely new pattern it was never trained on, while an anomaly detector can still flag it as "not normal" even with imperfect precision. In practice, fraud detection systems typically run both approaches in parallel (layered detection) — supervised models to efficiently catch known patterns with high precision, and unsupervised models as a safety net for the unknown — rather than treating this as a choice between one or the other.

## Methodology

### Mall Customer Segmentation

1. Exploratory Data Analysis (EDA)
2. Feature selection (excluding non-informative identifiers)
3. Feature scaling
4. Determining the optimal number of clusters (Elbow Method + Silhouette Score)
5. K-Means training and cluster visualization
6. Business interpretation of clusters
7. Dimensionality reduction investigation — PCA diagnosed via correlation analysis before applying (rather than applied by default), then t-SNE tested as a non-linear alternative once PCA proved unsuitable; distance-based metrics on the t-SNE embedding were treated with appropriate skepticism given the algorithm's known distortion of global distances

### Credit Card Fraud Detection

1. Exploratory Data Analysis (EDA) — assessing feature types and class imbalance
2. Feature scaling (`StandardScaler`)
3. Training Isolation Forest on features only, with the true label withheld
4. Converting Isolation Forest's `-1`/`1` output to the dataset's `0`/`1` fraud convention
5. Evaluating against the withheld true labels (confusion matrix, classification report) — the one point in this project where labels are used, purely for post-hoc validation
6. Benchmarking against a supervised `RandomForestClassifier` trained with the true labels, to quantify the real trade-off between the two approaches

## Key takeaways

- In unsupervised learning, there's no target to absorb the noise from an uninformative or unscaled feature — feature selection and scaling matter even more than in supervised learning.
- PCA only pays off when features are genuinely correlated/redundant; forcing it on largely independent features loses information without a real benefit. Diagnosing correlation before applying a technique — rather than applying it by default — is what separates deliberate methodology from following a checklist.
- A technique can be right for one purpose and wrong for another: t-SNE gave genuine visual confirmation of cluster structure but produced unreliable quantitative metrics, because it deliberately distorts global distances to preserve local neighborhoods. Knowing a tool's specific limitations — not just whether it "worked" — is what determines whether its output can be trusted for a given decision.
- A supervised model will almost always outperform an unsupervised one on a fixed historical dataset, because it directly learns from the true labels. That superiority doesn't automatically make the unsupervised approach useless — the real question is which one still works when the pattern being detected changes over time, as with evolving fraud techniques. Choosing between (or combining) approaches should be driven by that forward-looking question, not just by which one scores higher today.

## Tools

Python · pandas · scikit-learn · matplotlib · seaborn · kagglehub

## Author

Learning project — feedback and suggestions welcome!
