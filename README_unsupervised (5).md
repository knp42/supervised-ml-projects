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

### 3. Topic Modeling on 20 Newsgroups (NLP + Clustering)
K-Means clustering applied to ~11,300 raw text documents (20 Newsgroups dataset) to automatically discover topics — headers, footers, and quotes were stripped to prevent the model from relying on metadata instead of actual content, and no category label was used at any point during clustering.

- **Text vectorization:** used `TfidfVectorizer` (max 1,000 features, English stop words removed, min document frequency of 5) to convert raw text into 1,000-dimensional numeric vectors — a fundamentally different preprocessing step from any tabular project in this repo, since algorithms can't operate on raw text directly.
- **Curse of dimensionality:** K-Means run directly on the 1,000-dimension sparse TF-IDF matrix produced silhouette scores near zero (~0.01–0.015) regardless of k, far below any other project here. In very high-dimensional sparse spaces, distances between points become increasingly uniform, eroding K-Means' ability to separate meaningful groups — this is a known, expected limitation rather than a data quality issue.
- **Dimensionality reduction with TruncatedSVD:** applied `TruncatedSVD` (100 components) — the standard PCA-equivalent for sparse text data, since standard PCA doesn't support sparse matrices directly. Only ~28.5% of variance was retained, which is normal for text (natural language is inherently high-variance and non-redundant, unlike Mall Customer's structured numeric features), but silhouette scores still roughly quadrupled (from ~0.015 to ~0.06 at k=4), confirming the reduction restored real discriminative power to the distance calculations.
- **Cluster interpretation over raw metrics:** even the improved silhouette score (~0.06) remains low on the scale used elsewhere in this repo (Mall Customers: 0.55). Rather than treating this as failure, cluster centers were projected back into the original TF-IDF vocabulary space (`svd.inverse_transform`) to extract each cluster's top keywords. Result: 3 of 4 clusters were clearly and correctly interpretable — Sport (`game, team, hockey, players, season`), Religion (`god, jesus, bible, christian, faith`), and Computing/Tech support (`windows, drive, card, file, mail`) — with the 4th cluster capturing generic, cross-topic discussion words (`just, don, people, like, think`) that don't fit a single theme.
- **Key conclusion:** a low silhouette score doesn't necessarily mean a clustering result lacks value — it can reflect the metric's difficulty capturing genuinely fuzzy, overlapping boundaries in natural language, which are less clear-cut than numeric feature clusters. Qualitative validation (do the discovered groups make human sense?) is sometimes more informative than the quantitative score alone, especially in NLP contexts.

### 4. Global Development Clustering
K-Means clustering on 142 countries (Gapminder dataset, 2007 snapshot) using life expectancy, GDP per capita, and population to reveal development profiles — and to surface countries carrying multiple, compounding disadvantages at once.

- **Data reduction to one row per country:** the raw dataset has one row per country *per year* (1952–2007). Clustering years together instead of countries would have compared "France in 1952" to "France in 2007" as if they were unrelated points, producing clusters that mostly reflected *when* a row was recorded rather than any real difference between countries. Filtered to the most recent year (2007) so each country is a single, comparable observation — the same "one row per real-world unit" principle used for customers in the Mall Customer project.
- **Feature selection:** dropped `country`, `iso_alpha`, and `iso_num` — all identifiers with no real distance meaning (the same reasoning as `CustomerID`, including the less obvious case of `iso_num`, a numeric-looking ISO code that is still just an arbitrary identifier, not a real measurement). Also dropped `year`, which was constant after filtering to 2007 and carried no discriminative information.
- **Categorical encoding:** one-hot encoded `continent` (`drop_first=True`) — not because "some metrics don't handle categories well," but because K-Means computes numeric distances and cannot operate on text at all.
- **Choosing k (first pass):** Elbow Method and Silhouette Score converged less cleanly here than on Mall Customers — inertia declined gradually rather than showing a sharp bend, making the Silhouette Score the more decisive signal. It peaked at **k = 7 (0.6391)**.
- **A skewed feature dominating the clusters:** at k = 7, one cluster contained exactly two countries: **China and India**, separated from every other Asian nation. A `pop` histogram revealed why — population is extremely right-skewed (most countries under ~200M, with China and India isolated near 1.1–1.3B). `StandardScaler` rescales mean and variance but does not correct skew, so these two countries remained many standard deviations from everyone else even after scaling, letting population dominate the distance calculation over life expectancy and GDP.
- **Fix: log-transforming population before scaling.** `np.log(pop)` turned the distribution into a roughly bell-shaped curve. Re-running Elbow + Silhouette on the log-transformed features shifted the optimal cluster count to **k = 5 (0.5175)** — a lower peak score, but a more balanced one, since no single feature could dominate purely due to its raw scale anymore.
- **Result:** with population log-transformed, China and India rejoined a broader Asian cluster (alongside Afghanistan, Bangladesh, Indonesia, Iran, Iraq, Israel, Japan, and others) grouped primarily by development indicators rather than by population size alone.
- **Continent still partly shapes the clusters — and that's informative, not a flaw.** Several clusters aligned closely with continents (Africa, Europe, the Americas, Oceania each mostly formed their own group). But Asia consistently split into multiple clusters by development tier (e.g., a higher-income cluster with Japan, Israel, and Singapore, separate from a lower-income cluster with Afghanistan and Cambodia) — proof the clustering captures real economic variation within a continent, not just geography by another name.
- **Key conclusion:** a single skewed feature can silently dominate a clustering result even after standard scaling, producing a technically valid but narrow finding (here, "countries with extreme population" rather than "countries with similar development profiles"). Diagnosing feature distributions — not just applying `StandardScaler` by default — was necessary to get a result that matched the actual question being asked.

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

### Topic Modeling on 20 Newsgroups

1. Loading raw text data, stripping metadata (headers/footers/quotes) to avoid metadata-based shortcuts
2. Text vectorization with TF-IDF
3. Diagnosing near-zero silhouette scores on the raw high-dimensional sparse matrix (curse of dimensionality)
4. Dimensionality reduction with `TruncatedSVD` (the sparse-compatible equivalent of PCA)
5. Re-evaluating clustering quality (Silhouette Score) on the reduced space
6. Qualitative validation: projecting cluster centers back into vocabulary space to extract and interpret top keywords per cluster, since the quantitative metric alone was insufficient to judge quality

### Global Development Clustering

1. Reducing a multi-year panel dataset to one row per country (most recent year) to match the unit of comparison to the actual question being asked
2. Feature selection (dropping identifiers, including a numeric-looking but still arbitrary ISO code) and categorical encoding of `continent`
3. Determining k via Elbow + Silhouette on the raw features
4. Diagnosing a skewed feature (population) that was silently dominating cluster assignment, via histogram inspection
5. Log-transforming the skewed feature and re-scaling before re-clustering
6. Re-validating k on the corrected features and comparing cluster composition before and after the fix

## Key takeaways

- In unsupervised learning, there's no target to absorb the noise from an uninformative or unscaled feature — feature selection and scaling matter even more than in supervised learning.
- PCA only pays off when features are genuinely correlated/redundant; forcing it on largely independent features loses information without a real benefit. Diagnosing correlation before applying a technique — rather than applying it by default — is what separates deliberate methodology from following a checklist.
- A technique can be right for one purpose and wrong for another: t-SNE gave genuine visual confirmation of cluster structure but produced unreliable quantitative metrics, because it deliberately distorts global distances to preserve local neighborhoods. Knowing a tool's specific limitations — not just whether it "worked" — is what determines whether its output can be trusted for a given decision.
- A supervised model will almost always outperform an unsupervised one on a fixed historical dataset, because it directly learns from the true labels. That superiority doesn't automatically make the unsupervised approach useless — the real question is which one still works when the pattern being detected changes over time, as with evolving fraud techniques. Choosing between (or combining) approaches should be driven by that forward-looking question, not just by which one scores higher today.
- High-dimensional sparse data (like TF-IDF text vectors) suffers from the curse of dimensionality — distances between points become increasingly uniform as dimensions grow, undermining distance-based algorithms like K-Means. Dimensionality reduction (TruncatedSVD for sparse data) can partially restore separability, but text clustering typically won't reach the silhouette scores seen on clean structured data — and that's expected, not a sign of failure.
- Quantitative metrics don't always tell the full story: a low silhouette score can still coexist with clusters that are highly meaningful to a human reader. Validating unsupervised results qualitatively (e.g., inspecting top keywords per cluster) is sometimes the more honest way to judge success, particularly for text and other unstructured data.
- Scaling alone doesn't fix a skewed distribution. `StandardScaler` corrects for scale (mean/variance) but not for shape — a heavily right-skewed feature with a few extreme outliers can still dominate distance calculations after scaling. Inspecting feature distributions (histograms) before clustering, and applying a transformation like `log()` when a feature is skewed, can materially change which structure the algorithm finds.

## Tools

Python · pandas · scikit-learn · matplotlib · seaborn · kagglehub

## Author

Learning project — feedback and suggestions welcome!
