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

## Methodology

1. Exploratory Data Analysis (EDA)
2. Feature selection (excluding non-informative identifiers)
3. Feature scaling
4. Determining the optimal number of clusters (Elbow Method + Silhouette Score)
5. K-Means training and cluster visualization
6. Business interpretation of clusters
7. Dimensionality reduction investigation — PCA diagnosed via correlation analysis before applying (rather than applied by default), then t-SNE tested as a non-linear alternative once PCA proved unsuitable; distance-based metrics on the t-SNE embedding were treated with appropriate skepticism given the algorithm's known distortion of global distances

## Key takeaways

- In unsupervised learning, there's no target to absorb the noise from an uninformative or unscaled feature — feature selection and scaling matter even more than in supervised learning.
- PCA only pays off when features are genuinely correlated/redundant; forcing it on largely independent features loses information without a real benefit. Diagnosing correlation before applying a technique — rather than applying it by default — is what separates deliberate methodology from following a checklist.
- A technique can be right for one purpose and wrong for another: t-SNE gave genuine visual confirmation of cluster structure but produced unreliable quantitative metrics, because it deliberately distorts global distances to preserve local neighborhoods. Knowing a tool's specific limitations — not just whether it "worked" — is what determines whether its output can be trusted for a given decision.

## Tools

Python · pandas · scikit-learn · matplotlib · seaborn

## Author

Learning project — feedback and suggestions welcome!
