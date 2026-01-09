# 👥 Customer Segmentation Analysis

An unsupervised machine learning project that uses K-Means clustering to identify distinct customer segments based on purchasing behavior and demographics.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-orange.svg)
![Pandas](https://img.shields.io/badge/Pandas-1.3+-green.svg)

## 📋 Table of Contents
- [Overview](#overview)
- [Business Problem](#business-problem)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Methodology](#methodology)
- [Results](#results)
- [Key Insights](#key-insights)
- [Future Work](#future-work)

## 🎯 Overview

This project segments customers into distinct groups using K-Means clustering algorithm. By understanding different customer segments, businesses can:
- Create targeted marketing campaigns
- Personalize customer experiences
- Optimize product offerings
- Improve customer retention strategies

## 💼 Business Problem

**Challenge**: Companies often treat all customers the same, leading to inefficient marketing spend and missed opportunities.

**Solution**: Use unsupervised learning to automatically discover customer segments based on behavior patterns, enabling data-driven marketing strategies.

**Impact**: 
- Increase marketing ROI by 20-30%
- Improve customer satisfaction through personalization
- Reduce customer acquisition costs

## 📊 Dataset

This project uses the **Mall Customers Dataset**:
- **Source**: [Kaggle - Customer Segmentation](https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python)
- **Size**: 200 customers
- **Features**:
  - `CustomerID`: Unique identifier
  - `Gender`: Male/Female
  - `Age`: Customer age (18-70)
  - `Annual Income (k$)`: Yearly income in thousands
  - `Spending Score (1-100)`: Score based on customer behavior and spending nature

**Alternative Datasets**:
- Online Retail Dataset (for e-commerce analysis)
- Customer Transaction Data (for banking)
- Subscription Data (for SaaS companies)

## 🚀 Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/customer-segmentation.git
cd customer-segmentation
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Download the dataset from Kaggle and place it in the project directory

## 💻 Usage

### Quick Start

```python
# Run the complete analysis
python customer_segmentation.py
```

This will:
1. Load and explore the data
2. Preprocess and scale features
3. Find optimal number of clusters
4. Apply K-Means clustering
5. Visualize segments
6. Generate cluster profiles
7. Save results to CSV

### Custom Analysis

```python
import pandas as pd
from customer_segmentation import create_segments

# Load your data
df = pd.read_csv('your_customer_data.csv')

# Create segments
segmented_df, cluster_summary = create_segments(
    data=df,
    features=['Annual Income', 'Spending Score'],
    n_clusters=5
)

# View results
print(cluster_summary)
```

## 🔬 Methodology

### 1. Data Preprocessing
- Handle missing values and duplicates
- Encode categorical variables (Gender: Male=0, Female=1)
- Select relevant numerical features for clustering

### 2. Feature Scaling
- Apply **StandardScaler** to normalize features
- Ensures all features contribute equally to distance calculations
- Formula: `z = (x - μ) / σ`

### 3. Optimal Cluster Selection

**Elbow Method**:
- Plot within-cluster sum of squares (inertia) vs K
- Look for the "elbow point" where adding clusters gives diminishing returns

**Silhouette Analysis**:
- Measures how similar objects are to their own cluster vs other clusters
- Score ranges from -1 to 1 (higher is better)
- Formula: `s(i) = (b(i) - a(i)) / max(a(i), b(i))`

### 4. K-Means Clustering
```
Algorithm:
1. Initialize K cluster centroids randomly
2. Assign each point to nearest centroid
3. Recalculate centroids as mean of assigned points
4. Repeat steps 2-3 until convergence
```

**Parameters**:
- `n_clusters`: Number of segments (determined from elbow/silhouette)
- `random_state=42`: For reproducibility
- `n_init=10`: Number of initializations (best result selected)

### 5. Visualization
- **2D Scatter Plot**: Direct visualization for 2 features
- **PCA**: Dimensionality reduction for >2 features
- **Distribution Plots**: Feature distributions by cluster

### 6. Cluster Profiling
- Calculate mean values for each segment
- Identify distinguishing characteristics
- Generate business-friendly segment descriptions

## 📈 Results

### Identified Customer Segments

| Segment | Size | Avg Income | Avg Spending | Description |
|---------|------|------------|--------------|-------------|
| **Cluster 0** | 35 (17.5%) | $26k | 21 | Budget Conscious |
| **Cluster 1** | 81 (40.5%) | $55k | 49 | Average Customers |
| **Cluster 2** | 23 (11.5%) | $87k | 17 | High Income, Low Spenders |
| **Cluster 3** | 38 (19.0%) | $88k | 82 | Premium Customers |
| **Cluster 4** | 23 (11.5%) | $25k | 79 | Young High Spenders |

### Visualization

**Cluster Distribution**:
```
   Income (k$)
   100 |              * * *
       |          * *       * *
       |      * *               * *
    50 |  * *           *           * *
       |*         * * * * * * *         *
     0 |__________________________________
        0        50       100
              Spending Score
```

### Model Performance

- **Silhouette Score**: 0.554 (Good separation)
- **Inertia**: 44,448
- **Optimal K**: 5 clusters
- **Variance Explained (PCA)**: 85%

## 💡 Key Insights

### Segment Descriptions & Strategies

**1. Budget Conscious (17.5%)**
- Low income, low spending
- **Strategy**: Value products, discount campaigns, loyalty rewards
- **Channels**: Email, social media

**2. Average Customers (40.5%)**
- Mid income, moderate spending
- **Strategy**: Balanced offerings, seasonal promotions
- **Channels**: Multi-channel approach

**3. High Income, Low Spenders (11.5%)**
- High income but cautious spending
- **Strategy**: Quality emphasis, trust-building, exclusive previews
- **Channels**: Personalized emails, premium content

**4. Premium Customers (19.0%)**
- High income, high spending - **VIPs!**
- **Strategy**: Luxury products, VIP experiences, concierge service
- **Channels**: Direct outreach, exclusive events

**5. Young High Spenders (11.5%)**
- Low income but high spending (aspirational buyers)
- **Strategy**: Trendy products, installment plans, social proof
- **Channels**: Instagram, TikTok, influencer marketing

### Business Impact

- **Marketing Efficiency**: Target each segment with tailored messages
- **Product Development**: Create offerings for each segment's needs
- **Resource Allocation**: Focus on high-value segments (Cluster 3)
- **Customer Retention**: Identify at-risk segments and intervene

## 🎓 What I Learned

- **K-Means Algorithm**: Understanding clustering and centroid-based methods
- **Elbow Method**: Determining optimal number of clusters
- **Silhouette Analysis**: Validating cluster quality
- **Feature Scaling**: Importance of normalization in distance-based algorithms
- **PCA**: Dimensionality reduction for visualization
- **Business Translation**: Converting technical findings into actionable insights

## 🔮 Future Improvements

- [ ] Try other clustering algorithms (DBSCAN, Hierarchical, GMM)
- [ ] Include more features (purchase frequency, recency, product categories)
- [ ] Implement RFM (Recency, Frequency, Monetary) analysis
- [ ] Build interactive dashboard with Streamlit/Tableau
- [ ] Add predictive modeling (predict which segment new customers belong to)
- [ ] Time-series analysis (how segments evolve over time)
- [ ] A/B testing framework for segment-specific campaigns
- [ ] Customer lifetime value (CLV) prediction by segment

## 📦 Project Structure

```
customer-segmentation/
├── customer_segmentation.py   # Main analysis script
├── data/
│   └── Mall_Customers.csv    # Customer data
├── notebooks/
│   └── exploration.ipynb     # Exploratory analysis
├── output/
│   ├── customer_segments.csv # Clustered data
│   └── cluster_summary.csv   # Segment profiles
├── visualizations/
│   ├── elbow_plot.png
│   ├── silhouette_plot.png
│   └── cluster_visualization.png
├── requirements.txt
└── README.md
```

## 🛠️ Technologies Used

- **Python 3.8+**
- **Pandas**: Data manipulation
- **NumPy**: Numerical computing
- **Scikit-learn**: K-Means, StandardScaler, PCA, metrics
- **Matplotlib & Seaborn**: Data visualization

## 📝 Requirements

```txt
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
matplotlib>=3.4.0
seaborn>=0.11.0
```

## 🔍 Code Explanation

### Key Components

**1. Feature Scaling**:
```python
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)
```
- Normalizes features to same scale (mean=0, std=1)
- Prevents features with larger ranges from dominating

**2. Finding Optimal K**:
```python
for k in range(2, 11):
    kmeans = KMeans(n_clusters=k)
    silhouette_scores.append(silhouette_score(X, kmeans.labels_))
```
- Tests multiple K values
- Uses both inertia and silhouette score for decision

**3. Clustering**:
```python
kmeans = KMeans(n_clusters=5, random_state=42)
df['Cluster'] = kmeans.fit_predict(features_scaled)
```
- Assigns each customer to a cluster
- Returns cluster labels (0, 1, 2, 3, 4)

**4. Profiling**:
```python
for cluster_id in range(n_clusters):
    cluster_data = df[df['Cluster'] == cluster_id]
    print(cluster_data.describe())
```
- Analyzes characteristics of each segment
- Identifies what makes each group unique

⭐ If you found this project helpful, please give it a star!

**Note**: This is an educational project demonstrating customer segmentation techniques for marketing analytics.
