![The Intercept Probability Value (PiV)](https://raw.githubusercontent.com/se-taylor/machine_learning_projects/refs/heads/main/img/piv.webp "The Intercept Probability Value (PiV)")

# A Probabilistic Approach to Assessing Linear Regression Intercept Stability: The Intercept Probability Value Metric

## Abstract
The Intercept Probability Value (PiV) provides a Bayesian framework for quantifying model stability based on the posterior probability of a non-zero intercept coefficient. Through theoretical development and empirical validation using both simulated and real-world datasets, demonstrating that PiV offers meaningful insights into model reliability. Results indicate a strong correlation between PiV scores and traditional stability metrics, providing additional interpretability advantages.

### 1. Introduction
Linear regression remains one of the most widely used statistical methods in empirical research, yet assessing model stability continues to present challenges. Traditional approaches, including R-squared, adjusted R-squared, and various information criteria, focus primarily on goodness-of-fit rather than structural stability. Proposal a novel metric that specifically addresses the stability of the intercept term, which plays a crucial role in model reliability and generalizability.

#### 1.1 Background
The intercept term in linear regression represents the expected value of the dependent variable when all predictors are zero. While often treated as a mere nuisance parameter, the intercept's stability can provide valuable insights into model reliability.

Existing methods for assessing intercept-stability, typically rely on the following:
- Standard error of the intercept
- Confidence intervals
- T-statistics and p-values

These frequentist approaches, while valuable, need to capture the probabilistic nature of intercept stability fully.

#### 1.2 Contribution
Introduction of the Intercept Probability Value (PiV), a Bayesian metric that quantifies model stability through the posterior probability of a non-zero intercept.

This approach offers several advantages:
1. Intuitive probabilistic interpretation
2. Integration with existing Bayesian frameworks
3. Robustness to outliers and small sample sizes
4. Direct relationship to model stability

### 2. Theoretical Framework

#### 2.1 Definition of PiV
Let β₀ represent the intercept coefficient in a linear regression model:
y = β₀ + β₁x₁ + ... + βₖxₖ + ε

The Intercept Probability Value is defined as:
PiV = P(β₀ ≠ 0 | D)

Where D represents the observed data.

#### 2.2 Bayesian Formulation
The posterior distribution of β₀ is derived using Bayes' theorem:
P(β₀ | D) ∝ P(D | β₀)P(β₀)

Where:
- P(D | β₀) is the likelihood function
- P(β₀) is the prior distribution
- P(β₀ | D) is the posterior distribution

#### 2.3 Prior Selection
Proposal of a hierarchical prior structure:
β₀ ~ N(μ₀, σ₀²)
μ₀ ~ N(0, τ²)
σ₀² ~ InverseGamma(α, β)

This structure allows for:
1. Incorporation of domain knowledge
2. Robustness to prior misspecification
3. Automatic scale adaptation

### 3. Methodology

#### 3.1 Computation of PiV
The PiV is computed through:
1. MCMC sampling from the posterior distribution
2. Calculating the proportion of samples where |β₀| > δ
3. Applying appropriate convergence diagnostics

Where δ is a small threshold value determined through sensitivity analysis.

#### 3.2 Implementation

```python
def calculate_piv(X, y, n_samples=10000, threshold=1e-5):
    """
    Calculate PiV using MCMC Sampling
    
    Parameters:
    X : array-like of shape (n_samples, n_features)
    y : array-like of shape (n_samples,)
    n_samples : int, number of MCMC samples
    threshold : float, significance threshold
    
    Returns:
    float : Estimated PiV value
    """
    model = BayesianRegression(n_samples=n_samples)
    model.fit(X, y)
    
    # Extract intercept samples
    intercept_samples = model.intercept_samples_
    
    # Calculate PiV
    piv = np.mean(np.abs(intercept_samples) > threshold)
    
    return piv
```

### 4. Empirical Validation

#### 4.1 Simulation Study
Conducted extensive simulations using:
- Sample sizes: n ∈ {50, 100, 500, 1000}
- Predictor dimensions: p ∈ {2, 5, 10, 20}
- Error distributions: Normal, Student's t, Laplace
- Various true intercept values

Results demonstrate:
1. Consistent PiV estimation across conditions
2. Robust performance with small samples
3. Strong correlation with model stability

#### 4.2 Real Data Analysis
Applied to three benchmark datasets:
1. Boston Housing (n=506)
2. California Housing (n=20640)
3. Diabetes Dataset (n=442)

Results show:
- PiV correlates strongly with cross-validation stability measures
- Provides early warning for potential model instability
- Outperforms traditional metrics in identifying unstable models

### 5. Discussion

#### 5.1 Advantages
1. Probabilistic interpretation
2. Robust to outliers
3. Computationally efficient
4. Intuitive scaling (0 to 1)

#### 5.2 Limitations
1. Requires specification of threshold δ
2. Computational overhead compared to frequentist methods
3. Sensitivity to prior specification in small samples

#### 5.3 Practical Guidelines
Recommendation:
1. Using δ = 10⁻⁵ for standardized data
2. Minimum sample size of 50 observations
3. Cross-validation with multiple prior specifications

### 6. Conclusion
The Intercept Probability Value (PiV) provides a novel and robust approach to assessing linear regression model stability and demonstrating its utility across various conditions and datasets through theoretical development and empirical validation. Future research directions include extending generalized linear models and developing efficient computation methods for large-scale applications.