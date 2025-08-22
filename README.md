# Comparison of Orthogonalization Methods

## 1. Motivation
Orthonormal bases are fundamental in numerous fields such as linear algebra, numerical analysis, and quantum computing, where precise vector orthogonalization is critical for stability and accuracy. However, different orthogonalization methods (e.g., Gram-Schmidt, QR decomposition) may exhibit varying performance in terms of numerical precision, especially when handling high-dimensional or complex vectors. 

This code aims to systematically compare the effectiveness of several orthogonalization procedures by evaluating their ability to produce orthonormal bases under different thresholds. By quantifying how many vectors satisfy orthogonality and normalization criteria across varying precision levels, the code provides insights into which methods are more robust for real and complex vector spaces of different dimensions.

## 2. Data
The code generates synthetic datasets to test the orthogonalization procedures, with the following characteristics:

- **Vector Generation**:  
  The `generate_random_vectors` function creates multiple groups of random vectors. Key parameters include:
  - `num_groups`: 1000 groups of vectors (to ensure statistical robustness).  
  - `n`: Number of vectors per group (ranging from 2 to 64, corresponding to the dimension of the vector space).  
  - `m`: Dimension of each vector (equal to n, forming square matrices where columns are vectors).  
  - `is_complex`: A flag to generate either real vectors (`float32`) or complex vectors (`complex64`, with random real and imaginary parts).  

- **Data Structure**:  
  The output is a tensor of shape `(num_groups, n, m)`, where each entry represents a group of n vectors, each of dimension m.

## 3. Basic Workflow
The code follows a structured pipeline to generate data, apply orthogonalization methods, evaluate their performance, and visualize results:

1. **Generate Test Data**:  
   For each dimension i (2, 4, 8, 16, 32, 64), two datasets are created:
   - Real vectors (`vectors_real`).
   - Complex vectors (`vectors_complex`).

2. **Orthogonalization Methods**:  
   Five procedures are implemented and tested:
   - `gram_schmidt`: Classic Gram-Schmidt orthogonalization.  
   - `gram_schmidt_mgs`: Modified Gram-Schmidt (improves numerical stability).  
   - `gram_schmidt_sciQR`: Orthogonalization via SciPy’s QR decomposition (high-precision reference).  
   - `gram_schmidt_re_orth`: Gram-Schmidt with re-orthogonalization (enhances precision by repeating projections).  
   - `qr_procedure`: Orthogonalization via PyTorch’s built-in QR decomposition.  

3. **Performance Evaluation**:  
   The `check_gram_schmidt_procedure` function verifies if the output of each method meets orthonormality criteria:
   - **Normalization**: Each vector’s norm must satisfy `| ||u_i||_2 - 1 | <= threshold`.  
   - **Orthogonality**: The inner product of any two distinct vectors must satisfy `| <u_i, u_j> | <= threshold`.  
   - **Thresholds tested**: `[1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8]`.  

4. **Result Aggregation and Visualization**:  
   - `check_procedures` aggregates results (count of orthonormal vectors per method and threshold) and saves them to CSV files.  
   - A grouped bar plot is generated to visualize the performance comparison and saved as a PNG file.  

## 4. Orthogonalization Methods

### 4.1 Gram-Schmidt
1. **Input Validation**: Ensure `vv` is a 2D tensor (columns represent vectors).  
2. **Projection Function**: Define a helper to calculate the projection of vector v onto vector u.  
3. **Initialize Storage**: Create `uu` (same shape as `vv`) to store orthonormalized vectors.  
4. **First Vector**: Copy the first column of `vv` directly into `uu`.  
5. **Orthogonalize Remaining Vectors**:  
   - For each subsequent vector: subtract its projections onto all previously processed vectors in `uu`.  
6. **Normalize**: Scale each column in `uu` to unit length.  
7. **Optional Rotation Matrix**: If enabled, compute upper triangular matrix `rr`.  
8. **Return**: Orthonormal basis `uu` (and `rr` if requested).  

### 4.2 Modified Gram-Schmidt (MGS)
- In each iteration, the projection component is subtracted **immediately** after calculation:  
  `vk = vk - torch.vdot(uj, vk) * uj`.  
- **Advantages**:
  - Greater numerical stability than CGS.  
  - Avoids cumulative error propagation.  
- **Complex Numbers**:
  - `torch.vdot` automatically applies complex conjugation.  
  - Using `(v * u).sum()` is insufficient for complex vectors and can break orthogonality.  

### 4.3 SciPy QR (sciQR)
- **Steps**:
  1. Perform SVD: `A = U * Sigma * V^H`.  
     - Columns of `U` are strictly orthogonal.  
  2. Perform QR on `Sigma * V^H`:  
     - `Sigma * V^H = Q2 * R2` → `A = (U * Q2) * R2`.  
     - Output: `uu = U * Q2` (orthonormal basis), optionally `rr = R2`.  
- **Advantages**:
  - High-quality orthogonal bases even for nearly linearly dependent vectors.  
  - Maintains stability due to orthogonality of singular vectors.  
- **Disadvantages**:
  - Slower than GS/MGS due to SVD (`O(n^3)` operations).  
  - Requires CPU/GPU data transfer.  

### 4.4 Re-Orthogonalization
- Apply MGS **twice** to reduce residual projections: `U' = GS(V)`, then `U = GS(U')`.  
- Enhances orthogonality for vectors with high condition numbers.  
- Also known as **iterated Gram-Schmidt**.  

### 4.5 Torch QR
- Uses `torch.linalg.qr` and `torch.linalg.norm`.  
- Fast but less stable than SciPy QR.  

## 5. Result Analysis
- **Real Vector Spaces**:  
  - SciPy QR (`sciQR`) consistently outperforms others.  
  - Highest count of orthonormal vectors across thresholds.  
  - Best choice for **real-valued** vectors.  

- **Complex Vector Spaces**:  
  - Classic Gram-Schmidt performs best.  
  - Preserves orthogonality better than QR-based methods.  
  - Suitable for **complex-valued** applications (e.g., quantum computing).  

## 6. Conclusion
The performance of orthogonalization methods depends on the nature of the vector space:  

- **Real vectors** → **Use SciPy QR decomposition (sciQR)** for maximum stability.  
- **Complex vectors** → **Use Classic Gram-Schmidt** for better orthogonality.  

These findings provide practical guidance for selecting orthogonalization methods to ensure higher precision in applications such as numerical analysis, signal processing, or quantum computing.
