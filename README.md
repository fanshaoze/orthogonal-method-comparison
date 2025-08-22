# Comparison of Orthogonalization Methods

## 1. Motivation
Orthonormal bases are fundamental in many fields such as linear algebra, numerical analysis, and quantum computing, where precise vector orthogonalization is critical for stability and accuracy. However, different orthogonalization methods (e.g., Gram-Schmidt, QR decomposition) may perform differently in numerical precision, especially when handling high-dimensional or complex vectors.

This code systematically compares several orthogonalization procedures by evaluating how well they produce orthonormal bases under different thresholds. By counting how many vectors satisfy orthogonality and normalization conditions at various precision levels, the code provides insights into which methods are more robust for real and complex vector spaces of different dimensions.

## 2. Data
The code generates synthetic datasets to test the orthogonalization procedures:

- **Vector Generation**:  
  The `generate_random_vectors` function creates multiple groups of random vectors:
  - `num_groups`: 1000 groups of vectors.
  - `n`: Number of vectors per group (2 to 64).
  - `m`: Dimension of each vector (equal to n, forming square matrices where columns are vectors).
  - `is_complex`: Flag to generate real (`float32`) or complex (`complex64`) vectors.

- **Data Structure**:  
  Output tensor shape: `(num_groups, n, m)`.

## 3. Basic Workflow
1. **Generate Test Data**:  
   For dimensions `[2,4,8,16,32,64]`, two datasets are created:
   - Real vectors (`vectors_real`)
   - Complex vectors (`vectors_complex`)

2. **Orthogonalization Methods**:  
   - `gram_schmidt`: Classic Gram-Schmidt  
   - `gram_schmidt_mgs`: Modified Gram-Schmidt  
   - `gram_schmidt_sciQR`: SciPy QR decomposition  
   - `gram_schmidt_re_orth`: Gram-Schmidt with re-orthogonalization  
   - `qr_procedure`: PyTorch QR decomposition

3. **Performance Evaluation**:  
   - **Normalization**: each vector norm `||u_i||_2` should satisfy `| ||u_i||_2 - 1 | <= ε`  
   - **Orthogonality**: distinct vectors satisfy `| <u_i, u_j> | <= ε, i != j`  
   - **Thresholds tested**: `[1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8]`

4. **Result Aggregation and Visualization**:  
   - `check_procedures` counts orthonormal vectors per method and threshold, saves CSV.  
   - Generates grouped bar plot and saves PNG.

## 4. Orthogonalization Methods

### 4.1 Gram-Schmidt
Given matrix `V = [v1, v2, ..., vn]`:

1. `u1 = v1 / ||v1||_2`  
2. For k = 2 to n:  
   `wk = vk - sum(<uj, vk> * uj for j = 1..k-1)`  
   `uk = wk / ||wk||_2`  
3. Optionally, upper triangular matrix `R` with `R[i,j] = <ui, vj>`

Output: Orthonormal basis `U = [u1,...,un]` (and `R` if needed)

---

### 4.2 Modified Gram-Schmidt (MGS)
Subtract projections immediately:
`vk = vk - <uj, vk> * uj` for j < k  
`uk = vk / ||vk||_2`

- Improves numerical stability.  
- For complex vectors, use conjugate inner product: `<u, v> = u^H v`

---

### 4.3 SciPy QR (sciQR)
Uses SVD + QR decomposition:

1. `A = U Σ V^H`  
2. `Σ V^H = Q2 R2 => A = (U Q2) R2`  
3. Output: `U' = U Q2` (orthonormal basis), `R' = R2`

- Advantages: stable for nearly rank-deficient matrices  
- Disadvantages: slower (O(n^3)) and needs CPU/GPU transfer

---

### 4.4 Re-Orthogonalization
Apply Gram-Schmidt twice:

`U' = GS(V)`, then `U = GS(U')`

Reduces residual projections for ill-conditioned vectors

---

### 4.5 Torch QR
`A = Q R`, with `Q^H Q = I`

Fast but slightly less stable than SciPy QR

---

## 5. Result Analysis

- **Real Vector Spaces**: SciPy QR is the most stable, giving the highest count of orthonormal vectors  
- **Complex Vector Spaces**: Classic Gram-Schmidt preserves orthogonality best

---

## 6. Conclusion

- Real vectors: use SciPy QR for stability  
- Complex vectors: use classic Gram-Schmidt for strict orthogonality

This provides guidance for choosing orthogonalization methods for accurate computation in linear algebra, signal processing, or quantum computing.
