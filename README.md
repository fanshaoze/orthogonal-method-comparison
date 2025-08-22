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
   For each dimension in [2,4,8,16,32,64], two datasets are created:
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
   - **Normalization**: Each vector’s norm must satisfy  
     $$
     |\|u_i\|_2 - 1| \le \varepsilon.
     $$
   - **Orthogonality**: Distinct vectors must satisfy  
     $$
     |\langle u_i, u_j \rangle| \le \varepsilon, \quad i \ne j.
     $$
   - **Thresholds tested**: \([10^{-2}, 10^{-3}, 10^{-4}, 10^{-5}, 10^{-6}, 10^{-7}, 10^{-8}]\).  

4. **Result Aggregation and Visualization**:  
   - `check_procedures` aggregates results (count of orthonormal vectors per method and threshold) and saves them to CSV files.  
   - A grouped bar plot is generated to visualize the performance comparison and saved as a PNG file.  

## 4. Orthogonalization Methods

### 4.1 Gram-Schmidt
Given input matrix \(V = [v_1, v_2, \dots, v_n]\):

1. Set  
$$
u_1 = \frac{v_1}{\|v_1\|_2}.
$$
2. For each \(k = 2, \dots, n\):  
$$
w_k = v_k - \sum_{j=1}^{k-1} \langle u_j, v_k \rangle u_j, \quad
u_k = \frac{w_k}{\|w_k\|_2}.
$$
3. Optionally compute the upper triangular matrix \(R\) with  
$$
R_{ij} = \langle u_i, v_j \rangle.
$$

Output: Orthonormal basis \(U = [u_1,\dots,u_n]\) (and \(R\) if requested).

---

### 4.2 Modified Gram-Schmidt (MGS)
The projection is subtracted **immediately** after each step:
$$
v_k \gets v_k - \langle u_j, v_k \rangle u_j, \quad \text{for each } j < k,
$$
$$
u_k = \frac{v_k}{\|v_k\|_2}.
$$
- Improves numerical stability by reducing error accumulation.  
- For complex vectors, use the **conjugate inner product**:  
$$
\langle u, v \rangle = u^H v.
$$

---

### 4.3 SciPy QR (sciQR)
Uses SVD + QR decomposition for high precision:
$$
A = U \Sigma V^H.
$$
Since columns of \(U\) are orthogonal, perform
$$
\Sigma V^H = Q_2 R_2 \quad \Longrightarrow \quad
A = (U Q_2) R_2.
$$
Output: 
$$
U' = U Q_2 \quad \text{(orthonormal basis)}, \quad
R' = R_2.
$$

- **Advantages**: stable even if \(A\) is nearly rank-deficient.  
- **Disadvantages**: computationally expensive (\(O(n^3)\)) and requires CPU↔GPU data transfer.

---

### 4.4 Re-Orthogonalization
Apply Gram-Schmidt **twice**:
$$
U' = \mathrm{GS}(V), \quad U = \mathrm{GS}(U').
$$
This reduces residual projections for ill-conditioned vectors.  

---

### 4.5 Torch QR
Directly use PyTorch's implementation:
$$
A = Q R, \quad Q^H Q = I.
$$
Fast but slightly less stable than SciPy QR.

---

## 5. Result Analysis

- **Real Vector Spaces**:  
  SciPy QR (`sciQR`) consistently outperforms others, maintaining the highest count of orthonormal vectors across thresholds.  
  → **Best choice for real-valued vectors**.  

- **Complex Vector Spaces**:  
  Classic Gram-Schmidt achieves the best orthogonality preservation.  
  → **Best choice for complex-valued applications** (e.g., quantum computing).  

---

## 6. Conclusion

- For **real vectors**: use **SciPy QR decomposition (sciQR)** for maximum numerical stability.  
- For **complex vectors**: use **Classic Gram-Schmidt** to preserve orthogonality.  

These findings provide practical guidance for selecting orthogonalization methods to ensure higher precision in applications such as numerical analysis, signal processing, or quantum computing.
