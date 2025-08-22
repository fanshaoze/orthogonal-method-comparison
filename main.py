import torch
import csv
from scipy.linalg import svd, qr
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


def gram_schmidt(vv: torch.Tensor, if_rotation_matrix: bool = True):
    """Gram Schmidt procedure for a set of basis vv (dim=2, columns
    of vectors). The precision is at 1e-6 level.

    Args:
        vv (torch.Tensor): Dim=2 torch tensor of set of basis.
        if_rotation_matrix (bool, optional): Flag for computing rotation matrix. Defaults to True.

    Returns:
        torch.Tensor or (torch.Tensor, torch.Tensor): Normalized basis (and rotation matrix).
    """
    # Assert the input basis forms a 2D matrix, columns are the vectors
    assert (vv.dim() == 2)

    # save vv to the file
    # vv = vv.clone().detach()

    def projection(u, v):  # Assuming 1D torch tensor
        return (v * u).sum() / (u * u).sum() * u

    # uu is dim [2^num_qubits, K] orthonormal basis
    num_vec = vv.size(1)
    uu = torch.zeros_like(vv, dtype=vv.dtype, device=vv.device)
    uu[:, 0] = vv[:, 0].clone()
    for k in range(1, num_vec):
        vk = vv[:, k].clone()
        uk = 0
        for j in range(0, k):
            uj = uu[:, j].clone()
            uk = uk + projection(uj, vk)
        uu[:, k] = vk - uk
    for k in range(num_vec):
        uk = uu[:, k].clone()
        uu[:, k] = uk / uk.norm()
    # If we want to calculate the rotation matrix, this would be of form A = QR,
    # where A (vv) stores columns of basis vectors
    # Q (uu) stores columns of orthonormal basis, R is the rotation matrix.
    rr = torch.zeros(num_vec, num_vec, device=vv.device)
    if if_rotation_matrix:
        for ii in range(num_vec):
            for jj in range(ii, num_vec):
                rr[ii, jj] = torch.vdot(uu[:, ii], vv[:, jj])
        return uu, rr
    return uu


def qr_procedure(vv: torch.Tensor, if_rotation_matrix: bool = True):
    """
        # %% Use torch QR decomposition, so from this
        # official QR decomposition, looks like 1e-6 is the precision
        """
    Q, R = torch.linalg.qr(vv)
    Q = Q / torch.linalg.norm(Q, dim=0)
    uu = Q
    rr = R
    if if_rotation_matrix:
        return uu, rr
    return uu


def gram_schmidt_sciQR(vv: torch.Tensor, if_rotation_matrix: bool = True):
    from scipy.linalg import qr
    """
    high precision Gram-Schmidt orthogonalization using SciPy's QR decomposition.
    输入输出与原函数保持一致。
    Args:
        vv (torch.Tensor): [dim, num_vec]，列为待正交化向量
        if_rotation_matrix (bool, optional): 是否返回旋转矩阵R

    Returns:
        torch.Tensor 或 (torch.Tensor, torch.Tensor):
            uu: 正交化后的基向量 (与输入同device、dtype)
            rr: 旋转矩阵 (与输入同device、dtype)，仅当 if_rotation_matrix=True
    """

    assert vv.dim() == 2, "Input must be a 2D tensor with column vectors."
    device = vv.device
    in_dtype = vv.dtype
    is_complex = torch.is_complex(vv)
    # 升精度到 numpy 高精度计算
    vv_np = vv.detach().cpu().numpy().astype(np.complex128 if is_complex else np.float64, copy=False)
    # A = U Σ V^H（列空间正交性由 U 保证，极稳）
    U, s, Vh = svd(vv_np, full_matrices=False)  # shapes: U[m,k], s[k], Vh[k,k]
    # 将 SVD 转换为 QR 形式：令 R_temp = Σ V^H（k×k），再 QR 分解 R_temp = Q2 R2
    # 则 A = U Σ V^H = (U Q2) R2， 其中 Q = U Q2 仍为列正交，R2 为上三角
    R_temp = (np.diag(s) @ Vh)  # k×k
    Q2, R2 = qr(R_temp, mode='economic')  # Q2(k×k), R2(k×k 上三角)
    Q = U @ Q2  # m×k
    uu = torch.from_numpy(Q).to(device=device, dtype=in_dtype)
    rr = torch.from_numpy(R2).to(device=device, dtype=in_dtype)
    if if_rotation_matrix:
        return uu, rr
    return uu


def gram_schmidt_mgs(vv: torch.Tensor, if_rotation_matrix: bool = True):
    """
    high precision Gram-Schmidt orthogonalization with modified Gram-Schmidt.
    :param vv:
    :param if_rotation_matrix:
    :return:
    """
    num_vec = vv.size(1)
    uu = torch.zeros_like(vv, dtype=vv.dtype, device=vv.device)

    for k in range(num_vec):
        vk = vv[:, k].clone()
        for j in range(k):
            uj = uu[:, j]
            vk = vk - torch.vdot(uj, vk) * uj
        uu[:, k] = vk / vk.norm()

    if if_rotation_matrix:
        rr = torch.zeros(num_vec, num_vec, dtype=vv.dtype, device=vv.device)
        for ii in range(num_vec):
            for jj in range(ii, num_vec):
                rr[ii, jj] = torch.vdot(uu[:, ii], vv[:, jj])
        return uu, rr

    return uu


def gram_schmidt_re_orth(vv: torch.Tensor, if_rotation_matrix: bool = True):
    """
    high precision Gram-Schmidt orthogonalization with re-orthogonalization.
    :param vv:
    :param if_rotation_matrix:
    :return:
    """
    assert vv.dim() == 2, "Input must be a 2D tensor with column vectors."

    device = vv.device
    dtype = vv.dtype
    num_vec = vv.size(1)
    uu = torch.zeros_like(vv, device=device, dtype=dtype)

    for k in range(num_vec):
        vk = vv[:, k].clone()
        # 第一次正交化
        for j in range(k):
            vk -= torch.vdot(uu[:, j], vk) * uu[:, j]
        # 第二次重正交化
        for j in range(k):
            vk -= torch.vdot(uu[:, j], vk) * uu[:, j]
        uu[:, k] = vk / vk.norm()

    if if_rotation_matrix:
        rr = torch.zeros((num_vec, num_vec), device=device, dtype=dtype)
        for ii in range(num_vec):
            for jj in range(ii, num_vec):
                rr[ii, jj] = torch.vdot(uu[:, ii], vv[:, jj])
        return uu, rr

    return uu


def check_gram_schmidt_procedure(vectors, threshold=1e-7, print_info=False, method='gs'):
    """
        Check if the Gram-Schmidt procedure has been applied correctly to the local basis vectors.
        Args:
            excitations (List[str]): A list of excitation operators that defines the local basis.
            device (torch.device): The torch.device.
            threshold (float): The threshold for checking orthonormality.
            :param method:
            :param print_info:
        """

    # check with gram_schmidt
    # vectors = torch.cat(vectors, dim=0).T

    if method == "gs":
        orthonormal_basis, rr = gram_schmidt(vectors)
    elif method == 'sciQR':
        orthonormal_basis, rr = gram_schmidt_sciQR(vectors)
    elif method == 'mgs':
        orthonormal_basis, rr = gram_schmidt_mgs(vectors)
    elif method == 're_orth':
        orthonormal_basis, rr = gram_schmidt_re_orth(vectors)
    elif method == 'torch_qr':
        orthonormal_basis, rr = qr_procedure(vectors)
    else:
        raise ValueError("Please provide the method in (gs, mgs, sciQR, re_orth, torch_qr).")
    un_normal_count = 0
    un_orth_count = 0
    for ii in range(len(orthonormal_basis[0])):
        if torch.abs(torch.norm(orthonormal_basis[:, ii]) - 1.0) > threshold:
            un_normal_count += 1
            if print_info:
                print(f"Orthonormality check failed for vector {ii}, norm: {torch.norm(orthonormal_basis[:, ii])}")
        for jj in range(ii):
            if torch.abs(torch.inner(orthonormal_basis[:, jj], orthonormal_basis[:, ii]) - 0.0) > threshold:
                un_orth_count += 1
                if print_info:
                    print(f"Orthonormality check failed for vectors {ii} and {jj}, "
                          f"inner product: {torch.inner(orthonormal_basis[:, jj], orthonormal_basis[:, ii])}")
    if un_normal_count > 0 or un_orth_count > 0:
        if print_info:
            print(f"Un-normalized vectors: {un_normal_count}, Un-orthogonal vectors: {un_orth_count}")
        return False, un_normal_count, un_orth_count
    return True, un_normal_count, un_orth_count


def check_procedures(vectors, csv_file):
    """
    check the Gram-Schmidt procedures for a set of vectors and save the results to a CSV file.
    :param vectors:
    :param csv_file:
    :return:
    """
    thresholds = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8]

    procedures = ["gs", 'mgs', 'sciQR', 're_orth', 'torch_qr']
    csv_results = []
    init_vectors = [v.clone().detach() for v in vectors]
    headers = [' ']
    headers += procedures
    csv_results.append(headers)
    for t in thresholds:
        thre_results = [str(t)]
        for method in procedures:
            counter = 0
            for vector in vectors:
                vectors = [v.clone().detach() for v in init_vectors]
                orth, un_normal_count, un_orth_count = check_gram_schmidt_procedure(
                    vector, threshold=t, print_info=False, method=method)
                if orth:
                    counter += 1
            thre_results.append(counter)

        print(thre_results)
        csv_results.append(thre_results)
    with open(csv_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(csv_results)

    data = np.array(csv_results, dtype=object)  # 使用object类型可以混合存储字符串和数字

    # 绘制图表
    fig, ax = plot_grouped_bars(data, figsize=(10, 6))
    plt.savefig(csv_file.replace('.csv', '.png'), dpi=300, bbox_inches='tight')


def generate_random_vectors(num_groups=1000, n=1, m=1, is_complex=False):
    """
    生成随机的torch向量组

    参数:
        num_groups: 生成的向量组数量，默认为1000
        n: 每个组中向量的数量
        m: 每个向量的维度
        is_complex: 是否生成复数向量，默认为False（生成实数向量）

    返回:
        形状为(num_groups, n, m)的torch张量，数据类型为complex64
    """
    # 生成实部，范围在[-1, 1)之间
    real_part = torch.rand((num_groups, n, m), dtype=torch.float32)

    if is_complex:
        # 生成虚部，范围在[-1, 1)之间
        imag_part = torch.rand((num_groups, n, m), dtype=torch.float32)
        # 组合实部和虚部形成复数张量，类型为complex64
        result = torch.complex(real_part, imag_part).to(torch.complex64)
    else:
        # 实数情况，虚部为0
        result = torch.rand((num_groups, n, m), dtype=torch.float32)
    return result


def plot_grouped_bars(data, figsize=(12, 8), bar_width=0.8, show_values=True):
    """
    绘制分组柱状图（已修正组和类的对应关系）

    参数:
        data: m×n的二维数组，第一行和第一列包含标题
        figsize: 图表大小
        bar_width: 每组柱状图的总宽度
        show_values: 是否在柱子上显示数值
    """
    # 提取数据维度
    m, n = data.shape  # m是组数+1（含标题行），n是类别数+1（含标题列）

    # 提取标题（已修正对应关系）
    class_titles = data[1:, 0]  # 第一列从第二个开始作为类别标题
    # group_titles = data[0, 1:]  # 第一行从第二个开始作为每组的标题
    group_titles = ['Gram-Schmidt', 'Modified Gram-Schmidt', 'Scipy-QR', 'Re-orthogonalization', 'PyTorch-QR']

    # 提取实际数据并转置，修正组和类的对应关系
    plot_data = data[1:, 1:].astype(float).T  # 转置后：每组包含多个类别

    # 创建画布
    fig, ax = plt.subplots(figsize=figsize)

    # 设置每组柱子的位置
    x = np.arange(len(class_titles))  # 横轴刻度位置（类别作为横轴）

    # 计算每个柱子的宽度
    individual_width = bar_width / (n-1)  # 除以类别数(n-1)

    # 绘制每组柱状图（已修正循环逻辑）
    for i in range(n-1):
        positions = x - bar_width/2 + individual_width/2 + i * individual_width
        bars = ax.bar(positions, plot_data[i], width=individual_width, label=str(group_titles[i]))

        # 在柱子上显示数值
        if show_values:
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(class_titles, rotation=45, ha='right')
    ax.set_xlabel('thresholds')
    ax.set_ylabel('orthonormal counts')
    ax.set_title('')

    if np.all(np.mod(plot_data, 1) == 0):
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))


    ax.legend(
        title=' ',
        ncol=len(group_titles),  # adjust number of columns in legend to fit all groups
        loc='upper center',
        bbox_to_anchor=(0.5, -0.1),  # put legend below the plot
        frameon=False  # remove legend frame
    )

    # 调整布局
    plt.tight_layout()

    return fig, ax

if __name__ == "__main__":
    for i in [2, 4, 8, 16, 32, 64]:
        # generate random vectors, real and complex
        vectors_real = generate_random_vectors(num_groups=1000, n=i, m=i, is_complex=False)
        vectors_complex = generate_random_vectors(num_groups=1000, n=i, m=i, is_complex=True)
        # check the procedures for real vectors
        check_procedures(vectors=vectors_real, csv_file=f"gram_schmidt_check_real_{i}*{i}_results.csv")
        # check the procedures for complex vectors
        check_procedures(vectors=vectors_complex, csv_file=f"gram_schmidt_check_complex_{i}*{i}_results.csv")
