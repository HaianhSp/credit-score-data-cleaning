"""
Hàm dùng chung cho các notebook: kiểu biểu đồ, màu của từng nhãn, in code từ module ra màn hình,
và vài hàm thống kê (tìm miền hợp lệ, độ lớn hiệu ứng).
"""
import ast
import inspect
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

LABELS = ['Poor', 'Standard', 'Good']                       # thứ tự từ thấp đến cao
LABEL_RANK = {'Poor': 0, 'Standard': 1, 'Good': 2}
COLORS = {'Poor': '#c0392b', 'Standard': '#f39c12', 'Good': '#27ae60'}


def setup_style():
    """Cài kiểu biểu đồ và cách hiển thị bảng cho cả notebook."""
    plt.rcParams.update({'figure.dpi': 110, 'axes.spines.top': False, 'axes.spines.right': False, 'font.size': 10})
    pd.set_option('display.max_columns', 50)
    pd.set_option('display.width', 170)
    pd.set_option('display.max_colwidth', 60)


def show_code(module, *names):
    """In ra code của các hàm hoặc hằng số (theo tên) trong một module.
    Dùng ở notebook để người đọc thấy đúng đoạn code đang chạy mà không phải mở file trong src/."""
    source = inspect.getsource(module)
    lines = source.splitlines()
    found = {}
    for node in ast.parse(source).body:
        if isinstance(node, ast.FunctionDef):
            name = node.name
        elif isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
        else:
            continue
        start = node.lineno - 1
        while start > 0 and lines[start - 1].lstrip().startswith('#') and not lines[start - 1].startswith('# ---'):
            start -= 1                                                    # lấy luôn phần comment ngay phía trên
        found[name] = '\n'.join(lines[start:node.end_lineno])
    missing = [n for n in names if n not in found]
    if missing:
        raise KeyError(f'Không tìm thấy trong {module.__name__}: {missing}')
    print('\n\n'.join(found[n] for n in names))


def body_run(counts, min_count=50):
    """Tìm "thân" của một phân phối. `counts` là Series: số dòng của từng giá trị nguyên (index là giá trị).
    Thân là đoạn giá trị liền nhau, mỗi giá trị xuất hiện ít nhất `min_count` lần và chứa giá trị phổ biến nhất.
    Trả về (giá trị nhỏ nhất, giá trị lớn nhất) của đoạn đó."""
    frequent = set(counts[counts >= min_count].index)
    low = high = counts.idxmax()
    while low - 1 in frequent:
        low -= 1
    while high + 1 in frequent:
        high += 1
    return low, high


def effect_numeric(d, cols, target='Credit_Score'):
    """Mức liên hệ của từng cột số với nhãn ba lớp. Bỏ qua ô NaN của cột đó.
    - epsilon^2 từ kiểm định Kruskal-Wallis (H / (n - 1)): cho biết ba nhóm nhãn tách nhau bao nhiêu
    - Spearman với nhãn xếp thứ tự Poor < Standard < Good"""
    y = d[target]
    rank = y.map(LABEL_RANK)
    rows = []
    for col in cols:
        has_value = d[col].notna()
        groups = [d.loc[has_value & (y == k), col] for k in LABELS]
        h_stat, _ = stats.kruskal(*groups)
        n = has_value.sum()
        rows.append({'đặc trưng': col,
                     'ε² (Kruskal)': h_stat / (n - 1),
                     'Spearman với nhãn': stats.spearmanr(d.loc[has_value, col], rank[has_value])[0],
                     'n hợp lệ': int(n)})
    return pd.DataFrame(rows).set_index('đặc trưng')


def cramers_v(a, b):
    """Hệ số Cramér's V giữa hai biến phân loại (0 là độc lập, 1 là phụ thuộc hoàn toàn)."""
    table = pd.crosstab(a, b)
    chi2 = stats.chi2_contingency(table)[0]
    n = table.values.sum()
    return np.sqrt(chi2 / (n * (min(table.shape) - 1)))
