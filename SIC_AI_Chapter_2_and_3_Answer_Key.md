# Samsung Innovation Campus — AI Course
## Chapter 2 and Chapter 3: Solved Question Bank

**Source:** The two uploaded Chapter 2 and Chapter 3 quiz documents.  
**How to use:** Python snippets can be run independently unless a preceding setup block is indicated. Numeric outputs are given wherever the documents supply the necessary data. The Chapter 3 questions refer to four CSV files that were **not attached** (`data_studentlist.csv`, `data_census.csv`, `data_sales.csv`, `data_coffeeshop.csv`); the supplied answers are executable Pandas expressions rather than invented dataset results. For the Boston-housing exercise, the linked data source in the Chapter 2 question is used.

---

# Chapter 2 — Linear Algebra and Differentiation

### Q1. Examples of zero vector, one vector, square, diagonal, identity and symmetric matrices

```python
import numpy as np

zero_vector     = np.zeros(3)               # [0. 0. 0.]
one_vector      = np.ones(3)                # [1. 1. 1.]
square_matrix   = np.array([[1, 2], [3, 4]])
diagonal_matrix = np.diag([2, 5, 7])
identity_matrix = np.eye(3)
symmetric_matrix = np.array([[2, 3], [3, 4]])

print(zero_vector, one_vector)
print(square_matrix, diagonal_matrix, identity_matrix, symmetric_matrix,
      sep='\n')
```

A **square** matrix has equal numbers of rows and columns. A **diagonal** matrix has zero entries off the main diagonal. An **identity** matrix has ones on its main diagonal and zeroes elsewhere. A **symmetric** matrix satisfies `A.T == A`.

### Q2. Rectangular, upper triangular, identity and skew-symmetric matrices

```python
rectangular = np.array([[1, 2, 3], [4, 5, 6]])  # 2 × 3
upper_triangular = np.array([[1, 2, 3], [0, 4, 5], [0, 0, 6]])
identity = np.eye(3, dtype=int)
skew_symmetric = np.array([[0, 2, -3], [-2, 0, 4], [3, -4, 0]])

print(np.array_equal(skew_symmetric.T, -skew_symmetric))  # True
```

A skew-symmetric matrix satisfies `A.T = -A` (and has zeros on the diagonal).

### Q3. Share prices and quantities in rupees

Prices `p = [1200000, 750000, 600000]`, quantities `n = [2, 5, 4]`.

```python
p = np.array([1200000, 750000, 600000])
n = np.array([2, 5, 4])
total = np.dot(p, n)
print(total)  # 8550000
```

**Answer:** `1,200,000 × 2 + 750,000 × 5 + 600,000 × 4 = ₹8,550,000`.

### Q4. Share prices and quantities in won

```python
p = np.array([1000000, 800000, 500000])
n = np.array([3, 4, 5])
print(p @ n)  # 8700000
```

**Answer:** `3,000,000 + 3,200,000 + 2,500,000 = 8,700,000 won`.

### Q5. Dot-product similarity of digit images stored in Z

```python
from sklearn.datasets import load_digits
Z = load_digits().data
similarity_first_tenth = np.dot(Z[0], Z[9])
all_similarities = Z @ Z.T
print(Z.shape)                     # (1797, 64)
print(similarity_first_tenth)       # 2807.0
print(all_similarities.shape)       # (1797, 1797)
```

**Answer 1:** `2807.0` for the first and tenth images (Python indices 0 and 9).  
**Answer 2:** `Z @ Z.T` is the Gram matrix of pairwise dot-product similarities. Entry `(i, j)` is the dot product of digit vectors `i` and `j`. A larger dot product does **not** necessarily mean greater angular similarity when vector lengths differ.

### Q6. Repeat for X and use efficient matrix multiplication

```python
from sklearn.datasets import load_digits
X = load_digits().data
print(X[0] @ X[9])  # 2807.0
S = X @ X.T
print(S.shape)       # (1797, 1797)
```

**Answer:** The first/tenth-image dot product is `2807.0`; the efficient all-pairs computation is `X @ X.T`. **Source clarification:** the question calls these images “MNIST,” but the provided code loads scikit-learn's **digits** dataset, not the separate MNIST dataset.

### Q7. Similarity matrix and comparison with nested loops

The question refers to “Z from Question 3,” but Z is defined in **Question 5**.

```python
S = Z @ Z.T
```

**Explanation:** NumPy uses optimized compiled matrix multiplication instead of running millions of interpreted Python-loop iterations. The matrix product still requires substantial arithmetic, but it is typically much faster. It calculates all pairwise dot products in one operation and returns a `1797 × 1797` matrix for the sample digits dataset.

### Q8. Inverse matrix

The matrix shown in the document is

$$
A = \begin{bmatrix}2&0\\0&1\end{bmatrix}.
$$

Its inverse is

$$
\boxed{A^{-1} = \begin{bmatrix}\tfrac12&0\\0&1\end{bmatrix}}.
$$

```python
A = np.array([[2, 0], [0, 1]])
print(np.linalg.inv(A))  # [[0.5, 0. ], [0. , 1. ]]
```

### Q9. Boston house-price model using four observations

The question specifies `A = X[:4, [0, 4, 5, 6]]`, meaning CRIM, NOX, RM and AGE, and `b = y[:4]`. The **four-observation exact system** is `A @ x = b`, where `x` contains four weights. The original Boston data are retrieved using the URL already present in the question.

```python
import pandas as pd
import numpy as np
url = 'https://lib.stat.cmu.edu/datasets/boston'
raw = pd.read_csv(url, sep=r'\s+', skiprows=22, header=None)
X = np.hstack([raw.values[::2, :], raw.values[1::2, :2]])
y = raw.values[1::2, 2]
A = X[:4, [0, 4, 5, 6]]
b = y[:4]
x = np.linalg.solve(A, b)
print(dict(zip(['CRIM', 'NOX', 'RM', 'AGE'], x)))
```

**Calculated weights (approximately):**

| Feature | Weight | Interpretation in this four-row model |
|---|---:|---|
| CRIM | -312.7100 | Higher crime rate is associated with lower predicted price, other model inputs held fixed. |
| NOX | -115.1939 | Higher pollution concentration is associated with lower predicted price. |
| RM | +14.4996 | More rooms are associated with higher predicted price. |
| AGE | -0.1133 | A larger percentage of older housing is associated with lower predicted price. |

**Important:** These numerical weights fit only **four records** and are unstable, so they should not be treated as reliable population estimates or proof of causation.

### Q10. Boston house prices using least squares over the full dataset

The question uses all 13 input columns, with model `X @ w ≈ y` **without an explicit intercept**. After loading `X` and `y` exactly as in Q9, use:

```python
feature_names = [
    'CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE',
    'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT'
]
w = np.linalg.lstsq(X, y, rcond=None)[0]
print(pd.Series(w, index=feature_names).round(4))
for feature, weight in zip(feature_names, w):
    relation = 'positive' if weight > 0 else 'negative' if weight < 0 else 'zero'
    print(f'{feature}: {weight:+.4f} ({relation} association with predicted price)')
```

**Answer to part (1):** Print the fitted coefficients and use their signs to interpret each predictor while holding the other predictors fixed: positive coefficient → higher predicted house value; negative coefficient → lower predicted house value. The features to report are **CRIM, ZN, INDUS, CHAS, NOX, RM, AGE, DIS, RAD, TAX, PTRATIO, B and LSTAT**. The exact full-data weights require running the linked dataset code. They are not explicitly printed in the uploaded QB. Avoid treating coefficient magnitude as directly comparable across differently scaled features. The dataset's `B` variable is a historically constructed transformation, **not** simply the proportion of Black residents as the QB states.

**Answer to part (2):** Q9 solves four equations using **four observations and four features**, and fits them exactly if the matrix is invertible. Q10 uses **506 observations and 13 features** and minimizes the sum of squared errors; it does not generally match every observation exactly. Coefficients differ because both the sample and feature set differ. Also, both supplied models omit an intercept.

### Q11. Distances between users a, b and c

The source's equation objects specify `a = (4, 5, 2, 2)`, `b = (4, 0, 2, 0)` and `c = (2, 2, 0, 1)`.

For Euclidean distance, `d(u,v) = sqrt(sum((u-v)^2))`. For cosine **distance**, `d_cos(u,v) = 1 - (u·v)/(||u|| ||v||)`.

| Pair | Euclidean distance | Cosine similarity | Cosine distance |
|---|---:|---:|---:|
| a, b | √29 ≈ 5.3852 | 0.6389 | 0.3611 |
| a, c | √18 ≈ 4.2426 | 0.9524 | 0.0476 |
| b, c | √13 ≈ 3.6056 | 0.5963 | 0.4037 |

**Euclidean:** closest `b–c`; farthest `a–b`.  
**Cosine distance:** closest `a–c`; farthest `b–c`.

```python
a = np.array([4, 5, 2, 2]); b = np.array([4, 0, 2, 0]); c = np.array([2, 2, 0, 1])
from itertools import combinations
for (n1, u), (n2, v) in combinations([('a', a), ('b', b), ('c', c)], 2):
    print(n1, n2, np.linalg.norm(u-v),
          1 - (u @ v)/(np.linalg.norm(u)*np.linalg.norm(v)))
```

### Q12. Eigenvalues using the characteristic equation

$$
D = \begin{bmatrix}2&1\\1&2\end{bmatrix}, \qquad
\det(D-\lambda I) = (2-\lambda)^2-1 = 0.
$$

`(2 − λ − 1)(2 − λ + 1) = (1 − λ)(3 − λ) = 0`.

**Answer:** `λ₁ = 3`, `λ₂ = 1`.

```python
print(np.linalg.eigvals([[2, 1], [1, 2]]))  # [3. 1.]
```

### Q13. Evaluate log(12) using log(2) ≈ 0.69 and log(3) ≈ 1.10

$$
\log(12)=\log(3\cdot 2^2)=\log3+2\log2=1.10+2(0.69)=\boxed{2.48}.
$$

### Q14. Differentiate the three displayed functions

1. `f(x) = kx³ + ax² + b` → **`f'(x) = 3kx² + 2ax`**.
2. `f(x) = ln(ax+b)` → **`f'(x) = a/(ax+b)`** (where defined).
3. `f(x) = exp(kx)` → **`f'(x) = k exp(kx)`**.

### Q15. Inverse of the logistic function

From `y = σ(x) = 1/(1 + e^(−x))`, rearrange to `1/y − 1 = e^(−x)` and take logarithms.

**Answer:** `σ⁻¹(y) = ln(y/(1−y))`, with `0 < y < 1`.

### Q16. Differentiate the expressions involving k, a, b

The source's equation objects show:

1. `f(x) = x³ − 1` → **`f'(x) = 3x²`**.
2. `f(x) = log(x² − 3k)` → **`f'(x) = 2x/(x² − 3k)`** (assuming natural logarithm).
3. `f(x) = exp((ax)^b)` → **`f'(x) = ab(ax)^(b−1) exp((ax)^b)`**.

### Q17. SymPy partial derivatives of f(x,y) = exp(2x+3y)

Let `E = exp(2x+3y)`.

| Derivative | Answer |
|---|---|
| fx | `2E` |
| fy | `3E` |
| fxx | `4E` |
| fxy | `6E` |
| fyx | `6E` |
| fyy | `9E` |

```python
import sympy as sp
x, y = sp.symbols('x y')
f = sp.exp(2*x + 3*y)
print('fx =', sp.diff(f, x))
print('fy =', sp.diff(f, y))
print('fxx =', sp.diff(f, x, 2))
print('fxy =', sp.diff(f, x, y))
print('fyx =', sp.diff(f, y, x))
print('fyy =', sp.diff(f, y, 2))
```

### Q18. Partial derivatives of f(x,y) = exp(x² + 2y²)

Let `E = exp(x² + 2y²)`.

| Derivative | Answer |
|---|---|
| fx | `2xE` |
| fy | `4yE` |
| fxx | `(2 + 4x²)E` |
| fxy | `8xyE` |
| fyx | `8xyE` |
| fyy | `(4 + 16y²)E` |

### Q19. Differentiate Q16's functions using SymPy

```python
import sympy as sp
x, k, a, b = sp.symbols('x k a b')
functions = [x**3 - 1, sp.log(x**2 - 3*k), sp.exp((a*x)**b)]
for i, f in enumerate(functions, start=1):
    print(i, ':', sp.diff(f, x))
```

**Outputs:** `3*x**2`; `2*x/(x**2 - 3*k)`; `a*b*(a*x)**(b-1)*exp((a*x)**b)` (algebraically equivalent SymPy forms are valid).

### Q20. SymPy first and second partial derivatives

The function is identical to Q18: `f(x,y) = exp(x²+2y²)`. The six answers are the same as Q18.

```python
x, y = sp.symbols('x y')
f = sp.exp(x**2 + 2*y**2)
for name, expression in {
    'fx': sp.diff(f, x), 'fy': sp.diff(f, y),
    'fxx': sp.diff(f, x, 2), 'fxy': sp.diff(f, x, y),
    'fyx': sp.diff(f, y, x), 'fyy': sp.diff(f, y, 2)
}.items():
    print(name, '=', sp.factor(expression))
```

### Q21. Euclidean distances between X, Y and Z

`X = (4,2,5,3)`, `Y = (5,3,4,2)`, `Z = (1,2,2,5)`.

| Pair | Euclidean distance |
|---|---:|
| X–Y | `√4 = 2.0000` |
| X–Z | `√22 ≈ 4.6904` |
| Y–Z | `√30 ≈ 5.4772` |

**Closest:** `X–Y`. **Farthest:** `Y–Z`.

### Q22. Cosine similarities using Q21's vectors

| Pair | Dot product | Cosine similarity |
|---|---:|---:|
| X–Y | 52 | `52/54 = 0.9630` |
| X–Z | 33 | `33/√(54×34) ≈ 0.7702` |
| Y–Z | 29 | `29/√(54×34) ≈ 0.6768` |

**Most similar:** `X–Y`. **Least similar:** `Y–Z`.

```python
X = np.array([4, 2, 5, 3]); Y = np.array([5, 3, 4, 2]); Z = np.array([1, 2, 2, 5])
for (a, u), (b, v) in combinations([('X', X), ('Y', Y), ('Z', Z)], 2):
    print(a+b, 'Euclidean:', np.linalg.norm(u-v),
          'cosine similarity:', (u @ v)/(np.linalg.norm(u)*np.linalg.norm(v)))
```

---

# Chapter 3 — Pandas and NumPy

**Dataset reminder:** The Chapter 3 QB does not include its input CSVs. All expressions below assume the same column names used by the associated course CSV files. For the student file, the common schema uses **lowercase** column names: `gender`, `age`, `grade`, `absence`, `bloodtype`, `height`, `weight`. Run the initial `read_csv` statement for each section before the individual answers. The source quiz repeats item numbers on its third page; the first section below uses continuous numbering 1–15 so that every item is covered.

## Q1. Queries on `data_studentlist.csv` (15 subquestions)

```python
import pandas as pd
import numpy as np
df = pd.read_csv('data_studentlist.csv')
```

1. **Average height of male students**
   ```python
   df.loc[df['gender'] == 'M', 'height'].mean()
   ```
2. **Average age of female students**
   ```python
   df.loc[df['gender'] == 'F', 'age'].mean()
   ```
3. **Average height of female students**
   ```python
   df.loc[df['gender'] == 'F', 'height'].mean()
   ```
4. **Average weight of male students**
   ```python
   df.loc[df['gender'] == 'M', 'weight'].mean()
   ```
5. **Minimum height among blood type B students**
   ```python
   df.loc[df['bloodtype'] == 'B', 'height'].min()
   ```
6. **Average weight of female students**
   ```python
   df.loc[df['gender'] == 'F', 'weight'].mean()
   ```
7. **Tallest male student (height)**
   ```python
   df.loc[df['gender'] == 'M', 'height'].max()
   ```
8. **Shortest female student (height)**
   ```python
   df.loc[df['gender'] == 'F', 'height'].min()
   ```
9. **Tallest student with grade ≥ 3.5**
   ```python
   df.loc[df['grade'] >= 3.5, 'height'].max()
   ```
10. **Lightest male taller than 175 cm**
    ```python
    df.loc[(df['gender'] == 'M') & (df['height'] > 175), 'weight'].min()
    ```
11. **Heaviest female shorter than 160 cm**
    ```python
    df.loc[(df['gender'] == 'F') & (df['height'] < 160), 'weight'].max()
    ```
12. **Average grade with no absence ('N')**
    ```python
    df.loc[df['absence'] == 'N', 'grade'].mean()
    ```
13. **Average grade with absence ('Y')**
    ```python
    df.loc[df['absence'] == 'Y', 'grade'].mean()
    ```
14. **Average height of students with blood type A or AB**
    ```python
    df.loc[df['bloodtype'].isin(['A', 'AB']), 'height'].mean()
    ```
15. **Average height of male students with blood type A or AB**
    ```python
    df.loc[(df['gender'] == 'M') & df['bloodtype'].isin(['A', 'AB']), 'height'].mean()
    ```
16. **Average age of absentees with grade ≥ 3**
    ```python
    df.loc[(df['absence'] == 'Y') & (df['grade'] >= 3), 'age'].mean()
    ```

**Numbering note:** The original QB repeats question numbers on its following page. There are **16 distinct student-data prompts** here, numbered consecutively to avoid ambiguity.

## Q2. Queries on `data_census.csv`

```python
import pandas as pd
df = pd.read_csv('data_census.csv')
```

1. **Total population with ProvinceCode 115 or 116**
   ```python
   df.loc[df['ProvinceCode'].isin([115, 116]), 'Population'].sum()
   ```
2. **Average population in places with more males (`GenderRatio > 1`)**
   ```python
   df.loc[df['GenderRatio'] > 1, 'Population'].mean()
   ```
3. **Places with more males and fewer than 2 persons per household**
   ```python
   df.loc[(df['GenderRatio'] > 1) & (df['PersonsHousehold'] < 2)]
   ```
4. **Sort Households ascending; first 10 rows**
   ```python
   df.sort_values('Households').head(10)
   ```
5. **Sort Population descending; first 5 rows**
   ```python
   df.sort_values('Population', ascending=False).head(5)
   ```
6. **List places with average household size > 3**
   ```python
   df.loc[df['PersonsHousehold'] > 3,
          ['ProvinceCode', 'CityCode', 'PersonsHousehold']]
   ```

## Q3. Queries on `data_sales.csv`

```python
import pandas as pd
df = pd.read_csv('data_sales.csv')
# In this dataset, assume the columns are Region, Item, Units, UnitPrice.
# Inspect df.columns if your CSV version uses ItemType instead of Item.
item_col = 'Item' if 'Item' in df.columns else 'ItemType'
```

1. **New variable Amount = UnitPrice × Units**
   ```python
   df['Amount'] = df['UnitPrice'] * df['Units']
   ```
2. **Average UnitPrice by Region using groupby**
   ```python
   df.groupby('Region')['UnitPrice'].mean()
   ```
3. **Average UnitPrice by Region using pivot_table**
   ```python
   df.pivot_table(index='Region', values='UnitPrice', aggfunc='mean')
   ```
4. **Average UnitPrice and Units by Region, groupby**
   ```python
   df.groupby('Region')[['UnitPrice', 'Units']].mean()
   ```
5. **Average UnitPrice and Units by Region, pivot_table**
   ```python
   df.pivot_table(index='Region', values=['UnitPrice', 'Units'], aggfunc='mean')
   ```
6. **Total Units by Region and item type, pivot_table, zeros for missing values**
   ```python
   df.pivot_table(index='Region', columns=item_col, values='Units',
                  aggfunc='sum', fill_value=0)
   ```
7. **Total Amount by Region and item type**
   ```python
   df.pivot_table(index='Region', columns=item_col, values='Amount',
                  aggfunc='sum', fill_value=0)
   ```
   Run the **Amount** assignment in subquestion 1 first.
8. **Maximum UnitPrice by Region using groupby**
   ```python
   df.groupby('Region')['UnitPrice'].max()
   ```

## Q4. Coffee-shop analysis and multiple-choice questions

### Q4(1). Frequency table by starting year and line plot, 1997–2014

```python
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('data_coffeeshop.csv', na_values=[' '])
df['yearOfStart'] = pd.to_numeric(df['yearOfStart'], errors='coerce')
frequency = df['yearOfStart'].value_counts().sort_index()
print(frequency)
plt.figure(figsize=(9, 4))
plt.plot(frequency.index, frequency.values, marker='o')
plt.xlim(1997, 2014)
plt.xlabel('Year started')
plt.ylabel('Number of coffee shops started')
plt.title('Coffee shops by opening year')
plt.grid(alpha=0.3)
plt.show()
```

### Q4(2). Separate In/Out businesses and plot overlapping trends

Since the business-status column name is not specified in the QB, the following finds the column whose nonempty values include both `In` and `Out`.

```python
status_col = next(
    col for col in df.columns
    if {'In', 'Out'}.issubset(set(df[col].dropna().astype(str).str.strip()))
)
status = df[status_col].astype(str).str.strip()
counts = (df.assign(_status=status)
          .groupby(['yearOfStart', '_status']).size()
          .unstack(fill_value=0).sort_index())
fig, ax = plt.subplots(figsize=(9, 4))
for label in ['In', 'Out']:
    if label in counts.columns:
        ax.plot(counts.index, counts[label], label=label, marker='o')
ax.set_xlim(1997, 2014)
ax.set_xlabel('Year started')
ax.set_ylabel('Number of coffee shops started')
ax.set_title('Coffee-shop openings: still in vs out of business')
ax.legend()
ax.grid(alpha=0.3)
plt.show()
```

### Q4(3) and Q4(4). Interpret the plots

The QB supplies **no coffee-shop records or plot outputs**, so a specific claim of increase/decrease is not supported by the two uploads. Examine whether yearly counts trend upward or downward across 1997–2014. Compare the `In` and `Out` lines: the first counts shops from a starting-year cohort that remain in business; the second counts those that have closed. **Caution:** a chart of shops *started in each year* is not the same thing as the *total number currently operating*, and a younger cohort naturally has had less time to close.

### Q4(5)–Q4(22). Multiple-choice answer key

| Subquestion | Correct choice | Answer / explanation |
|---|---|---|
| 5. Reshape into 2 rows × 3 columns | **①** | `np.reshape(arr, (2, 3))` (requires exactly six elements). |
| 6. Shuffle array | **④** | `np.random.shuffle(arr)` shuffles in place; `np.random.shuffle()` names the function but needs its array argument. |
| 7. Pandas mean of Height | **⑧** | `df['Height'].mean()` (capitalization matches the MCQ). |
| 8. Create a NumPy array | **⑪** | `np.array([1,2,3,4,5])`. |
| 9. 2-D structure | **②** | `([1,2,3], [4,5,6])` is the intended 2-D arrangement. Strictly, a NumPy 2-D array is `np.array([[1,2,3],[4,5,6]])`. |
| 10. Number of dimensions | **②** | `np.ndim(arr)`; `arr.ndim` also works. The MCQ omits the required array argument. |
| 11. First element | **①** | `print(myArr[0])`. |
| 12. Print 8 from 2-D array | **③** | `print(arr[1, 2])`. |
| 13. Slice `[3,4,5]` | **②** | `print(arr[2:5])`. |
| 14. Join arrays | **②** | `np.concatenate((arr1, arr2))`. |
| 15. Indices of value 4 | **③** | `np.where(arr == 4)` → `(array([1, 3, 5, 6]),)`. |
| 16. Random integer 0–99 | **③** | `random.randint(100)` when `random` is `numpy.random`; to **include 100**, use `np.random.randint(0,101)`. |
| 17. 1000 values ~ Normal(50, 0.2²) | **①** | `random.normal(size=1000, loc=50, scale=0.2)`. |
| 18. Elementwise addition | **①** | `np.add(arr1, arr2)`. |
| 19. Elementwise subtraction | **None of the listed choices** | Correct syntax is `np.subtract(arr1, arr2)`; `np.sub`, `np.minus`, and `np.substract` are not NumPy functions. |
| 20. NumPy rounding methods | **③** | All three exist (`np.fix`, `np.trunc`, `np.around`), but `fix` and `trunc` round toward zero rather than to nearest. |
| 21. `np.cumsum([1,2,3])` | **④** | `[1 3 6]`. |
| 22. Create Pandas Series | **①** | `pd.Series(mylist)`. |

---

## Final verification notes

- **Chapter 2 Q5–Q6:** The first–tenth dot product (`2807`) is computed using the exact `sklearn.datasets.load_digits()` code in the QB.
- **Chapter 2 Q9–Q10:** The Boston example uses the data URL supplied in the document. Q9 numerical coefficients are calculated from the first four rows. Q10 includes the complete method to calculate its full-data weight vector and print its signs; no unsupported coefficient table is invented.
- **Chapter 3 Q1–Q4:** Exact dataset-specific summaries and coffee-shop trends cannot be verified without the four CSV files. All requested filters, aggregations, tables and plots are provided.
- **Source typos flagged explicitly:** Chapter 2 Q7's Z reference; the “MNIST” label in Q6; the duplicated numbering in Chapter 3 Q1; Chapter 3 Q4(19) lacks a correct option. 
