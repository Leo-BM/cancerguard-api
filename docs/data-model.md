# Data Model — CancerGuard API

← [Voltar ao índice](./README.md)

---

## 1. Dataset de Origem

| Atributo | Valor |
|---|---|
| Nome | Breast Cancer Wisconsin (Diagnostic) |
| Fonte | UCI Machine Learning Repository |
| Acesso via | `sklearn.datasets.load_breast_cancer` |
| Total de amostras | 569 |
| Features | 30 numéricas contínuas |
| Classes | Maligno (212) / Benigno (357) |
| Proporção de classes | ~37% maligno / 63% benigno |

---

## 2. Features

As 30 features são extraídas de imagens digitalizadas de aspirados por agulha fina (FNA) de massas mamárias. Cada feature é calculada para o núcleo celular presente na imagem.

As features são organizadas em 3 grupos de 10:

### Grupo 1 — Mean (média)

| Feature | Descrição |
|---|---|
| `mean_radius` | Média das distâncias do centro às bordas do perímetro |
| `mean_texture` | Desvio padrão dos valores de escala de cinza |
| `mean_perimeter` | Perímetro médio |
| `mean_area` | Área média |
| `mean_smoothness` | Variação local nos comprimentos do raio |
| `mean_compactness` | Perímetro² / área − 1.0 |
| `mean_concavity` | Severidade das partes côncavas do contorno |
| `mean_concave_points` | Número de partes côncavas do contorno |
| `mean_symmetry` | Simetria |
| `mean_fractal_dimension` | Aproximação da "coastline" fractal − 1 |

### Grupo 2 — SE (erro padrão)

| Feature | Descrição |
|---|---|
| `radius_se` | Erro padrão do raio |
| `texture_se` | Erro padrão da textura |
| `perimeter_se` | Erro padrão do perímetro |
| `area_se` | Erro padrão da área |
| `smoothness_se` | Erro padrão da suavidade |
| `compactness_se` | Erro padrão da compacidade |
| `concavity_se` | Erro padrão da concavidade |
| `concave_points_se` | Erro padrão dos pontos côncavos |
| `symmetry_se` | Erro padrão da simetria |
| `fractal_dimension_se` | Erro padrão da dimensão fractal |

### Grupo 3 — Worst (pior valor)

| Feature | Descrição |
|---|---|
| `worst_radius` | Maior raio médio das 3 maiores células |
| `worst_texture` | Maior textura média |
| `worst_perimeter` | Maior perímetro médio |
| `worst_area` | Maior área média |
| `worst_smoothness` | Maior suavidade |
| `worst_compactness` | Maior compacidade |
| `worst_concavity` | Maior concavidade |
| `worst_concave_points` | Maior número de pontos côncavos |
| `worst_symmetry` | Maior simetria |
| `worst_fractal_dimension` | Maior dimensão fractal |

---

## 3. Pré-processamento

### StandardScaler

Aplicado sobre todas as 30 features antes do treinamento:

$$z = \frac{x - \mu}{\sigma}$$

- **Motivação:** O SVM com kernel RBF é sensível à escala das features. Features com magnitude maior (ex: `mean_area` ~ 654) dominariam as de menor magnitude (ex: `mean_smoothness` ~ 0.09) sem normalização.
- **Artefato:** O scaler ajustado é salvo como `scaler.joblib` no MLflow junto ao modelo, garantindo que **exatamente o mesmo transformador** seja usado em treino e em predição.

### Divisão treino/teste

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y       # mantém proporção de classes em ambos os splits
)
```

---

## 4. Target (Label)

| Valor numérico | Classe | Significado |
|---|---|---|
| `0` | `malignant` | Tumor maligno |
| `1` | `benign` | Tumor benigno |

> Atenção: o scikit-learn usa 0=maligno e 1=benigno neste dataset. A API converte para os strings `"malignant"` / `"benign"` na camada de output.

---

## 5. Modelo Serializado

| Atributo | Valor |
|---|---|
| Algoritmo | SVM com kernel RBF |
| Hiperparâmetros | `C=1.0`, `gamma="scale"`, `probability=True` |
| Formato de serialização | MLflow Model (wrapper sobre joblib) |
| Artefatos associados | `scaler.joblib` |
| Localização em produção | MLflow Model Registry, stage = `"Production"` |

### Por que SVM RBF?

O kernel RBF transforma os dados em um espaço de dimensão infinita implicitamente, capturando fronteiras de decisão não-lineares sem necessidade de feature engineering adicional. Comparação com os demais kernels testados:

| Modelo | Acurácia | Recall (Maligno) |
|---|---|---|
| SVM Linear | 95.1% | 87.9% |
| SVM Polinomial | 97.9% | 94.8% |
| **SVM RBF** ✓ | **98.2%** | **96.8%** |

### Por que Recall é a métrica de decisão?

$$\text{Recall} = \frac{TP}{TP + FN}$$

Em diagnóstico oncológico, um **Falso Negativo** (classificar um tumor maligno como benigno) pode resultar em atraso no tratamento com consequências graves. Um **Falso Positivo** gera ansiedade e exames adicionais — indesejável, mas recuperável. Portanto, o Recall sobre a classe `malignant` é a métrica prioritária de seleção do modelo.
