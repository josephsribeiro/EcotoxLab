# EcotoxLab ⚗

**Análise de Toxicidade Aquática – CL₅₀ / TL₅₀ com modelos Probit, Logit e Spearman‑Kärber**  
*Streamlit + SciPy + Matplotlib*

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://ecotoxlab.streamlit.app/)

---

## 📖 Sobre o programa

O **EcotoxLab** é uma aplicação web interativa para cálculo de **Concentração Letal Mediana (CL₅₀)** e **Tempo Letal Mediano (TL₅₀)** a partir de dados de mortalidade em ensaios toxicológicos aquáticos.

A ferramenta implementa os métodos estatísticos consolidados no pacote **R {ecotox}** (Hlina et al., 2021) e permite:

- Escolha entre **LC (concentração letal)** e **LT (tempo letal)**
- Modelos paramétricos: **Probit** e **Logit** (GLM binomial)
- Modelo não‑paramétrico: **Spearman‑Kärber** (apenas LC)
- Correção de Abbott para mortalidade no controle
- Teste de bondade de ajuste (χ² de Pearson) com p‑valor
- Gráfico dose‑resposta totalmente personalizável (cores, fontes, elementos)
- Exportação dos resultados (JPG, SVG, CSV)

---

## ✨ Funcionalidades principais

| Módulo | Descrição |
|--------|------------|
| **Modo LC** | Concentração fixa × mortalidade (múltiplas doses + controle) |
| **Modo LT** | Tempo fixo de exposição × mortalidade acumulada (leituras temporais) |
| **Correção de Abbott** | Ajuste da mortalidade observada pela mortalidade do controle |
| **Modelos paramétricos** | Probit e Logit via regressão binomial (GLM) – estima CL₅₀, slope, IC95% |
| **Spearman‑Kärber** | Método não‑paramétrico para CL₅₀ (apenas LC) |
| **Teste χ² de Pearson** | Avaliação do ajuste do modelo (p‑valor > 0,05 indica bom ajuste) |
| **Personalização do gráfico** | Curva, ICs, pontos, rótulos, cores, fontes, fundo escuro/claro |
| **Exportação** | Gráfico em JPG (300 DPI) e SVG (vetorial) + tabela de resultados em CSV |

---

## 🧪 Métodos estatísticos implementados

### 1. Regressão binomial (GLM) – Probit e Logit
- **Função `calc_glm()`**  
  Ajusta um modelo linear generalizado com ligação **probit** (`ndtr`) ou **logit** (`logit`).  
  - Variável independente: `log10(concentração)` ou `log10(tempo)`  
  - Variável dependente: proporção de mortos (com pesos pelo número de indivíduos)  
  - Estimação por máxima verossimilhança (iteração reweighted least squares)  
  - Retorna: CL₅₀ (ou TL₅₀), slope, intercepto, IC95% (método Delta), z‑value, variância do log(CL), χ² de Pearson, p‑valor GOF.

### 2. Spearman‑Kärber (apenas LC)
- **Função `calc_spearman_karber()`**  
  Método não‑paramétrico para cálculo da CL₅₀ (Wheeler et al., 2006).  
  - Utiliza as doses logaritmizadas e mortalidades corrigidas por Abbott.  
  - Calcula a área sob a curva de mortalidade.  
  - Estima a variância pelo método de Thompson (1947).  
  - Retorna CL₅₀, IC95% (aproximação log‑normal), χ² de Pearson e p‑valor.

### 3. Correção de Abbott
- **Função `abbott(p, ctrl)`**  
  Ajusta a mortalidade observada (`p`) pela mortalidade do controle (`ctrl`):  
  `p_corrigido = (p - ctrl) / (1 - ctrl)`.  
  Aplicada automaticamente nas doses > 0.

### 4. Teste de bondade de ajuste (GOF)
- **Função `chi_sq_pval(chi2, df)`**  
  Calcula o p‑valor do χ² de Pearson para o modelo ajustado.  
  - **Interpretação**: `p > 0,05` → modelo adequado aos dados (não rejeita H₀).  
  - `p ≤ 0,05` → ajuste ruim, resultados devem ser interpretados com cautela.

---

## 🚀 Como executar

### 1. Pré‑requisitos
Python 3.8 ou superior

### 2. Instalar dependências

```bash
pip install streamlit numpy pandas scipy matplotlib
