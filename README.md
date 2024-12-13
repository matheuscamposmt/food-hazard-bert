# The Food Hazard Detection Challenge

Um projeto colaborativo para a [*SemEval 2025 Task 9: The Food Hazard Detection Challenge*](https://food-hazard-detection-semeval-2025.github.io/), utilizando modelos de linguagem natural avançados. Este repositório inclui implementações, experimentos e relatórios para a detecção de riscos alimentares com base em textos.

## Membros do Time
- **Matheus Campos**
- **Daniel Menezes**
- **Matheus Laureano**

## Objetivo
O objetivo deste projeto é desenvolver uma solução eficiente para detectar riscos alimentares com base em textos, empregando técnicas de **Transformers** e **Processamento de Linguagem Natural (NLP)**.  
A abordagem principal inclui:
- **Análise e pré-processamento de dados**: preparação do texto bruto e engenharia de características adicionais.
- **Treinamento e avaliação de modelos**: desenvolvimento de um modelo baseado no DeBERTaV2 e integração de características temporais.
- **Geração de insights**: análise dos resultados para obter informações úteis para o domínio da segurança alimentar.

---

## Estrutura do Repositório
```plaintext
├── data/                 # Dados brutos, pré-processados e scripts de preparação
├── notebooks/            # Notebooks Jupyter para experimentação e análise
├── models/               # Código para modelos (incluindo a integração de features adicionais)
├── configs/              # Configurações do modelo e treinamento
├── report/               # Relatório LaTeX colaborativo no Overleaf
├── results/              # Saída dos modelos, gráficos e visualizações
├── scripts/              # Scripts auxiliares para execução e análise
└── README.md             # Documentação do projeto (este arquivo)
```

---

## Arquitetura Implementada

A arquitetura implementada é baseada no modelo **DeBERTaV2**, com extensões que permitem integrar características adicionais (temporais) à cabeça de classificação. Os principais componentes incluem:

### **1. Modelo Base: DeBERTaV2**
- O modelo pré-treinado **DeBERTaV2** é usado para gerar embeddings robustas a partir de textos descritivos.
- Utilizamos a variante `microsoft/deberta-v3-small`, que equilibra desempenho e eficiência.

### **2. Cabeça de Classificação Personalizada**
A arquitetura da cabeça de classificação foi modificada para integrar informações textuais e características adicionais:
- **Saída do Modelo Base**: As embeddings do token `[CLS]` são extraídas como representações do texto.
- **Engenharia de Características**: Características temporais (e.g., data de ocorrência) são normalizadas e integradas.
- **Combinação de Representações**:
  - As embeddings textuais e as características adicionais são concatenadas.
  - A representação combinada é passada por uma camada linear para gerar logits.
- **Função de Perda**: Utilizamos `CrossEntropyLoss` para tarefas de classificação multiclasse.

### **3. Normalização de Características Temporais**
- Características temporais contínuas são normalizadas usando **Min-Max Scaling**.
- Para características periódicas (e.g., meses), foi implementada **codificação cíclica** com seno e cosseno para capturar a periodicidade.

### **4. Otimização e Regularização**
- Otimizador: `AdamW` para melhor adaptação à regularização L2.
- Scheduler: Decaimento linear com aquecimento para ajuste dinâmico da taxa de aprendizado.
- Dropout: Adicionado após a combinação das características para mitigar overfitting.

### Fluxo de Dados no Modelo:
1. **Entrada**:
   - Texto descritivo (tokenizado pelo `DeBERTaV2Tokenizer`).
   - Características adicionais (normalizadas e, se aplicável, codificadas ciclicamente).
2. **Processamento**:
   - O texto é transformado em embeddings pelo modelo DeBERTaV2.
   - As embeddings do texto são concatenadas com as características adicionais.
3. **Saída**:
   - Logits de classificação para cada categoria de risco alimentar.

A arquitetura é modular e facilmente extensível para incorporar novos tipos de características ou realizar ajustes.

---

## Principais Componentes
### Modelos Utilizados
O projeto emprega o **DeBERTaV2**, um modelo de linguagem avançado, com as seguintes personalizações:
- **Cabeçalho de classificação customizado**: Integração de **características temporais normalizadas** junto às embeddings do Transformer.
- **Treinamento supervisionado**: Otimização do modelo para a classificação de múltiplas categorias de riscos alimentares.
- **Normalização e Engenharia de Características**: Para melhorar a contribuição de variáveis temporais.

---

## Requisitos
Certifique-se de que as seguintes dependências estão instaladas:

### Bibliotecas Principais:
- Python 3.x
- `transformers`
- `torch`
- `scikit-learn`
- `pandas`
- `numpy`
- `jupyter`

### Ferramentas Adicionais:
- Overleaf (para colaboração no relatório)

### Instalação:
```bash
pip install -r requirements.txt
```

---

## Como Executar
### 1. Preparar o Conjunto de Dados:
Converta os dados brutos para um formato processado e integrado:
```bash
python scripts/preprocess.py --input data/raw --output data/processed
```

### 2. Analisar os Dados:
Explore os dados e obtenha insights iniciais usando Jupyter Notebooks:
```bash
jupyter notebook notebooks/analysis.ipynb
```

### 3. Treinar o Modelo:
Treine o modelo DeBERTaV2 com as características integradas:
```bash
python models/train.py --config configs/train_config.yaml
```

### 4. Avaliar o Modelo:
Calcule métricas de desempenho no conjunto de teste:
```bash
python models/evaluate.py --model models/best_model --data data/processed/test.csv
```

---

## Resultados
### Principais Descobertas:
- **Precisão**: [Adicione os valores das métricas]
- **Integração de características temporais**: Melhorou a acurácia do modelo em [X]% em relação à baseline.
- **Visualizações**: Consulte a pasta `results/` para gráficos e métricas detalhadas.

### Exemplos de Previsões:
- Entrada: `"Recall Notification: FSIS-033-94 - Sausage contaminated"`
  - Saída: `Categoria de Risco: Biological`
- Entrada: `"Plastic fragments in chicken breast - High risk"`
  - Saída: `Categoria de Risco: Foreign Body`

---

## Contribuições
Os membros do time trabalharam colaborativamente nas seguintes áreas:
- **Matheus Campos**: Engenharia de dados, desenvolvimento da arquitetura dos modelos.
- **Daniel Menezes**: Módulo de validação, visualização de resultados e estruturação do relatório.
- **Matheus Laureano**: Pré-processamento de dados, experimentação em notebooks e módulo de treinamento.

Contribuições detalhadas podem ser encontradas no relatório.
