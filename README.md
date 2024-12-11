# The Food Hazard Detection Challenge

Um projeto colaborativo explorando a *SemEval 2025 Task 9: The Food Hazard Detection Challenge* utilizando modelos de linguagem avançados. Este repositório inclui a implementação, dados e relatório para a tarefa selecionada.

## Membros do Time
- **Matheus Campos**
- **Daniel Menezes**
- **Matheus Laureano**

## Objetivo
O objetivo deste projeto é desenvolver uma solução para a detecção de riscos alimentares com base em textos, utilizando técnicas de processamento de linguagem natural de ponta. Isso envolve:
- Preparação e análise dos dados.
- Treinamento e avaliação do modelo.
- Relatar resultados e descobertas em um formato estruturado.

## Estrutura
```
├── data/                 # Conjunto de dados e scripts de pré-processamento
├── notebooks/            # Notebooks Jupyter para exploração e experimentos
├── models/               # Código para treinamento e avaliação do modelo
├── report/               # Fonte LaTeX no Overleaf para o relatório
├── results/              # Métricas de saída e visualizações
└── README.md             # Documentação do projeto (este arquivo)
```

## Requisitos
- Python 3.x
- Bibliotecas: `transformers`, `scikit-learn`, `pandas`, `numpy`, `jupyter`
- Ferramentas adicionais: Overleaf para colaboração no relatório

Instale as dependências:
```bash
pip install -r requirements.txt
```

## Como Executar
1. Prepare o conjunto de dados:
   ```bash
   python preprocess.py --input data/raw --output data/processed
   ```
2. Explore os dados e visualize os resultados com notebooks Jupyter:
   ```bash
   jupyter notebook notebooks/analysis.ipynb
   ```
3. Treine o modelo:
   ```bash
   python train.py --config configs/train_config.yaml
   ```
4. Avalie os resultados:
   ```bash
   python evaluate.py --model models/best_model --data data/processed/test.csv
   ```

## Relatório
O relatório final é escrito de forma colaborativa no Overleaf e segue o [Template de Procedimentos ACL 2023](https://www.overleaf.com/latex/templates/acl-2023-proceedings-template/qjdgcrdwcnwp). Ele inclui:
- Introdução
- Trabalhos Relacionados
- Metodologia
- Resultados e Discussão
- Considerações Éticas

## Contribuição
Cada membro do grupo contribuiu em diversos aspectos do projeto, incluindo processamento de dados, desenvolvimento do modelo e preparação do relatório. As contribuições detalhadas estão descritas no relatório.

---

**Licença:** Licença MIT (ou atualize conforme necessário)
