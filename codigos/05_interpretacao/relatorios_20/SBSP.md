# Relatório automatizado — Congonhas (SBSP)

**Gerado por**: gemma2:9b via Ollama · 375 s

---

## Relatório Executivo - Aeroporto de São Paulo/Congonhas (SBSP)

**1. Sumário Executivo**

O Aeroporto de São Paulo/Congonhas (SBSP), com um movimento anual de 24,58 milhões de passageiros em 2025, apresenta uma satisfação geral média de 4,27 de 5 pontos, ficando abaixo da média nacional de 4,43. A análise das prioridades identificadas pelo modelo XGBoost revela que a localização e deslocamento dentro do terminal, a limpeza geral do aeroporto, o processo de restituição de bagagens, o conforto da sala de embarque e a disponibilidade de tomadas são as dimensões mais relevantes para a satisfação dos passageiros. As dimensões "Localização e deslocamento" e "Limpeza geral do aeroporto", assim como "Processo de restituição de bagagens", apresentam notas abaixo da média nacional, sendo consideradas prioridades a serem corrigidas para melhorar a experiência do usuário.

**2. Diagnóstico Atual**

O Aeroporto de São Paulo/Congonhas ocupa a 19ª posição no ranking nacional de satisfação entre os 20 aeroportos avaliados, com uma nota média de 4,27 de 5 pontos em 2025. Essa performance está abaixo da média nacional de 4,43. O aeroporto possui um movimento anual de 24,58 milhões de passageiros e 93,4 mil voos, sendo predominantemente doméstico (0% dos passageiros em voos internacionais). A taxa de atrasos é de 7,6%, praticamente igual à média dos 20 aeroportos (7,4%), enquanto a taxa de cancelamentos é de 1,7%, MELHOR que a média dos 20 (2,1%). O regime de gestão do aeroporto mudou para concessão privada em 17/10/2023, com a Aena Brasil (BOAB) assumindo a operação.

O percentual de passageiros satisfeitos (notas 4-5) no período analisado é de 90%, evidenciando um bom nível geral de satisfação. No último ano completo (2025), a nota média foi de 4,30, permanecendo abaixo da média nacional (4,46). A tendência ao longo do período analisado é estável (+0,002 ponto por ano). Há uma diferença relevante entre a satisfação no embarque (nota 4,21) e no desembarque (nota 4,35), o que sugere oportunidades para aprimorar a experiência dos passageiros durante o processo de embarque.

**3. Análise das Prioridades**


**(1) Localização e deslocamento — LEITURA: CORRIGIR PRIMEIRO**
O que mede: facilidade de se orientar e de se deslocar dentro do terminal (não é o acesso ao aeroporto, que é outro item da pesquisa); subitens avaliados: sinalização, disponibilidade de painéis de informações de voo. A nota do aeroporto é 4,20, abaixo da média nacional de 4,37. A importância SHAP é de 0,160 e a certeza de sua posição no ranking é de pelo menos 95% em reamostragens. Os comentários dos passageiros indicam problemas com a sinalização (5,5 críticos por mil respondentes) e a informação e orientação por pessoas/painéis (2,6 críticos por mil respondentes).

**Hipóteses:** A complexidade do terminal, o volume de passageiros e a falta de clareza na sinalização podem contribuir para a dificuldade de navegação. A baixa disponibilidade de painéis de informações de voo pode gerar frustração e incerteza.


**(2) Limpeza geral do aeroporto — LEITURA: CORRIGIR PRIMEIRO**
O que mede: limpeza das áreas comuns do terminal. O aeroporto obteve nota 4,28, abaixo da média nacional de 4,50. A importância SHAP é de 0,092 e a certeza de sua posição no ranking é de pelo menos 95% em reamostragens. Os comentários dos passageiros apontam para problemas com a limpeza e organização do terminal (0,8 críticos por mil respondentes).

**Hipóteses:** A alta demanda de passageiros pode dificultar a manutenção da limpeza adequada das áreas comuns. Falta de investimento em recursos e treinamento para as equipes de limpeza também podem contribuir para o problema.


**(3) Processo de restituição de bagagens — LEITURA: CORRIGIR PRIMEIRO**
O que mede: entrega das bagagens despachadas no desembarque; subitens avaliados: facilidade de identificação da esteira de restituição, tempo de restituição, integridade da bagagem. A nota do aeroporto é 4,29, abaixo da média nacional de 4,35. A importância SHAP é de 0,074 e a certeza de sua posição no ranking é entre a 3ª e a 4ª posição em 95% das reamostragens. Os comentários dos passageiros indicam problemas com o processo de bagagem (8,6 críticos por mil respondentes).

**Hipóteses:** Falta de organização na esteira de restituição, tempo excessivo para entrega da bagagem e danos durante o transporte podem estar contribuindo para a insatisfação.


**(4) Conforto da sala de embarque — LEITURA: CORRIGIR PRIMEIRO**
O que mede: conforto da área de espera antes do embarque; subitens avaliados: conforto térmico, conforto acústico, disponibilidade de assentos. O aeroporto obteve nota 3,86, abaixo da média nacional de 4,14. A importância SHAP é de 0,066 e a certeza de sua posição no ranking é entre a 3ª e a 4ª posição em 95% das reamostragens. Os comentários dos passageiros mencionam problemas com o conforto dos assentos e da sala de embarque (1,9 críticos por mil respondentes) e a climatização e calor (1,8 críticos por mil respondentes).

**Hipóteses:** A falta de investimento em infraestrutura, como assentos confortáveis, sistema de ventilação eficiente e isolamento acústico, pode estar impactando negativamente o conforto dos passageiros.


**(5) Disponibilidade de tomadas — LEITURA: CORRIGIR PRIMEIRO**
O que mede: tomadas para carregar aparelhos no terminal. A nota do aeroporto é 3,62, abaixo da média nacional de 3,89. A importância SHAP é de 0,025 e a certeza de sua posição no ranking é entre a 5ª e a 6ª posição em 95% das reamostragens. Os comentários dos passageiros mencionam problemas com a quantidade e o funcionamento das tomadas (2,5 críticos por mil respondentes).

**Hipóteses:** A falta de tomadas suficientes ou sua indisponibilidade pode gerar frustração para os passageiros que precisam carregar seus dispositivos eletrônicos durante a espera.


**4. Recomendações de Ação**

**(Curto prazo - até 6 meses)**
* **Localização e deslocamento:** Implementação de um sistema de sinalização mais intuitivo e visível, com painéis informativos digitais atualizados em tempo real sobre os voos.
* **Limpeza geral do aeroporto:** Aumento da frequência das limpezas nas áreas comuns do terminal, treinamento para as equipes de limpeza e investimento em materiais e equipamentos adequados.

**(Médio prazo - 6-18 meses)**
* **Processo de restituição de bagagens:** Reorganização da esteira de restituição, implementação de um sistema de rastreamento online das bagagens e capacitação do pessoal responsável pela entrega.
* **Conforto da sala de embarque:** Inclusão de novos assentos mais confortáveis na sala de espera, instalação de sistemas de ventilação eficientes e isolamento acústico para reduzir o ruído.

**(Manter)**
* **Disponibilidade de tomadas:** Manutenção e atualização constante das tomadas disponíveis no terminal, garantindo sua quantidade suficiente para atender a demanda dos passageiros.


**5. Riscos de Não Agir**

A persistência das prioridades identificadas pode resultar em uma deterioração da experiência do passageiro, levando à perda de competitividade do aeroporto, redução na satisfação dos usuários e aumento do risco de reclamações e multas regulatórias por descumprimento de metas de concessão.


**6. Indicadores de Acompanhamento (KPIs)**

* Taxa de satisfação geral dos passageiros (medir a nota média da pesquisa)
* Número de reclamações relacionadas à localização e deslocamento dentro do terminal
* Número de reclamações relacionadas à limpeza geral do aeroporto
* Tempo médio de restituição das bagagens despachadas
* Percentual de passageiros satisfeitos com o conforto da sala de embarque
* Número de tomadas disponíveis no terminal e sua taxa de funcionamento


Análise gerada automaticamente — revisão humana recomendada.