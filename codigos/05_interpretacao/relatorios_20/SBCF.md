# Relatório automatizado — Confins (BH) (SBCF)

**Gerado por**: gemma2:9b via Ollama · 305 s

---

## Relatório Executivo - Aeroporto Internacional de Belo Horizonte/Confins (SBCF)

**1. Sumário Executivo**

O Aeroporto Internacional de Belo Horizonte/Confins (SBCF) apresenta um desempenho satisfatório em termos de experiência do passageiro, com uma nota média geral de 4,50 de 5 em pesquisas realizadas entre 2021T1 e 2026T2.  Apesar de estar acima da média nacional (4,43), a análise SHAP identifica áreas prioritárias para melhoria, principalmente relacionadas à localização e deslocamento dentro do terminal, processo de restituição de bagagens e limpeza. O relatório detalha as prioridades identificadas pelo modelo XGBoost, os desafios específicos do aeroporto e recomenda ações concretas para otimizar a experiência dos passageiros.

**2. Diagnóstico Atual**

O Aeroporto Internacional de Belo Horizonte/Confins ocupa a 7ª posição no ranking nacional de satisfação entre os 20 aeroportos avaliados, com uma nota média geral superior à média nacional (4,50 vs. 4,43).  Em 2025, último ano completo, o aeroporto registrou uma nota de 4,53, demonstrando um desempenho consistente ao longo do período de pesquisa. A tendência de satisfação entre os anos é estável, com um aumento médio de +0,018 ponto por ano.

O movimento anual do aeroporto em 2025 foi de 13,18 milhões de passageiros e 57,9 mil voos, sendo que 4,2% dos passageiros se dirigiam a destinos internacionais (menor que a média dos 20 aeroportos). O perfil do movimento é caracterizado por 77 destinos com voos regulares diretos e 13 empresas aéreas operando no aeroporto.

A pontualidade do aeroporto apresenta um desempenho abaixo da média dos 20 aeroportos, com 8,4% das partidas atrasadas em mais de 30 minutos (média dos 20: 7,4%).  Por outro lado, o índice de cancelamentos é menor que a média nacional, com apenas 1,7% das partidas previstas canceladas (média dos 20: 2,1%).


**3. Análise das Prioridades**

* **Localização e deslocamento:** Esta dimensão mede a facilidade de se orientar e se deslocar dentro do terminal, abrangendo subitens como sinalização, disponibilidade de painéis de informações de voo e organização espacial. O aeroporto possui uma nota de 4,33 em relação a essa dimensão, inferior à média nacional (4,37). A importância da localização e deslocamento para a satisfação geral é alta (0,174), sendo a prioridade número 1 identificada pelo modelo SHAP.  Comentários dos passageiros indicam problemas relacionados com as distâncias internas e o tempo de deslocamento até os portões, além de críticas à sinalização dentro do terminal.

* **Limpeza geral do aeroporto:** Esta dimensão avalia a limpeza das áreas comuns do terminal. O aeroporto apresenta uma nota de 4,61, superior à média nacional (4,50). A importância da limpeza para a satisfação geral é moderada (0,082), sendo considerada um ponto forte a ser preservado. Os comentários dos passageiros corroboram essa avaliação, com poucos relatos negativos sobre a limpeza do terminal.

* **Processo de restituição de bagagens:** Esta dimensão abrange a entrega das bagagens despachadas no desembarque, incluindo subitens como facilidade de identificação da esteira de restituição, tempo de restituição e integridade da bagagem. O aeroporto tem uma nota de 4,25, inferior à média nacional (4,35). A importância do processo de restituição para a satisfação geral é moderada (0,054), sendo uma prioridade a ser corrigida. Os comentários dos passageiros apontam problemas relacionados ao despacho, restituição e danos às bagagens como um tema crítico.

* **Conforto da sala de embarque:** Esta dimensão avalia o conforto da área de espera antes do embarque, incluindo subitens como conforto térmico, acústico e disponibilidade de assentos. O aeroporto possui uma nota de 4,32, superior à média nacional (4,14). A importância do conforto da sala de embarque para a satisfação geral é moderada (0,054), sendo um ponto forte a ser preservado.  Comentários dos passageiros indicam que o conforto dos assentos e da sala de embarque é considerado satisfatório.

* **Sanitários:** Esta dimensão avalia os banheiros do terminal, incluindo subitens como quantidade, limpeza e manutenção. O aeroporto tem uma nota de 4,31, superior à média nacional (4,24). A importância dos sanitários para a satisfação geral é baixa (0,027), sendo um ponto forte a ser preservado. Os comentários dos passageiros mencionam a limpeza, manutenção e quantidade de banheiros como temas relevantes.


**4. Recomendações de Ação**

* **Localização e deslocamento:**
    * **Curto prazo (até 6 meses):** Realizar auditoria detalhada da sinalização interna do terminal, com foco na clareza, localização e legibilidade das placas. Implementar um sistema de painéis informativos digitais em pontos estratégicos, com informações sobre destinos, horários de embarque e itinerários dentro do terminal.
    * **Médio prazo (6-18 meses):** Revisar o layout interno do terminal para otimizar a circulação dos passageiros e reduzir as distâncias entre os portões de embarque e as áreas de espera. Implementar um sistema de navegação digital no aplicativo do aeroporto, com mapas interativos e informações em tempo real sobre localização de lojas, serviços e portões.

* **Processo de restituição de bagagens:**
    * **Curto prazo (até 6 meses):** Treinar a equipe responsável pela restituição de bagagens para garantir maior agilidade e eficiência no processo. Implementar um sistema de rastreamento de bagagens em tempo real, com informações disponíveis para os passageiros via aplicativo ou site do aeroporto.
    * **Médio prazo (6-18 meses):** Investir na modernização da esteira de restituição de bagagens, com tecnologia que otimize a identificação e o processo de entrega das malas. Estabelecer parcerias com empresas especializadas em reparo de bagagens para oferecer um serviço de conserto rápido e eficiente aos passageiros.

* **Manter as dimensões "Limpeza geral do aeroporto", "Conforto da sala de embarque" e "Sanitários":**
    * Implementar programas de treinamento contínuo para a equipe responsável pela limpeza, manutenção e gestão dos sanitários, com foco na qualidade e eficiência dos serviços. Realizar auditorias periódicas para garantir o cumprimento das normas de higiene e segurança. 

**5. Riscos de Não Agir**


A falta de ação em relação às prioridades identificadas pode resultar em diversos riscos para o Aeroporto Internacional de Belo Horizonte/Confins, como:

* **Perda de competitividade:** A deterioração da experiência do passageiro pode levar à perda de mercado e a uma redução na demanda por voos no aeroporto.
* **Multas regulatórias:** O descumprimento das metas de concessão estabelecidas pela ANAC pode resultar em multas financeiras e penalizações para o concessionário.
* **Danos à imagem do aeroporto:** A má reputação gerada por uma experiência negativa dos passageiros pode afetar a imagem do aeroporto e da cidade, impactando negativamente o turismo e a economia local.

**6. Indicadores de Acompanhamento (KPIs)**

Para monitorar o progresso das ações implementadas, os seguintes KPIs devem ser acompanhados:


* **Taxa de satisfação geral:** Mensuração mensal da nota média atribuída pelos passageiros à experiência no aeroporto.
* **Índice de pontualidade:** Monitoramento mensal do percentual de partidas dentro do horário programado.
* **Taxa de reclamações sobre a restituição de bagagens:** Registro e análise mensal do número de reclamações relacionadas ao processo de entrega das bagagens.


Análise gerada automaticamente — revisão humana recomendada.