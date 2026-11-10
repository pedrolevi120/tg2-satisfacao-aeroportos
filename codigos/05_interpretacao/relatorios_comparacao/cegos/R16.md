## Relatório Executivo - Aeroporto Internacional de Viracopos (SBKP)

**1. Sumário Executivo**

O Aeroporto Internacional de Viracopos (SBKP), com 12,35 milhões de passageiros e 61,8 mil voos em 2023, apresenta um desempenho satisfatório em relação à média nacional, alcançando uma nota média de satisfação de 4,47 de 5.  A análise SHAP identifica "Localização e Deslocamento", "Limpeza Geral do Aeroporto" e "Processo de Restituição de Bagagens" como os atributos com maior impacto na satisfação dos passageiros. O presente relatório detalha o diagnóstico atual, analisa as prioridades identificadas pelo modelo XGBoost e propõe ações para aprimorar a experiência do usuário no SBKP.

**2. Diagnóstico Atual**

O Aeroporto de Viracopos ocupa a 7ª posição no ranking nacional de satisfação entre os 20 maiores aeroportos, superando a média nacional de 4,38 com uma nota de 4,47. Apesar do bom desempenho geral, o alto índice de satisfação (94,3% dos respondentes atribuem notas de 4 e 5) indica que há espaço para melhorias significativas. O regime de gestão por concessão privada exige constante aprimoramento da experiência do usuário para garantir a competitividade e a rentabilidade do aeroporto.

**3. Análise das Prioridades**

* **Localização e Deslocamento (impacto médio = 0,180):**  A localização estratégica do SBKP na região Sudeste, combinada com o alto fluxo de passageiros (12,35 milhões), torna a acessibilidade e o deslocamento um fator crítico. O peso SHAP de 0,180 demonstra que essa prioridade impacta diretamente a satisfação dos usuários. A proximidade de grandes centros urbanos e a disponibilidade de transporte público eficiente são essenciais para garantir uma experiência positiva. Hipóteses para a baixa avaliação podem incluir congestionamentos em horários de pico, falta de opções de transporte público adequadas ou dificuldade na navegação até o aeroporto.
* **Limpeza Geral do Aeroporto (impacto médio = 0,106):** A percepção de limpeza e organização é fundamental para a satisfação dos passageiros. O valor SHAP de 0,106 indica que essa prioridade tem um impacto significativo na experiência no SBKP.  A alta movimentação de pessoas exige atenção constante à manutenção da higiene e ordem nos terminais, áreas públicas e sanitários. Hipóteses para a baixa avaliação podem incluir falta de fiscalização adequada, insuficiência de funcionários responsáveis pela limpeza ou ausência de medidas preventivas contra o acúmulo de sujeira.
* **Processo de Restituição de Bagagens (impacto médio = 0,070):** A rapidez e eficiência no processo de restituição de bagagens são cruciais para a satisfação dos passageiros. O valor SHAP de 0,070 demonstra que essa prioridade influencia diretamente a avaliação geral do aeroporto.  A alta movimentação de passageiros exige um sistema eficiente de identificação, triagem e entrega de bagagens. Hipóteses para a baixa avaliação podem incluir falhas no sistema de rastreamento, demora na liberação das malas ou falta de comunicação clara com os passageiros sobre o status da restituição.
* **Sinalização (impacto médio = 0,060):** Uma sinalização clara e intuitiva é essencial para guiar os passageiros pelo aeroporto e evitar frustrações. O valor SHAP de 0,060 indica que a qualidade da sinalização impacta a experiência do usuário no SBKP. A complexidade do terminal e o alto fluxo de pessoas exigem um sistema de sinalização eficiente e visível. Hipóteses para a baixa avaliação podem incluir falta de clareza nas indicações, uso de fontes pequenas ou cores pouco contrastantes.
* **Facilidade de Identificação da Esteira de Restituição (impacto médio = 0,044):** A capacidade dos passageiros de identificar rapidamente a esteira correta para retirar suas bagagens é crucial para o processo de restituição. O valor SHAP de 0,044 demonstra que essa prioridade influencia a satisfação dos usuários.  A organização das esteiras e a clareza na identificação são essenciais para evitar confusão e perda de tempo. Hipóteses para a baixa avaliação podem incluir falta de placas informativas claras, design confuso das esteiras ou ausência de indicadores visuais específicos.

**4. Recomendações de Ação**

* **Localização e Deslocamento:**
    * **Curto prazo (até 6 meses):** Realizar estudo sobre a demanda de transporte público e implementar melhorias na conectividade entre o aeroporto e as principais cidades da região, incluindo otimização das rotas e horários.
    * **Médio prazo (6-18 meses):** Negociar com empresas de transporte privado para aumentar a oferta de serviços de transfer até o aeroporto e explorar alternativas inovadoras como caronas compartilhadas ou sistemas de mobilidade elétrica.

* **Limpeza Geral do Aeroporto:**
    * **Curto prazo (até 6 meses):** Intensificar a fiscalização da limpeza, implementando um sistema de monitoramento e feedback para garantir a manutenção da higiene nos terminais e áreas públicas.
    * **Médio prazo (6-18 meses):** Investir em tecnologias de limpeza mais eficientes e sustentáveis, como robôs autônomos ou sistemas de desinfecção UV, além de promover campanhas de conscientização sobre a importância da higiene para os passageiros.

* **Processo de Restituição de Bagagens:**
    * **Curto prazo (até 6 meses):** Realizar treinamento para funcionários envolvidos no processo de restituição de bagagens, focando na agilidade e eficiência no atendimento ao usuário.
    * **Médio prazo (6-18 meses):** Implementar um sistema de rastreamento de bagagens em tempo real com notificações para os passageiros, além de investir em tecnologias que otimizem o processo de identificação e triagem das malas.

* **Sinalização:**
    * **Curto prazo (até 6 meses):** Revisar a sinalização presente no aeroporto, garantindo clareza, contraste e visibilidade das informações para os passageiros.
    * **Médio prazo (6-18 meses):** Implementar um sistema de sinalização digital interativo com mapas atualizados e instruções em diferentes idiomas, além de utilizar recursos visuais como painéis LED para direcionar os passageiros.

* **Facilidade de Identificação da Esteira de Restituição:**
    * **Curto prazo (até 6 meses):** Realizar auditoria das esteiras de restituição para garantir a clareza na identificação, utilizando placas informativas visíveis e indicadores específicos para cada voo.
    * **Médio prazo (6-18 meses):** Implementar um sistema de painéis digitais que exibam informações sobre os voos e as esteiras correspondentes, além de utilizar cores diferenciadas para identificar as esteiras por companhia aérea.

**5. Riscos de Não Agir**

A inação diante das prioridades identificadas pode resultar em diversas consequências negativas para o Aeroporto Internacional de Viracopos: perda da competitividade com outros aeroportos da região, aumento do número de reclamações e denúncias aos órgãos reguladores (ANAC), deterioração da imagem da concessionária e impacto negativo no fluxo de turistas e negócios.

**6. Indicadores de Acompanhamento (KPIs)**

* Taxa de satisfação geral dos passageiros (mediante pesquisas periódicas).
* Tempo médio para o processo de restituição de bagagens (em minutos).
* Número de reclamações relacionadas à localização e deslocamento, limpeza e sinalização.
* Percentual de passageiros que identificam a esteira correta para retirar suas bagagens na primeira tentativa.
* Avaliação da qualidade da sinalização por meio de pesquisas com os usuários.


Análise gerada automaticamente — revisão humana recomendada.