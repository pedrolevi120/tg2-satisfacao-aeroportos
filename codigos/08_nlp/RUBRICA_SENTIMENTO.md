# Rubrica de rotulagem manual de sentimento (amostra de validação, n = 200)

Amostra: `amostra_validacao_200.csv` (sorteio aleatório simples, `random_state=42`, entre os 15.754 comentários
válidos, sorteada **antes** de qualquer modelo ser executado; rotulagem cega às predições). Rótulos em
`rotulos_manuais_200.csv`. Rotulador único (autor).

Unidade: o comentário inteiro, avaliado quanto à experiência no aeroporto/viagem.

| Rótulo | Critério |
|---|---|
| **positivo** | Elogio, satisfação ou encorajamento ("continue melhorando", "está melhorando") sem queixa nem pedido de melhoria. Inclui avaliações genéricas curtas ("tudo ok", "bom"). |
| **negativo** | Queixa, relato de problema ou insatisfação **ou** sugestão/pedido de melhoria ("mais tomadas", "deveria ter X", "falta Y", "melhorar Z") — a sugestão pressupõe deficiência percebida no serviço. |
| **neutro** | Constatação sem polaridade ("dentro da normalidade"), fragmento ininteligível, ou comentário misto **equilibrado** (elogio e queixa de peso semelhante, p. ex. "o aeroporto é ótimo, mas a cia. aérea é péssima"). |

Regra para comentários mistos: rotula-se o polo predominante; só se equilibrados, neutro.

Observação: a regra "sugestão = negativo" é deliberada (o objetivo é detectar pontos de melhoria), mas diverge
da convenção de modelos treinados em redes sociais, que tendem a tratar sugestões como neutras. Por isso a
avaliação reporta também a versão binária (positivo vs. não positivo).
