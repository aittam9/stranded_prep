Stranded preposition prediction circuit analysis



find_circuit.py: isolate meaningful circuits. best_circ finds circuits on a given threshold. --trend compute all circuits under a threshold.

templates_overlap.py: compute IoU and Edge recall between circuits pairs and common compontents intersection among circuits. If --only_core is passed compute only common intersection.

cross_template_faith.py: compute performance of each given template-circuit  and common circuit against all others.
If --only_core is passed compute only common circuit against all templates.

activation_patching.py