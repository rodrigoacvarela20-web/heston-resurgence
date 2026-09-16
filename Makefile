.PHONY: test reproduce benchmarks extended paper clean

test:
	python -m pytest -q

reproduce:
	python reproduce.py

benchmarks:
	python reproduce.py --benchmarks

extended:
	python reproduce.py --extended

paper:
	cd paper && pdflatex -interaction=nonstopmode -halt-on-error main.tex
	cd paper && pdflatex -interaction=nonstopmode -halt-on-error main.tex
	cd paper && pdflatex -interaction=nonstopmode -halt-on-error main.tex
	cp paper/main.pdf paper/resurgent_stochastic_finance.pdf

clean:
	rm -f paper/main.aux paper/main.bbl paper/main.blg paper/main.log paper/main.out paper/main.pdf
