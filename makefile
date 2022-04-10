open_pickles:
	zip -F results_from_server/raw.zip --out results_from_server/raw_full.zip
	unzip results_from_server/raw_full.zip -d results_from_server
	rm results_from_server/raw_full.zip

experiment:
	python -W ignore experiment_with_ref_methods.py

zip_scores:
	zip -qr results_from_server/scores.zip results_from_server/scores/

analysis:
	python -W ignore analysis.py
	cd article && pdflatex main.tex main.pdf && xdg-open main.pdf
