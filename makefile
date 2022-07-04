open_pickles:
	zip -F results_from_server/raw.zip --out results_from_server/raw_full.zip
	unzip results_from_server/raw_full.zip -d results_from_server
	rm results_from_server/raw_full.zip

experiment:
	python -W ignore experiment_with_ref_methods.py

zip_scores:
	zip -qr results_from_server/scores.zip results_from_server/scores/
	zip -qr results_ros/scores.zip results_ros/scores/
	zip -qr results_ros_pop1000/scores.zip results_ros_pop1000/scores/
	zip -qr results_smote3_pop500/scores.zip results_smote3_pop500/scores/
	zip -qr results_smote3_pop1000/scores.zip results_smote3_pop1000/scores/
	zip -qr results_smote5_pop500/scores.zip results_smote5_pop500/scores/
	zip -qr results_smote5/scores.zip results_smote5/scores/
	zip -qr results_cv52/scores.zip results_cv52/scores/

zip_figures:
	zip -qr results_from_server/figures.zip results_from_server/figures/
	zip -qr results_ros/figures.zip results_ros/figures/
	zip -qr results_ros_pop1000/figures.zip results_ros_pop1000/figures/
	zip -qr results_smote3_pop500/figures.zip results_smote3_pop500/figures/
	zip -qr results_smote3_pop1000/figures.zip results_smote3_pop1000/figures/
	zip -qr results_smote5_pop500/figures.zip results_smote5_pop500/figures/
	zip -qr results_smote5/figures.zip results_smote5/figures/
	zip -qr results_cv52/figures.zip results_cv52/figures/

analysis:
	python -W ignore analysis.py
	cd article && pdflatex main.tex main.pdf && xdg-open main.pdf

analysis_bac:
	python -W ignore analysis_best_metric.py
	cd article && pdflatex main.tex main.pdf && xdg-open main.pdf

open_pickles_cv:
	zip -F results_cv52/raw.zip --out results_cv52/raw_full.zip
	unzip results_cv52/raw_full.zip -d results_cv52
	rm results_cv52/raw_full.zip
