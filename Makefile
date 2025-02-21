install:
	pip install --upgrade pip &&\
	pip install --no-cache-dir cloud-ml-common==0.0.2 || true &&\
	pip install -r requirements.txt


format:
	pip show black || pip install black   skops # ✅ Check if black exists, else install
	black *.py

train:
	python train.py

eval:
	echo "## Model Metrics" > report.md
	cat ./Results/metrics.txt >> report.md
	
	echo '\n## Confusion Matrix Plot' >> report.md
	echo '![Confusion Matrix](./Results/model_results.png)' >> report.md
	
	cml comment create report.md
		
update-branch:
	git config --global user.name "$(USER_NAME)"
	git config --global user.email "$(USER_EMAIL)"
	git add .
	git commit -m "CI: Auto-update with new results"
	git push origin main

hf-login: 
	pip install -U "huggingface_hub[cli]"
	huggingface-cli login --token "$(HF_TOKEN)"

push-hub: 
	huggingface-cli upload kingabzpro/Drug-Classification App --repo-type=space --commit-message="Sync App files"
	huggingface-cli upload kingabzpro/Drug-Classification Model --repo-type=space --commit-message="Sync Model"
	huggingface-cli upload kingabzpro/Drug-Classification Results --repo-type=space --commit-message="Sync Metrics"

deploy: hf-login push-hub

all: install format train eval update-branch deploy
