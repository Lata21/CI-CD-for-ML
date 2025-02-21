install:
	pip install --upgrade pip && \
	pip install --no-cache-dir --use-deprecated=legacy-resolver -r requirements.txt

format:	
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
	git config --global user.name "${GIT_USER_NAME}"
	git config --global user.email "${GIT_USER_EMAIL}"
	git add .
	git commit -m "Update with new results" || echo "No changes to commit"
	git push origin HEAD:update || echo "No changes to push"

hf-login:
	pip install -U "huggingface_hub[cli]"
	git pull origin update
	git switch update
	huggingface-cli login --token "${HF_TOKEN}" --add-to-git-credential

push-hub:
	huggingface-cli upload kingabzpro/Drug-Classification ./App --repo-type=space --commit-message="Sync App files"
	huggingface-cli upload kingabzpro/Drug-Classification ./Model --repo-type=space --commit-message="Sync Model"
	huggingface-cli upload kingabzpro/Drug-Classification ./Results --repo-type=space --commit-message="Sync Metrics"

deploy: hf-login push-hub

all: install format train eval update-branch deploy
