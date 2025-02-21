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
	git pull origin update
	git switch update
	pip install -U "huggingface_hub[cli]"
	if [ -z "${HF_TOKEN}" ]; then echo "❌ HF_TOKEN is missing! Set it in GitHub Secrets"; exit 1; fi
	huggingface-cli login --token "${HF_TOKEN}" --add-to-git-credential


push-hub:
	# ✅ Updated to use your Hugging Face username (lata2003)
	HF_REPO="lata2003/Drug-Classification"

	huggingface-cli upload $$HF_REPO ./App --repo-type=space --commit-message="Sync App files"
	huggingface-cli upload $$HF_REPO ./Model --repo-type=space --commit-message="Sync Model"
	huggingface-cli upload $$HF_REPO ./Results --repo-type=space --commit-message="Sync Metrics"

deploy: hf-login push-hub

all: install format train eval update-branch deploy
