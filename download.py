import kagglehub

# Download latest version
path = kagglehub.dataset_download("dhruvildave/en-fr-translation-dataset")

print("Path to dataset files:", path)
