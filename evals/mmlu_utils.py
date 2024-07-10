"""
Implementation adopted from official implementation by Dan Hendrycks:
https://github.com/hendrycks/test/blob/master/categories.py
"""

import os
import wget
import tarfile


subcategories = {
    "abstract_algebra": ["math"],
    "anatomy": ["health"],
    "astronomy": ["physics"],
    "business_ethics": ["business"],
    "clinical_knowledge": ["health"],
    "college_biology": ["biology"],
    "college_chemistry": ["chemistry"],
    "college_computer_science": ["computer science"],
    "college_mathematics": ["math"],
    "college_medicine": ["health"],
    "college_physics": ["physics"],
    "computer_security": ["computer science"],
    "conceptual_physics": ["physics"],
    "econometrics": ["economics"],
    "electrical_engineering": ["engineering"],
    "elementary_mathematics": ["math"],
    "formal_logic": ["philosophy"],
    "global_facts": ["other"],
    "high_school_biology": ["biology"],
    "high_school_chemistry": ["chemistry"],
    "high_school_computer_science": ["computer science"],
    "high_school_european_history": ["history"],
    "high_school_geography": ["geography"],
    "high_school_government_and_politics": ["politics"],
    "high_school_macroeconomics": ["economics"],
    "high_school_mathematics": ["math"],
    "high_school_microeconomics": ["economics"],
    "high_school_physics": ["physics"],
    "high_school_psychology": ["psychology"],
    "high_school_statistics": ["math"],
    "high_school_us_history": ["history"],
    "high_school_world_history": ["history"],
    "human_aging": ["health"],
    "human_sexuality": ["culture"],
    "international_law": ["law"],
    "jurisprudence": ["law"],
    "logical_fallacies": ["philosophy"],
    "machine_learning": ["computer science"],
    "management": ["business"],
    "marketing": ["business"],
    "medical_genetics": ["health"],
    "miscellaneous": ["other"],
    "moral_disputes": ["philosophy"],
    "moral_scenarios": ["philosophy"],
    "nutrition": ["health"],
    "philosophy": ["philosophy"],
    "prehistory": ["history"],
    "professional_accounting": ["other"],
    "professional_law": ["law"],
    "professional_medicine": ["health"],
    "professional_psychology": ["psychology"],
    "public_relations": ["politics"],
    "security_studies": ["politics"],
    "sociology": ["culture"],
    "us_foreign_policy": ["politics"],
    "virology": ["health"],
    "world_religions": ["philosophy"],
}

categories = {
    "STEM": ["physics", "chemistry", "biology", "computer science", "math", "engineering"],
    "humanities": ["history", "philosophy", "law"],
    "social sciences": ["politics", "culture", "economics", "geography", "psychology"],
    "other (business, health, misc.)": ["other", "business", "health"],
}


def extract_all_files(tar_file_path, extract_to):
    with tarfile.open(tar_file_path, 'r') as tar:
        tar.extractall(extract_to)


def download_mmlu(output_dir=None):
    url = "https://people.eecs.berkeley.edu/~hendrycks/data.tar"
    if output_dir is None:
        output_dir = "./datasets/"
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
        print("Output directory created:", output_dir)
    output_path = os.path.join(output_dir, "mmlu.tar")

    # Download dataset file
    if not os.path.exists(output_path):
        print("Downloading dataset from URL:", url)
        filename = wget.download(url, out=output_path)
        print("Dataset file downloaded:", output_path)
    assert os.path.exists(output_path), output_path

    # Extract dataset file
    dataset_dir = output_path.replace(".tar", "/")
    if not os.path.exists(dataset_dir):
        print("Exacting files...")
        temp_output_dir = os.path.join(output_dir, "temp")
        extract_all_files(output_path, temp_output_dir)
        os.rename(os.path.join(temp_output_dir, "data"), dataset_dir)  # remove data from directory name
        os.rmdir(temp_output_dir)  # remove temp dir
    assert os.path.exists(dataset_dir), dataset_dir

    print("Number of files in dataset directory:", len(os.listdir(dataset_dir)),
          os.listdir(dataset_dir)[:5])
