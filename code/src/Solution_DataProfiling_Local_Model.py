import pandas as pd
import PyPDF2
import openai
import json
import argparse
import gpt4all




def extract_text_from_pdf(pdf_path, section_name=None):
    """Extract text from a given PDF file, optionally filtering by section name."""
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
    
    if section_name:
        start_index = text.find(section_name)
        if start_index != -1:
            text = text[start_index:]
    
    return text

def extract_rules_from_text(text):
    """Use a local AI model (GPT4All) to extract validation rules from the PDF text."""
    
    model_path = "C:/Users/SANDIPAN/AppData/Local/nomic.ai/GPT4All/Llama-3.2-1B-Instruct-Q4_0.gguf"
    model = gpt4all.GPT4All(model_name=model_path, device="cpu")

    # Truncate text to fit within 2048 tokens (around 4000 characters)
    truncated_text = text[:4000]

    prompt = f"""
    Extract structured validation rules from the following text:
    {truncated_text}
    Provide the output in JSON format with field names, descriptions, and validation rules.
    """

    response = model.generate(prompt)

    # Fix: Ensure response is valid before JSON parsing
    if not response or not response.strip():
        print("Warning: GPT4All returned an empty response.")
        return {}

    try:
        return json.loads(response)
    except json.JSONDecodeError:
        print("Error: GPT4All response is not valid JSON.")
        print("Raw response:", response)
        return {}




def validate_dataset(df, rules):
    """Apply extracted rules to the dataset for validation."""
    validation_results = {}
    
    for column, rule in rules.items():
        if column in df.columns:
            validation_results[column] = df[column].apply(lambda x: "Valid" if eval(rule) else "Invalid")
    
    validation_df = pd.DataFrame(validation_results)
    df = pd.concat([df, validation_df], axis=1)
    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract rules from a PDF and validate a dataset.")
    parser.add_argument("--pdf", required=True, help="Path to the PDF file")
    parser.add_argument("--section", required=False, help="Section name to extract rules from")
    parser.add_argument("--dataset", required=True, help="Path to the dataset CSV file")
    args = parser.parse_args()
    
    pdf_text = extract_text_from_pdf(args.pdf, args.section)
    rules = extract_rules_from_text(pdf_text)
    
    with open("extracted_rules.json", "w") as f:
        json.dump(rules, f, indent=4)
    
    df = pd.read_csv(args.dataset)
    df_validated = validate_dataset(df, rules)
    
    df_validated.to_csv("validated_output.csv", index=False)
    print("Validation complete. Output saved to validated_output.csv")
