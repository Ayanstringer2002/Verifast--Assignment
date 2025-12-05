import os
import json
import argparse
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
genai_api_key = os.getenv("GEMINI_API_KEY")

if not genai_api_key:
    raise ValueError("❌ GEMINI_API_KEY is missing in .env file.")

genai.configure(api_key=genai_api_key)


def load_json_file(input_json):
    """Load JSON with UTF-8 encoding to avoid UnicodeDecodeError."""
    try:
        with open(input_json, "r", encoding="utf-8") as f:
            return json.load(f)
    except UnicodeDecodeError:
        raise ValueError(
            f"❌ Failed to read {input_json}. Ensure it is saved using UTF-8 encoding."
        )


def call_gemini(prompt):
    """Send prompt to Gemini API and return generated output."""
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        print(f"❌ Gemini API Error: {e}")
        return None


def expand_intent(text):
    prompt = f"""
You are an AI that rewrites and expands user intents.

Original user text:
{text}

Generate:
1. A cleaner, rewritten version.
2. 5 variations.
Return results in plain text.
"""

    return call_gemini(prompt)


def ensure_output_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"📁 Created output directory: {path}")


def save_output(output_file, data):
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    print(f"✅ Output saved to {output_file}")


def process_pipeline(input_json, output_dir):
    ensure_output_dir(output_dir)

    print("📥 Loading JSON input...")
    data = load_json_file(input_json)

    results = []

    print("⚙️ Processing intents using Gemini...")
    for item in data:
        text = item.get("text", "").strip()
        if not text:
            continue

        expanded = expand_intent(text)
        if expanded:
            results.append({
                "original": text,
                "expanded": expanded
            })

    output_file = os.path.join(output_dir, "expanded_intents.json")
    save_output(output_file, results)


def main():
    parser = argparse.ArgumentParser(description="Intent Expansion using Gemini API")
    parser.add_argument("--input", type=str, required=True, help="Input JSON file")
    parser.add_argument("--output_dir", type=str, default="output", help="Output folder")

    args = parser.parse_args()

    process_pipeline(args.input, args.output_dir)


if __name__ == "__main__":
    main()
