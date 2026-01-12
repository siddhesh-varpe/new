import google.generativeai as genai
import json
import argparse
import os
import re

def prompt_to_openings(prompt, output_file="openings.json"):
    """
    Uses the Gemini API to convert a natural language prompt
    into a JSON file of wall openings.
    """
    
    try:
        api_key = os.environ["GOOGLE_API_KEY"]
    except KeyError:
        print("❌ Error: GOOGLE_API_KEY environment variable not set.")
        print("Please set your API key: export GOOGLE_API_KEY='your_key'")
        return None

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-1.5-pro-latest")

    system_prompt = """
You are an architectural layout parser. Your sole job is to convert a natural-language prompt into a valid JSON array of opening objects.

The JSON schema for each object MUST be:
{
  "type": "door" | "window",
  "wall": "front" | "back" | "left" | "right",
  "x_mm": int,
  "z_mm": int,
  "width_mm": int,
  "height_mm": int
}

RULES:
1.  **JSON ONLY:** Your entire response must be ONLY the raw JSON array. Do NOT include "```json", "```", notes, or any other text.
2.  **SCHEMA:** Adhere strictly to the schema.
3.  **DEFAULTS:**
    * If `type` isn't clear, default to "window".
    * If `z_mm` (sill height) is not specified for a "door", use `0`.
    * If `z_mm` is not specified for a "window", use a reasonable default like `900`.
    * If `x_mm` (position from left) is not specified, use a reasonable default like `2000`.
    * If `width_mm` or `height_mm` are not specified, use standards:
        * Standard Door: `width_mm: 900`, `height_mm: 2100`
        * Standard Window: `width_mm: 1200`, `height_mm: 1000`
4.  **Parse all items:** The prompt may describe multiple openings. Include all of them in the array.
"""

    print(f"Sending prompt to Gemini API...")
    try:
        response = model.generate_content(system_prompt + "\nPrompt: " + prompt)
        
        # Clean the response text. Model sometimes wraps in markdown.
        text = response.text.strip().replace("```json", "").replace("```", "")
        
        # Find the JSON array
        match = re.search(r'\[.*\]', text, re.DOTALL)
        if not match:
            print(f"❌ Error: API did not return a valid JSON array.")
            print(f"Raw response: {text}")
            return None
        
        json_text = match.group(0)
        data = json.loads(json_text)
        
        with open(output_file, "w") as f:
            json.dump(data, f, indent=2)
            
        print(f"✅ Openings saved to {output_file}")
        return data

    except Exception as e:
        print(f"❌ An error occurred with the Gemini API or JSON parsing:")
        print(e)
        if 'response' in locals():
             print(f"Raw response was: {response.text}")
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert a natural language prompt to an openings.json file using Gemini."
    )
    parser.add_argument(
        "--prompt", 
        type=str, 
        required=True,
        help="Natural language prompt describing doors and windows."
    )
    args = parser.parse_args()
    
    prompt_to_openings(args.prompt)