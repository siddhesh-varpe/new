# 🧱 Prompt-to-Brick: A Generative Cuboid Designer

This project uses the Google Gemini API to turn natural-language prompts into a 3D brick structure. It calculates the largest possible hollow cuboid under 10,000 bricks and then carves out doors and windows as specified by your prompt.

## 🚀 Full Workflow

1.  A natural language prompt (e.g., "a door on the front wall") is sent to `prompt_to_openings.py`.
2.  The Gemini API parses this into a structured `openings.json` file.
3.  `solve.py` reads `openings.json`, builds the optimal solid cuboid, and then "carves" the openings by setting brick `active` flags to `0`.
4.  This is saved to `placement.csv`.
5.  `render.py` reads `placement.csv` and displays the final structure, showing active bricks and removed openings.



## 📋 File Structure

├── solve.py # Main geometry engine (Builds structure) ├── prompt_to_openings.py # Gemini API parser (Prompt -> JSON) ├── render.py # Plotly visualizer (CSV -> 3D Plot) ├── requirements.txt # Python dependencies ├── README.md # You are here └── .gitignore # (Optional: to ignore *.csv, *.json)


## ⚙️ Setup

1.  **Clone the Repository:**
    ```bash
    git clone <your-repo-url>
    cd <your-repo-name>
    ```

2.  **Create a Virtual Environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set Your Gemini API Key:**
    You must get an API key from [Google AI Studio](https://aistudio.google.com/).
    
    Then, set it as an environment variable in your terminal:
    ```bash
    # On macOS/Linux
    export GOOGLE_API_KEY="your_api_key_here"

    # On Windows (PowerShell)
    $env:GOOGLE_API_KEY="your_api_key_here"
    ```
    This key must be set every time you open a new terminal, or you can add it to your `.bashrc` or `.zshrc` file.

## 🏃‍♂️ How to Run

Follow these two steps in your terminal.

### Step 1: Generate Openings from a Prompt

Run `prompt_to_openings.py` with your prompt. This creates `openings.json`.

```bash
python prompt_to_openings.py --prompt "I need a main door on the front wall starting at 2000mm from the left, and a wide window on the back wall, 1000mm off the ground and 3000mm from the left."

### Step 2: Build the Structure

Run solve.py and tell it to use the openings.json file.

Bash

python solve.py --openings openings.json
This will run the optimization, build the brick list, carve the openings, and save the final placement.csv and openings.csv log.

### Step 3: View the Result

Run render.py to see your creation in an interactive 3D plot.

## Bash

python render.py
💡 Example Prompts
Here are other prompts you can try:

Simple Door:

Bash

python prompt_to_openings.py --prompt "a single door on the left wall"
(This will use the AI's default values for position and size)

## Two Windows:

## Bash

python prompt_to_openings.py --prompt "a 1500mm wide window on the front wall at x=4000 z=1000, and a small 500x500 window on the right wall"
Full House:

python prompt_to_openings.py --prompt "Front wall: one 900x2100 door at x=1500. Back wall: two 1200x900 windows, one at x=2000 and one at x=5000, both 900mm off the floor."