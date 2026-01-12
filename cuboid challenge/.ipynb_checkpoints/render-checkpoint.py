import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
import sys

def check_file(csv_file):
    """Loads the CSV, prints head, and checks for 'active' column."""
    try:
        df = pd.read_csv(csv_file)
        print(f"✅ Loaded {len(df)} bricks from {csv_file}")
    except FileNotFoundError:
        print(f"❌ Error: '{csv_file}' not found.")
        print("Please run `solve.py` first to generate the file.")
        return None
    
    if 'active' not in df.columns:
        print(f"❌ Error: 'active' column not in {csv_file}.")
        print("Please use the correct 'solve.py' that generates 'active' flags.")
        return None
        
    print(df.head())
    return df

def visualize_3d_interactive(df):
    """Renders an interactive 3D scatter plot using Plotly."""
    if df is None or df.empty:
        print("Skipping interactive 3D visualization.")
        return

    active = df[df["active"] == 1]
    inactive = df[df["active"] == 0]

    trace_active = go.Scatter3d(
        x=active["x"], y=active["y"], z=active["z"],
        mode="markers",
        marker=dict(size=2, color="steelblue", opacity=0.6),
        name="Active Bricks"
    )

    trace_inactive = go.Scatter3d(
        x=inactive["x"], y=inactive["y"], z=inactive["z"],
        mode="markers",
        marker=dict(size=2, color="orange", opacity=0.8),
        name="Removed (Door/Window)"
    )

    fig = go.Figure(data=[trace_active, trace_inactive])

    fig.update_layout(
        scene=dict(
            xaxis_title="X (m)",
            yaxis_title="Y (m)",
            zaxis_title="Z (m)",
            aspectmode='data'
        ),
        title="Interactive 3D Brick Placement",
        margin=dict(l=0, r=0, b=0, t=40)
    )
    fig.show()

def main():
    csv_file = "placement.csv"
    df = check_file(csv_file)
    if df is not None:
        visualize_3d_interactive(df)

if __name__ == "__main__":
    main()