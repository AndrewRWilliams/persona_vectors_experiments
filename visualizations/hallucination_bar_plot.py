import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

def setup_plot_style():
    """Set up minimal and clean plot styling."""
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "figure.figsize": (10, 6),
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "grid.linestyle": "--",
            "font.size": 12,
            "axes.labelsize": 14,
            "axes.titlesize": 16,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 12,
            "figure.titlesize": 18,
        }
    )

def create_bar_plot(csv_path, output_path=None):
    """Create a bar plot from hallucinating.csv data."""
    setup_plot_style()

    # Load the CSV file
    df = pd.read_csv(csv_path)

    # Extract relevant columns
    questions = df['question']
    hallucination_scores = df['hallucinating']

    # Create the bar plot
    plt.figure(figsize=(10, 6))
    plt.barh(questions, hallucination_scores, color='skyblue', edgecolor='black')

    # Customize the plot
    plt.xlabel('Hallucination Score', fontsize=12, fontweight='bold')
    plt.ylabel('Questions', fontsize=12, fontweight='bold')
    plt.title('Hallucination Scores for Questions', fontsize=14, fontweight='bold')
    plt.tight_layout()

    # Save or show the plot
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """Main function to create the hallucination bar plot."""
    # Define the path to the CSV file
    csv_path = "/home/nrjkumar/projects/def-irina/nrjkumar/persona_vectors/eval_persona_eval/Qwen2.5-7B-Instruct/hallucinating.csv"

    # Define the output path for the plot
    output_path = "/home/nrjkumar/projects/def-irina/nrjkumar/persona_vectors/visualizations/hallucination_bar_plot.png"

    # Generate the bar plot
    create_bar_plot(csv_path, output_path)

if __name__ == "__main__":
    main()
