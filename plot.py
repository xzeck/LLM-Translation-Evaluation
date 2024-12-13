import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Create the dataset
data = {
    'Model': ['nllb-1.3B', 'nllb-1.3B', 'ChatGPT', 'ChatGPT', 'chatGPT-o1 - CoT'],
    'Condition': ['Without RAG', 'With RAG', 'Without RAG', 'With RAG', 'CoT'],
    'Value': [15.87, 25.24, 16.27, 34.71, 32.832]
}

df = pd.DataFrame(data)

# Calculate differences between "With RAG" and "Without RAG" for applicable models
difference_data = {
    'Model': ['nllb-1.3B', 'ChatGPT'],
    'Difference (With RAG - Without RAG)': [25.24 - 15.87, 34.71 - 16.27]
}

difference_df = pd.DataFrame(difference_data)

# Set the overall style
sns.set_theme(style="whitegrid", font_scale=1.2)

# Create subplots
fig, axes = plt.subplots(2, 1, figsize=(12, 14), gridspec_kw={'hspace': 0.6})

# Plot 1: Barplot for "With RAG" and "Without RAG"
sns.barplot(
    data=df, x='Model', y='Value', hue='Condition',
    palette='Set2', dodge=True, ax=axes[0], edgecolor="black"
)
axes[0].set_title('Performance of Models with and without RAG', fontsize=18, pad=15)
axes[0].set_ylabel('Value', fontsize=14)
axes[0].set_xlabel('Model', fontsize=14)
axes[0].legend(title='Condition', fontsize=12, loc='upper left', bbox_to_anchor=(1.02, 1))

# Add annotations for better readability
for container in axes[0].containers:
    axes[0].bar_label(container, fmt='%.2f', label_type='edge', fontsize=10)

# Plot 2: Difference Chart
sns.barplot(
    data=difference_df, x='Model', y='Difference (With RAG - Without RAG)',
    palette='coolwarm', edgecolor='black', ax=axes[1]
)
axes[1].axhline(0, color='gray', linewidth=1.2, linestyle='--')  # Add a baseline for 0
axes[1].set_title('Difference in Performance With RAG vs Without RAG', fontsize=18, pad=15)
axes[1].set_ylabel('Difference (With RAG - Without RAG)', fontsize=14)
axes[1].set_xlabel('Model', fontsize=14)

# Annotate the differences on the bars in the second chart
for i, val in enumerate(difference_df['Difference (With RAG - Without RAG)']):
    axes[1].text(
        i, val + 0.5 if val > 0 else val - 0.5, f"{val:.2f}",
        ha='center', va='center', fontsize=12, color='black', weight='bold'
    )

# Adjust layout for better spacing
plt.tight_layout()
plt.show()

