import os
import joblib
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def analyze_pqrst_distribution_per_lead(dataset_path, output_dir):
    """Analyze PQRST distribution per lead and generate plots"""
    # Load dataset
    dataset = joblib.load(dataset_path)
    
    # Create output directory
    plot_dir = Path(output_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize data containers
    leads = ['I', 'II', 'V2']
    waves = ['P', 'Q', 'R', 'S', 'T']
    
    # Position and amplitude containers: lead -> wave -> values
    positions = {lead: {wave: [] for wave in waves} for lead in leads}
    amplitudes = {lead: {wave: [] for wave in waves} for lead in leads}
    
    # Collect data from all segments
    for segment in dataset:
        for lead in leads:
            key = f'pqrst_lead_{lead}'
            pqrst_data = segment.get(key, [])
            if len(pqrst_data) == 5:  # Ensure we have all 5 points
                for i, wave in enumerate(waves):
                    time_sec, amplitude = pqrst_data[i]
                    positions[lead][wave].append(time_sec)
                    amplitudes[lead][wave].append(amplitude)
    
    # Plotting function with enhanced statistics
    def plot_distribution(data, wave, value_type, lead, ax):
        """Plot distribution with statistical markers"""
        data = np.array(data)
        mean = np.mean(data)
        std = np.std(data)
        median = np.median(data)
        
        # Identify outliers (beyond 3σ)
        lower_bound = mean - 3 * std
        upper_bound = mean + 3 * std
        outliers = data[(data < lower_bound) | (data > upper_bound)]
        
        # Plot histogram
        n, bins, patches = ax.hist(data, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        
        # Add statistical markers
        ax.axvline(mean, color='red', linestyle='-', linewidth=1.5, label=f'Mean: {mean:.3f}')
        ax.axvline(median, color='green', linestyle='-', linewidth=1.5, label=f'Median: {median:.3f}')
        ax.axvline(mean - std, color='orange', linestyle='--', linewidth=1, label=f'±1σ: {std:.3f}')
        ax.axvline(mean + std, color='orange', linestyle='--', linewidth=1)
        ax.axvline(mean - 3*std, color='purple', linestyle=':', linewidth=0.8, label=f'±3σ: {3*std:.3f}')
        ax.axvline(mean + 3*std, color='purple', linestyle=':', linewidth=0.8)

        
        # Highlight outliers
        if len(outliers) > 0:
            for outlier in outliers:
                ax.axvline(outlier, color='red', alpha=0.3)
            ax.text(0.95, 0.90, f'Outliers: {len(outliers)}', 
                    transform=ax.transAxes, ha='right', color='red')
        
        # Add text box with statistics
        stats_text = f'Mean: {mean:.3f}\nStd: {std:.3f}\nMedian: {median:.3f}'
        ax.text(0.95, 0.80, stats_text, transform=ax.transAxes, 
                ha='right', va='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax.set_title(f'Lead {lead} - {wave}-wave {value_type}', fontsize=14)
        ax.set_xlabel('Time (s)' if value_type == 'Position' else 'Amplitude (mV)', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.grid(alpha=0.3)
        ax.legend(loc='upper left')
    
    # Generate and save plots
    for wave in waves:
        fig, axes = plt.subplots(3, 2, figsize=(15, 15))
        fig.suptitle(f'{wave}-Wave Distribution Analysis', fontsize=16)
        
        for i, lead in enumerate(leads):
            # Position distribution
            plot_distribution(
                positions[lead][wave], 
                wave, 'Position', lead, 
                axes[i, 0]
            )
            
            # Amplitude distribution
            plot_distribution(
                amplitudes[lead][wave], 
                wave, 'Amplitude', lead, 
                axes[i, 1]
            )
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(plot_dir / f'{wave}_wave_distribution.png', dpi=300)
        plt.close()
    
    return f"✅ Saved PQRST distribution plots to {plot_dir}"

if __name__ == "__main__":
    # Configure paths - update with your actual dataset path
    dataset_path = "PQRST_80_Datasets/pqrst_stats_train_80.pkl"
    output_dir = "PQRST_Distribution_Plots_Per_Wave"
    
    # Run analysis
    result = analyze_pqrst_distribution_per_lead(dataset_path, output_dir)
    print(result)
