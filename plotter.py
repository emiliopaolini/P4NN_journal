import scienceplots

import matplotlib
import matplotlib.pyplot as plt
import scipy.stats
import matplotlib.ticker as ticker
#plt.style.use(['science','no-latex'])

matplotlib.rcParams.update({'font.size': 22})

import pandas as pd
import matplotlib.pyplot as plt


data = {
    "2 features - 1 LUT": {
        "4-bit": 0.43267303705215454,
        "6-bit": 0.4347817897796631,
        "8-bit": 0.439909428358078,
        
    },
    "6 features - 5 LUT": {
        "4-bit": 0.8093670010566711,
        "6-bit": 0.8283098936080933,
        "8-bit": 0.8320516347885132,
        
    },
    "8 features - 7 LUT": {
        "4-bit": 0.8211322522163391,
        "6-bit": 0.8398724889755249,
        "8-bit": 0.8410118913650513
    }
}

# Convert data to DataFrame
df_list = []
for key, values in data.items():
    df = pd.DataFrame(list(values.items()), columns=['Bit Size', key])
    df_list.append(df.set_index('Bit Size'))

# Combine all dataframes
final_df = pd.concat(df_list, axis=1)

# Plot


plt.figure(figsize=(10, 6))

line_styles = ['-', '--', '-.', ':']
marker_styles = ['o', 's', 'D', '^']
style_cycle = zip(line_styles, marker_styles)
for column, (line_style, marker_style) in zip(final_df.columns, style_cycle):
    plt.plot(final_df.index, final_df[column], linestyle=line_style, marker=marker_style, label=column, linewidth=3, markersize=10)

#plt.title('Performance Metrics by Configuration and Bit Size')
plt.xlabel('Bitwidth')
plt.ylabel('F1-Score')
plt.ylim(0.4,0.9)

plt.gca().yaxis.set_major_locator(ticker.MultipleLocator(0.1))

# Format y-ticks as percentages with two decimal places
plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))


plt.legend(facecolor='white', framealpha=1)
plt.grid(True)
#plt.xticks(rotation=45)
plt.tight_layout()
#plt.show()
plt.savefig('results/iot_multiclass.pdf',format='pdf',dpi=300,bbox_inches='tight')