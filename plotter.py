import scienceplots

import matplotlib
import matplotlib.pyplot as plt
import scipy.stats
plt.style.use(['science','no-latex'])

matplotlib.rcParams.update({'font.size': 18})

import pandas as pd
import matplotlib.pyplot as plt


# Data CIC 
data = {
    "8 features - 1 LUT": {
        "4-bit": 0.847601592540741,
        "6-bit": 0.8493005037307739,
        "8-bit": 0.9059754014015198,
    },
    "2 features - 1 LUT": {
        "4-bit": 0.7145018577575684,
        "6-bit": 0.7192800641059875,
        "8-bit": 0.7194924354553223,
        "16-bit": 0.717182993888855,
        "32-bit": 0.718112051486969,
    },
    "6 features - 5 LUT": {
        "4-bit": 0.6635873913764954,
        "6-bit": 0.84027498960495,
        "8-bit": 0.8748108744621277,
        "16-bit": 0.8707228302955627,
        "32-bit": 0.8687584400177002,
    },
    "8 features - 7 LUT": {
        "4-bit": 0.7355525493621826,
        "6-bit": 0.7791404724121094,
        "8-bit": 0.850043773651123,
        "16-bit": 0.8569079446792603,
        "32-bit": 0.8701842522621155,
    }
}
'''

# DATA Traffic classification
data = {
    "8 features - 1 LUT": {
        "4-bit": 0.8474571108818054,
        "6-bit": 0.8927105665206909,
        "8-bit": 0.9159387946128845,
    },
    "2 features - 1 LUT": {
        "4-bit": 0.8372840285301208,
        "6-bit": 0.8370901942253113,
        "8-bit": 0.8373369574546814,
        "16-bit": 0.8376283645629883,
        "32-bit": 0.8375728130340576,
    },
    "6 features - 5 LUT": {
        "4-bit": 0.8152152895927429,
        "6-bit": 0.8316896557807922,
        "8-bit": 0.8344212770462036,
        "16-bit": 0.8505409359931946,
        "32-bit": 0.8577998876571655,
    },
    "8 features - 7 LUT": {
        "4-bit": 0.8521140813827515,
        "6-bit": 0.8847944140434265,
        "8-bit": 0.8804467606544495,
        "16-bit": 0.8993271589279175,
        "32-bit": 0.9126664996147156,
    }
}
'''
# Convert data to DataFrame
df_list = []
for key, values in data.items():
    df = pd.DataFrame(list(values.items()), columns=['Bit Size', key])
    df_list.append(df.set_index('Bit Size'))

# Combine all dataframes
final_df = pd.concat(df_list, axis=1)

# Plot
plt.figure(figsize=(10, 6))
for column in final_df.columns:
    plt.plot(final_df.index, final_df[column], marker='o', label=column)

plt.title('Performance Metrics by Configuration and Bit Size')
plt.xlabel('Bit Size')
plt.ylabel('F1-Score')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()