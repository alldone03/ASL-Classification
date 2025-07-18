import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df = pd.read_csv("hand_features.csv")

# Hitung jumlah data per kelas
label_counts = df['label'].value_counts()

# Tampilkan di terminal
print("Jumlah sampel per kelas:")
print(label_counts)

# Plot bar chart
plt.figure(figsize=(10, 5))
sns.barplot(x=label_counts.index, y=label_counts.values, palette="viridis")
plt.xlabel("Kelas")
plt.ylabel("Jumlah Sampel")
plt.title("Sebaran Data Gesture Tangan")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()