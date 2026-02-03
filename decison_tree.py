import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
import matplotlib.pyplot as plt

# ---------------------------
# Create Color Dataset (RGB)
# ---------------------------
data = {
    "R": [255, 200, 0,   0,   255, 255, 0,   0,   128, 255],
    "G": [0,   0,   255, 200, 255, 255, 255, 0,   128, 255],
    "B": [0,   0,   0,   0,   0,   255, 0,   255, 128, 255],
    "color": [
        "Red", "Red", "Green", "Green",
        "Yellow", "White", "Green", "Blue",
        "Gray", "White"
    ]
}

df = pd.DataFrame(data)
print("Dataset Sample:")
print(df)

# ---------------------------
# Split data
# ---------------------------
X = df[["R", "G", "B"]]
y = df["color"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# ---------------------------
# Train Decision Tree
# ---------------------------
clf = DecisionTreeClassifier(
    criterion="entropy",
    max_depth=3,
    random_state=42
)

clf.fit(X_train, y_train)

# ---------------------------
# Accuracy
# ---------------------------
accuracy = clf.score(X_test, y_test)
print(f"\nAccuracy on test data: {accuracy*100:.2f}%")

# ---------------------------
# Print Tree Structure
# ---------------------------
tree_rules = export_text(clf, feature_names=["R", "G", "B"])
print("\nDecision Tree Structure:\n")
print(tree_rules)

# ---------------------------
# Plot Tree
# ---------------------------
plt.figure(figsize=(14,8))
plot_tree(
    clf,
    feature_names=["R", "G", "B"],
    class_names=clf.classes_,
    filled=True,
    rounded=True
)
plt.show()

# ---------------------------
# Predict a New Color
# ---------------------------
new_color = [[255, 255, 0]]   # Yellow
prediction = clf.predict(new_color)

print(f"\nPredicted color for RGB {new_color[0]}: {prediction[0]}")
