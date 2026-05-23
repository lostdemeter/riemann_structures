"""ResonantFeatureHash demo."""
from resonant.featurehash import ResonantFeatureHash

fh = ResonantFeatureHash(n_features=100)

features = ["color=red", "shape=circle", "size=large", "color=red"]
for f in set(features):
    idx, sgn = fh.signed(f)
    print(f"  {f} → index={idx}, sign={sgn:+d}")

vec = fh.feature_vector(features)
print(f"\n  Feature vector: {vec}")
