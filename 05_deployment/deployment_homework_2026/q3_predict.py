
import hashlib
import json
import pickle
from pathlib import Path
from model import normalize_record

# Check the artifact again immediately before unpickling.
model_bytes = Path("pipeline.bin").read_bytes()
expected = "1646bbdcd38d4f044da6b630c5b332c93a314245a8c21929011c42de51f629f1"
assert hashlib.sha256(model_bytes).hexdigest() == expected
pipeline = pickle.loads(model_bytes)

lead = json.loads(Path("lead_q3.json").read_text())
record = normalize_record(lead)
classes = list(pipeline.classes_)
positive_column = classes.index(1)
probabilities = pipeline.predict_proba([record])
probability = float(probabilities[0, positive_column])
assert 0 <= probability <= 1
print(json.dumps({
    "conversion_probability": probability,
    "answer_to_three_decimals": f"{probability:.3f}",
    "class_order": [int(value) for value in classes],
    "pipeline_steps": list(pipeline.named_steps),
}))
