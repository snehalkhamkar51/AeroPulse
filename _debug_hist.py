import io
from fastapi.testclient import TestClient
import backend.main as main
client = TestClient(main.app)
info = client.get("/model-info").json()
feats = info["feature_names"]
print("feats", feats)
samples = client.get("/sample-engines").json()
eng = samples["engines"][0]
rows = [(1, c, row) for c, row in zip(eng["cycles"][:8], eng["features"][:8])]
buf = io.StringIO()
buf.write("engine_id,cycle," + ",".join(feats) + "\n")
for engine_id, cycle, values in rows:
    buf.write(f"{engine_id},{cycle}," + ",".join(str(v) for v in values) + "\n")
csv_text = buf.getvalue()
print("header", csv_text.splitlines()[0])
import pandas as pd
df = pd.read_csv(io.StringIO(csv_text))
print("df columns", list(df.columns))
print("cycle col type", type(df["cycle"]))
