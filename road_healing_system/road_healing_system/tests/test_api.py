"""Flask API: full flow, reports, and error handling."""

import io
import time

import cv2
import numpy as np

from core.synthetic_data import SCENARIOS


def png(img):
    return cv2.imencode(".png", img)[1].tobytes()


def wait(client, jid):
    for _ in range(600):
        st = client.get(f"/api/jobs/{jid}/status").get_json()
        if st["state"] != "running":
            return st
        time.sleep(0.05)
    raise AssertionError("job did not finish")


def new_job(client, name="07_multiple_components", use_mask=False):
    jid = client.post("/api/jobs/sample", json={"name": name, "use_mask": use_mask}).get_json()["status"]["job_id"]
    assert client.post(f"/api/jobs/{jid}/run", json={"params": {}}).status_code == 202
    return jid, wait(client, jid)


def test_full_review_flow_graph_and_pdf(client):
    jid, st = new_job(client)
    assert st["state"] == "review" and st["stages"][-2]["status"] == "waiting"
    res = client.get(f"/api/jobs/{jid}/results").get_json()["results"]
    gaps = res["gaps"]
    assert gaps and res["voice"]["mask"]["ta"] and gaps[0]["xai"]["voice"]["analysis"]["ta"]
    r = client.post(f"/api/jobs/{jid}/gaps/{gaps[0]['id']}/decision", json={"action": "approve"}).get_json()
    assert r["outcome"]["recon_status"] == "reconstructed"
    for n in ("original_view", "enlarged", "mask_view", "graph_view", "preview", "result", "mask_after", "graph_after"):
        assert f"gaps/{gaps[0]['id']}/{n}.png" in r["status"]["files"], n
    for g in gaps[1:]:
        client.post(f"/api/jobs/{jid}/gaps/{g['id']}/decision", json={"action": "reject"})
    assert client.get(f"/api/jobs/{jid}/status").get_json()["state"] == "done"
    graph = client.get(f"/api/jobs/{jid}/graph").get_json()
    assert sum(e["kind"] == "reconstructed" for e in graph["edges"]) == 1
    assert graph["stats"]["after"]["components"] == graph["stats"]["before"]["components"] - 1
    rep = client.post(f"/api/jobs/{jid}/report").get_json()
    pdf = client.get(rep["url"])
    assert pdf.status_code == 200 and pdf.data[:5] == b"%PDF-"
    for f in ("gap_analysis.csv", "user_decisions.csv", "results.json", "all.zip"):
        assert client.get(f"/api/jobs/{jid}/download/{f}").status_code == 200
    assert client.get(f"/api/jobs/{jid}/files/layers/reconstructed.png").status_code == 200


def test_decision_errors(client):
    jid, _ = new_job(client, "04_width_mismatch", use_mask=True)
    r = client.post(f"/api/jobs/{jid}/gaps/G01/decision", json={"action": "approve"})
    assert r.status_code == 409 and "not recommended" in r.get_json()["error"]
    assert client.post(f"/api/jobs/{jid}/gaps/G01/decision", json={"action": "maybe"}).status_code == 400
    assert client.post(f"/api/jobs/{jid}/gaps/G99/decision", json={"action": "reject"}).status_code == 400
    img = png(SCENARIOS["01_straight_gap"]().image)
    jid2 = client.post("/api/jobs", data={"image": (io.BytesIO(img), "a.png")}, content_type="multipart/form-data").get_json()["status"]["job_id"]
    assert client.post(f"/api/jobs/{jid2}/gaps/G01/decision", json={"action": "approve"}).status_code == 400
    assert client.post(f"/api/jobs/{jid2}/report").status_code == 400


def test_upload_errors(client):
    assert client.post("/api/jobs", data={}).status_code == 400
    r = client.post("/api/jobs", data={"image": (io.BytesIO(b"abc"), "x.gif")}, content_type="multipart/form-data")
    assert r.status_code == 400 and "Unsupported" in r.get_json()["error"]
    r = client.post("/api/jobs", data={"image": (io.BytesIO(b"\x89PNG broken"), "x.png")}, content_type="multipart/form-data")
    assert r.status_code == 400 and "could not be decoded" in r.get_json()["error"]
    img = png(SCENARIOS["01_straight_gap"]().image)
    r = client.post("/api/jobs", data={"image": (io.BytesIO(img), "a.png"), "mask": (io.BytesIO(png(np.zeros((9, 9), np.uint8))), "m.png")},
                    content_type="multipart/form-data")
    assert r.status_code == 400 and "must match" in r.get_json()["error"]
    big = b"\x89PNG" + b"0" * (3 * 1024 * 1024)
    r = client.post("/api/jobs", data={"image": (io.BytesIO(big), "b.png")}, content_type="multipart/form-data")
    assert r.status_code == 413


def test_ids_params_and_whitelists(client):
    assert client.get("/api/jobs/nope/status").status_code == 404
    assert client.get("/api/jobs/" + "0" * 32 + "/results").status_code == 404
    assert client.post("/api/jobs/sample", json={"name": "../config"}).status_code == 404
    img = png(SCENARIOS["01_straight_gap"]().image)
    jid = client.post("/api/jobs", data={"image": (io.BytesIO(img), "a.png")}, content_type="multipart/form-data").get_json()["status"]["job_id"]
    assert client.post(f"/api/jobs/{jid}/run", json={"params": {"cars.acceptance_threshold": 3}}).status_code == 400
    assert client.post(f"/api/jobs/{jid}/run", json={"params": {"cars.nope": 1}}).status_code == 400
    assert client.post(f"/api/jobs/{jid}/run", data="{bad", content_type="application/json").status_code == 400
    assert client.get(f"/api/jobs/{jid}/files/../config.py").status_code == 404
    assert client.get(f"/api/jobs/{jid}/files/stages/secret.png").status_code == 404
    assert client.get(f"/api/jobs/{jid}/download/../app.py").status_code == 404
    assert client.get("/").status_code == 200
    keys = {i["key"] for g in client.get("/api/config").get_json()["groups"] for i in g["items"]}
    assert {"segmentation.method", "path.max_mean_cost", "gap.mutual_best"} <= keys
