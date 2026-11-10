# Context-Aware Shortest-Path Healing System for Occlusion-Robust Road Network Reconstruction

A local web application that:
- extracts roads from a satellite or aerial image;
- finds genuine gaps where a road is hidden (by trees, shadows and similar);
- explains for every gap whether reconnecting it is supported by the evidence, in English and Tamil, on screen and by voice;
- reconstructs a gap only after you approve it, one gap at a time, with a graph-based trellis method (dynamic programming on a directed acyclic graph; no Dijkstra).

Built with Python, Flask, OpenCV, NumPy, scikit-image, NetworkX and ReportLab, plus a plain HTML/CSS/JavaScript interface. No deep learning, GPU, internet connection or external web libraries are needed.

---

## 1. Quick start on Windows 11

1. Install **Python 3.10–3.12** from python.org and tick **"Add python.exe to PATH"**.
2. Double-click **`run.bat`**.
   - The first start creates `.venv`, installs `requirements.txt` and copies `.env.example` to `.env`.
   - The browser opens at **http://127.0.0.1:5000**.
3. Stop the server with **Ctrl+C**.

**Tests:** `run_tests.bat`, or `python -m pytest -q` (57 tests, about 15 seconds).

**For Tamil voice:** use **Microsoft Edge**, which offers Tamil voices on Windows 11. Alternatively, add Tamil with speech under *Settings → Time & language → Language & region*. Without a Tamil voice the app shows Tamil captions and says that no voice is installed.

---

## 2. What changed in the latest update, and why

| Area | Before | Now | Why |
| --- | --- | --- | --- |
| Candidate gaps | Mutual-best endpoint pairs with distance, direction, width, crossing and water checks | Those checks plus a **road-to-road gate before CARS** (section 5) | Only genuine road-to-road disconnections may reach CARS |
| Road-to-road gate | – | Both fragments verified as road structures **from the image** (not the mask alone); trajectories joinable without an S-bend; a feasible continuous connection exists | Isolated ends, natural endings and unrelated nearby roads were still reaching CARS |
| Road-side (T) joins | On | **Off by default** (optional) | They create new intersections rather than closing a road-to-road gap |
| Review order | Detection order | **Gaps around possible occlusions first** (vegetation- or shadow-like cover ≥ 25%) | Requested priority |
| Reconstruction | Weighted Dijkstra (required by the previous brief) | **Graph-based trellis with dynamic programming** (section 8). The same cost terms are kept; Dijkstra is not used anywhere | This brief forbids Dijkstra and Dijkstra-based methods. The optional same-network detour now uses Bellman–Ford |
| Review interface | Four slides per gap (analysis, explanation, approval, path) | **One dedicated review screen per gap**: original image, enlarged gap, road mask, local road graph, proposed-path preview, measured cover, full analysis, Approve / Reject | Requested |
| Preview | Path known only after approval | The path that approval would draw is computed **before** the decision and shown as a dashed, labelled preview | Informed decisions |
| Navigation | Next blocked until a decision was made | Free Previous/Next between gaps; an undecided gap stays **pending** (with a notice) and can be revisited; navigation is locked while a reconstruction runs; nothing advances automatically | Requested |
| CARS formula and weights | wA 0.25, wO 0.30, wW 0.10, wG 0.10, wN 0.25; threshold 0.65 | **Unchanged** | As required |

**Earlier changes (still in place):**
- the contextual road extraction;
- the water-like cue, which counts as no evidence;
- bilingual explanations and voice guidance;
- the interactive graph, before/after comparison, map explorer and PDF report.

## 3. The workflow (slides)

1. **Original image:** with file details. The original is never modified.
2. **Road mask:** side by side with the original. *How it was extracted* shows the four intermediate steps.
3. **Road graph:** the actual NetworkX graph, interactive (section 9).
4. **Road fragments:** connected components with their measured length, width and image road-likelihood, plus open ends and border ends.
5. **Road-to-road candidate gaps:** the list (occlusion-region gaps first) with recommendations, and every excluded endpoint pair with its reason.
6. **Gap review:** one dedicated screen per gap.
7. **Final road network:** the original image, original mask, every gap by status, the final mask and the final network, plus totals and the PDF button.
8. **Before and after:** synchronised side-by-side zoom and pan, or a reveal slider.
9. **Map explorer:** layers, plus gap markers that open the gap's review screen.

### The gap review screen

The header shows the gap id, "candidate *n* of *N*", the recommendation, the decision, the region priority, and a "*k* of *N* decided" progress bar.

| View | Content (all drawn from real data) |
| --- | --- |
| **A. Original image** | Whole image; fragment A tinted cyan, fragment B magenta; both road ends marked; the gap region outlined; measured vegetation- or shadow-like pixels around the gap outlined as "possible occlusion" (only when measured) |
| **B. Enlarged gap** | Crop containing both fragments and the whole gap; A/B labels; arrows along each fragment's direction; the endpoint distance drawn and labelled |
| **C. Road mask** | Fragment A cyan, B magenta, other roads white; ends and the candidate gap. After approval: the updated mask with the new segment in green |
| **D. Local road graph** | The real graph edges and nodes around the gap; fragment edges coloured, other edges grey, junctions white, endpoints yellow; the proposed connection dashed orange. After approval: the reconstructed edge solid green |
| **E. Proposed path** | The connection approval would draw, as a dashed orange overlay labelled **"PREVIEW – not reconstructed"**, with a Gap/Proposed toggle to compare. After approval: the actual result labelled **"RECONSTRUCTED"** |
| Measured cover | The cover categories measured inside and around the gap |

Any view opens in a zoomable viewer.

**The analysis panel lists:**
- the gap id;
- fragment A and B: length, width, road-likelihood in the image, end id and heading;
- the endpoint distance;
- the direction difference;
- width consistency;
- occlusion evidence and the environment (with cover bars);
- graph connectivity;
- CARS against the threshold;
- the recommendation;
- the explanation;
- the full CARS calculation.

**Approve / Reject:**
- **Approve connection** reconstructs only this gap. The success message appears only after the server confirms the reconstruction, and C, D and E switch to the actual result.
- **Reject connection** records the rejection and leaves the network unchanged.
- Neither action advances automatically: you continue with **Next**.
- Approval is disabled for not-recommended gaps, with the reason shown.
- A rejected gap can be reconsidered later; a reconstructed one cannot be changed.

## 4. Road extraction (no trained model)

The **contextual** method combines cues instead of relying on colour alone:

1. **Exclusions.** These are never road:
   - vegetation-like pixels (excess green > 0.08);
   - water-like pixels (blue exceeding red and green by 12);
   - very dark pixels (V < 60).
2. **Loose colour gate.** Greyish, mid-bright pixels become candidates.
3. **Width filter (top-hat).** A morphological opening with a disc as wide as the widest road (32 px) finds regions *wider* than any road, such as open ground, fields, yards and car parks. Those are removed, including where they touch a road.
4. **Road colour model.** A Lab mean and covariance is learned from strict seed pixels of *this* image, giving a road likelihood per pixel.
5. **Line-likeness.** A Sato ridge filter on that likelihood favours long thin structures over patches.
6. **Shape filter.** Each component's length/width is measured from its skeleton. Compact shapes (below 3.0, such as roofs) and over-wide components are dropped. Road borders are then regrown up to 3 px into allowed road-coloured pixels.

**Measured on the synthetic scenes** (visible roads as ground truth):
- On the distractor scene (bare grey ground, grey roof, car park, river), mask precision is **0.27 with the old threshold and 1.00 with contextual**.
- Over all 18 scenes, precision is **0.87 → 0.99** and IoU **0.85 → 0.92**, while recall drops **0.98 → 0.93** because road edges are trimmed slightly.

**Limitations:**
- No trained segmentation model is included.
- Long grey roofs, concrete channels, grey riverbanks and dirt tracks can still be misjudged.
- Roads under dense canopy are missed, which is what the gap review is for.
- You can upload a better road mask instead; it replaces this step.

## 5. Genuine road-to-road gaps

Each item below is checked, and every excluded pair is listed with its reason.

**Geometric checks.**
1. The ends belong to **different** connected fragments, and neither is at the image border (a border end is a road leaving the image).
2. They are 1.5–70 px apart and **each points within 60° of the other**.
3. Their widths are compatible: narrower/wider ≥ 0.25.
4. **No other road** lies between them.
5. The gap is **not mostly water-like**.

**Road-to-road gate.** This decides what is allowed to reach CARS.

6. **Both fragments are identifiable road structures in the image:**
   - visible length (skeleton + one width) ≥ 20 px;
   - elongated (length/width ≥ 2.5, unless the fragment runs off the image edge);
   - mean road-colour likelihood of the *image* along the fragment ≥ 0.30;
   - likelihood along the last 20 px before each road end ≥ 0.25.

   The road mask alone is not enough. Without an image the fragments are "unverified", and the gap can only be *uncertain*.
7. **Trajectory continuity:** the two road directions can be joined without an S-shaped bend (bend mismatch ≤ 50°).

**Pairing.**

8. **Mutual best match:** each end is the other's best partner.

**Feasibility.**

9. **A continuous connection is constructible:** the trellis search (section 8) finds a valid path. That path becomes the preview.

**Review order.** Remaining gaps are numbered G01, G02, … with **possible-occlusion gaps first** (vegetation- or shadow-like ≥ 25% of the gap), then gaps where road is still visible, then others.

**Optional.** End-to-road-side (T) joins are off by default.

**Never analysed.**
- isolated ends with no compatible partner;
- natural dead ends;
- intersections;
- separate branches;
- every leftover mask component;
- non-road regions (removed during extraction).

## 6. CARS (unchanged)

```
CARS = 0.25·A + 0.30·O + 0.10·W + 0.10·G + 0.25·N      (scores in [0, 1], weights normalised)
```

| Feature | What it measures |
| --- | --- |
| A, alignment | Ends facing each other along a straight line or a smooth arc; S-bends and offsets score low |
| O, occlusion evidence | Cover inside the gap: road-like 0.9, vegetation-like 1.0, shadow-like 1.0, textured 0.5, water-like 0, other 0 |
| W, width consistency | 1 − relative width difference / 0.8 |
| G, local geometry | Tightest turn radius versus road width, and curvature continuity with the fragments |
| N, network connectivity | 0.75 for joining separate fragments; lowered if it crosses a road or competes with another gap |

**Environment information.** The cover in a ring *around* the gap is measured and shown in the analysis and the explanation. The cover *inside* the gap enters CARS through O. The environment ring is not given its own weight, so the existing formula stays unchanged.

**Missing evidence.** It counts as 0.5 (the neutral policy), never as strong positive evidence. A gap whose score clears the threshold but has missing evidence is marked uncertain.

**Recommendation:**
- **Recommended:** CARS ≥ 0.65 with complete evidence.
- **Uncertain:**
  - CARS is between 0.60 and 0.65 (the review band);
  - or evidence is missing;
  - or a higher-scoring gap competes for the same road end (only one can be approved).
- **Not recommended:** everything else.

## 7. Explanations ("Explainable AI")

CARS is a transparent weighted score, not a trained model. `core/explanations.py` turns each gap's **measured values** into statements, using fixed rules, for example:
- "Vegetation-like (green) pixels cover 94% of the gap…" appears only when that fraction is at least 25%;
- "The road ends have inconsistent orientations: they deviate 36°…" appears when alignment is low.

Each statement is tagged as supporting, against, neutral, uncertain or context. The cues are colour and texture heuristics, so the wording says "vegetation-like" or "shadow-like" and never names objects such as trees or buildings. Every statement is produced in English and Tamil from the same values, so screen and voice always agree.

## 8. Approval and one-by-one reconstruction (graph-based trellis, no Dijkstra)

**The trellis graph** (`core/trellis_reconstruction.py`):

1. **Expected course:** a cubic Hermite curve leaving fragment A along its road direction and arriving along fragment B's.
2. **Stations:** every 3 px along that course. Each station offers lateral positions 1 px apart, up to 1.5 × road width (at least 6 px) either side. The first and last stations are pinned to the two road ends.
3. **Graph:** a NetworkX DiGraph whose nodes are (station, lateral position). Edges join consecutive stations whose positions differ by at most 2 px, so the route stays continuous. The graph is acyclic by construction.
4. **Costs** (the same terms as the earlier path cost, all ≥ 0):

   ```
   node:  (1 − road likelihood) + 2·min((offset / s)², 9) + 6·[on an unrelated road]
   edge:  step · (1 + mean node cost) + step · (1 − cos(step heading − course tangent)) / 2
   ```

5. **Dynamic programming:** nodes are processed in topological order (station by station), and each edge is relaxed exactly once. On a directed acyclic graph this gives the exact minimum-cost route in time proportional to nodes + edges. No priority queue, Dijkstra or A* is involved.
6. **Path checks:** length ≤ 1.8 × gap; mean cost ≤ 3.0 per pixel (the **path cost threshold**, separate from CARS); at most 15% on unrelated roads.

**What approval does.** The same computation runs before the decision (the preview) and again on approval, using the **current** mask so that earlier approvals count as road. On approval:
- the path is drawn at the measured road width;
- the two fragments must then actually be connected;
- **only then** are the mask, road graph and component counts updated and saved.

A failed or blocked attempt leaves everything unchanged and is never counted as a success.

**The explanation after reconstruction reports:**
- the number of stations, positions, nodes and edges;
- the total and per-pixel cost;
- length versus gap;
- the sideways deviation from the expected course;
- the cost breakdown;
- the cover along the new section;
- the cost of following the expected course exactly, and why the chosen route was cheaper (or that the expected course itself was cheapest).

## 9. Voice, graph, comparison, map and PDF

**Voice guidance.** Controls: Enable/Mute, English/தமிழ், Auto, Replay and Stop.
- **What is spoken:**
  - stage announcements (upload, mask, graph, fragments, gaps);
  - each gap's evidence;
  - its CARS score and meaning;
  - the decision prompt;
  - "reconstructing with the graph-based trellis method" when you approve;
  - the result, only *after* the server confirms it;
  - rejection or skip notices;
  - the final summary with actual counts.
- **No overlaps:** each new utterance stops the previous one, except the reconstruction result, which waits for the "reconstructing…" sentence to finish.
- **Captions:** the caption bar always shows the text being spoken.
- **Engine:** the browser's speech engine (Web Speech API), working offline with installed voices. If speech fails or no Tamil voice exists, the status bar says so and captions continue.

**Interactive graph.**
- **What it shows:** the real NetworkX graph in image coordinates over the image. Edges follow their road geometry and are coloured by component; reconstructed edges are thick green; endpoints are yellow (grey at the border) and junctions magenta.
- **Clicking a node or edge** shows its id, position, neighbours, length (the weight), path cost and gap id, and zooms to that edge's image region.
- **Components** can be highlighted, and statistics are shown for the original and current state.

**Before and after.** The comparison uses the actual outputs and counts only validated reconstructions.

**Map explorer.** Coordinates are image pixels; no geographic coordinates are claimed.

**PDF report** (*Reports → Generate PDF report*, or the button on the final slide):
- project and input details, software versions and methods;
- the image and mask;
- a section per gap: images, CARS table, evidence, recommendation;
- user decisions with timestamps;
- reconstruction results with path explanations;
- before/after views and the final graph;
- statistics.

Every page shows "Page X of Y". Missing items say "not available". The report is in English; Tamil would need an embedded Tamil font, which is not bundled.

---

## 10. Output folders

```
storage/jobs/<job_id>/
  input/        uploads exactly as received (never modified)
  stages/       original, road_mask, seg_* steps, skeleton_endpoints, road_graph, fragments,
                candidate_gaps, recommendations, decisions, before_overlay, final_overlay, current_mask
  gaps/G01/     original_view, enlarged, mask_view, graph_view, preview, evidence
                (+ result, mask_after, graph_after once reconstructed), analysis.json
  layers/       RGBA map layers: mask, roads, environment, reconstructed, current_mask
  masks/        original_road_mask, current_road_mask, reconstructed_only
  reports/      gap_analysis.csv, user_decisions.csv, fragments.csv, excluded_pairs.csv, endpoints.csv,
                evaluation_metrics.json/.csv, road_graph.json, results.json, configuration.json,
                processing_log.txt, pdf/report_<date-time>.pdf
storage/history.json
```

## 11. API

All errors are JSON `{"error": …}` with 400, 404, 409, 413 or 500.

| Method | Path | Purpose |
| --- | --- | --- |
| POST | `/api/jobs` | upload (image, optional mask, optional reference) |
| POST | `/api/jobs/sample` | create a job from a test scene |
| POST | `/api/jobs/<id>/run` | analyse (stages 1–9) with `{"params": {...}}` |
| GET | `/api/jobs/<id>/status` · `/results` | live status · everything shown |
| POST | `/api/jobs/<id>/gaps/<gap>/decision` | `{"action": "approve" \| "reject" \| "skip"}` |
| GET | `/api/jobs/<id>/graph` | current graph: nodes, edges, components, statistics |
| POST | `/api/jobs/<id>/report` | generate a PDF |
| GET | `/api/jobs/<id>/files/<path>` · `/download/<file>` | whitelisted images · reports, PDFs, `all.zip` |
| GET | `/api/config` · `/api/samples` · `/api/history` | form schema · test scenes · history |

## 12. Results on the synthetic test data

All test scenes are procedurally generated (`samples/`, each with `meta.json` marked `"synthetic": true`). **None of these numbers are results on real satellite imagery.**

**Named scenarios.** All 10 scene scenarios behave as expected with the gapped mask and when extracting roads from the image, once the recommended or uncertain gaps are approved.
- True gaps under trees, shadows and mask dropouts are reconnected.
- The offset S-bend dead ends (scene 3) are now stopped at the road-to-road gate.
- The width mismatch and unrelated track reach CARS as genuine road-to-road pairs and are *not recommended*, with reasons.
- Far-apart roads give no candidates.
- The corrupted file (scenario 9) is refused with a clear message.

**24 randomized scenes** contain 48 true gaps and 21 traps: dead-end streets aimed at another road that stop 30–60 px short. The user approved recommended gaps only.

| Input | Recommended / uncertain / not recommended | Rebuilt | Gap precision | Gap recall | Pairs excluded before CARS |
| --- | --- | --- | --- | --- | --- |
| gapped masks | 45 / 1 / 0 | 45 | 0.992 | 0.900 | 19 |
| roads extracted from the image | 28 / 0 / 0 | 28 | 0.995 | 0.401 | 20 |

With extracted roads, gap recall is low mainly because "missing road" also counts the road-edge pixels the extraction trims across the whole image. Gap reconstruction cannot recover those, so read that figure as a segmentation limit rather than as missed gaps.

**Speed.** On a 400 × 400 scene, analysis takes about 0.1–0.3 s, and each approved trellis reconstruction well under 0.1 s.

## 13. Limitations

- **Extraction is heuristic.** Road extraction and occlusion cues are colour, texture and shape rules, not learned models.
- **Defaults are design parameters.** Defaults (weights, thresholds, extraction settings) were checked on synthetic scenes only. Validate them on your own images with reference masks.
- **Gaps inside a connected network are off by default.** A gap where the network stays connected around it is only considered if *Also look inside a connected network* is enabled.
- **Tamil voice depends on the browser and Windows voices.**
- **The PDF is English only.**

## 14. Project structure

```
app.py · config.py · requirements.txt · .env.example · run.bat · run_tests.bat · README.md
core/  preprocessing.py  segmentation.py  skeletonization.py  endpoint_extraction.py  graph_construction.py
       gap_detection.py (incl. road-to-road gate)  context_features.py  cars_scoring.py  decision_engine.py
       explanations.py  gap_visuals.py  trellis_reconstruction.py (graph-based, no Dijkstra)
       reconstruction.py (preview, one-by-one approval, validation)
       evaluation.py  visualization.py  export.py  pdf_report.py  pipeline.py  synthetic_data.py
       geometry.py  errors.py  utils.py
web/   jobs.py (job folders, background analysis, decisions, history)  routes.py (REST API)
templates/index.html   static/css/styles.css
static/js/  app.js (carousel, review, views)  voice.js (bilingual voice)  viewer.js (zoom/pan)  graph.js (graph)
samples/    13 synthetic scenes + 1 corrupted file      scripts/generate_samples.py
tests/      57 tests: pipeline, segmentation, gap detection and gate, CARS, trellis reconstruction (no Dijkstra),
            explanations, evaluation, API, PDF
```
