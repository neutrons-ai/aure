/* ================================================================
   AuRE – setup.js
   Client-side logic for the interactive setup page:
     • Server-side file / folder browser (modal)
     • Analysis launch + status polling
   ================================================================ */

/* ---- state --------------------------------------------------- */
let browserMode = "file";        // "file" | "dir"
let browserCurrentPath = null;
let browserParentPath = null;
let browserModalInstance = null;
let pollTimer = null;

const KNOWN_NODES = ["intake", "analysis", "modeling", "fitting", "evaluation"];

const COLORS = [
  "#6c757d", "#0d6efd", "#198754", "#dc3545",
  "#fd7e14", "#6610f2", "#20c997", "#d63384",
];
let _liveResultsFetched = false;  // avoid re-fetching while still waiting

const STORAGE_KEY = "aure_setup";
const LAST_DATA_DIR_KEY = "aure_last_data_dir";
const REFLECTIVITY_EXTS = ".txt,.refl,.ort,.dat,.csv,.tsv";

let plottedFiles = []; // [{ path, Q, R, dR, visible, isFit }]

function _parentDir(path) {
  if (!path) return "";
  const idx = path.lastIndexOf("/");
  return idx > 0 ? path.slice(0, idx) : "";
}

function _saveLastDataDirFromFile(filePath) {
  const dir = _parentDir(filePath);
  if (!dir) return;
  try { localStorage.setItem(LAST_DATA_DIR_KEY, dir); } catch (_) {}
}

function _getLastDataDir() {
  try {
    const saved = localStorage.getItem(LAST_DATA_DIR_KEY);
    if (saved) return saved;
  } catch (_) {}

  const currentDataFile = (document.getElementById("data-file") || {}).value || "";
  return _parentDir(currentDataFile.trim());
}

/* ---- persist / restore form values --------------------------- */

function _saveFormValues() {
  const vals = {
    data_file: document.getElementById("data-file").value,
    sample_desc: document.getElementById("sample-desc").value,
    hypothesis: document.getElementById("hypothesis").value,
    output_dir: document.getElementById("output-dir").value,
    interactive: document.getElementById("interactive-mode").checked,
    max_iterations: parseInt(document.getElementById("max-iterations").value, 10) || 5,
  };
  try { localStorage.setItem(STORAGE_KEY, JSON.stringify(vals)); } catch (_) {}
}

function _restoreFormValues() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return;
    const vals = JSON.parse(raw);
    if (vals.data_file)   document.getElementById("data-file").value = vals.data_file;
    if (vals.sample_desc) document.getElementById("sample-desc").value = vals.sample_desc;
    if (vals.hypothesis)  document.getElementById("hypothesis").value = vals.hypothesis;
    if (vals.output_dir)  document.getElementById("output-dir").value = vals.output_dir;
    if (vals.interactive)  document.getElementById("interactive-mode").checked = vals.interactive;
    if (vals.max_iterations) document.getElementById("max-iterations").value = vals.max_iterations;
  } catch (_) {}
}

document.addEventListener("DOMContentLoaded", function () {
  _restoreFormValues();
  const restoredDataFile = document.getElementById("data-file").value.trim();
  if (restoredDataFile) {
    _setFittingDataFile(restoredDataFile);
  } else {
    _renderPlottedFilesList();
    _renderSetupReflectivityPlot();
  }
});

/* ---- LLM badge helper ---------------------------------------- */

function _llmBadges(calls) {
  if (!calls || calls.length === 0) return '<span class="text-muted">—</span>';
  return calls.map(function(c) {
    if (!c.success)
      return '<span class="badge bg-danger" title="' + (c.error || '').replace(/"/g, '&quot;') + '">✗ failed</span>';
    if (c.used_fallback)
      return '<span class="badge bg-warning text-dark" title="' + (c.fallback_reason || '').replace(/"/g, '&quot;') + '">⚠ fallback</span>';
    return '<span class="badge bg-success">✓ ok</span>';
  }).join(" ");
}

/* ---- file / folder browser ----------------------------------- */

function openBrowser(mode) {
  browserMode = mode;
  if (mode === "file") {
    document.getElementById("browser-title").textContent = "Select Data File";
  } else if (mode === "compare-file") {
    document.getElementById("browser-title").textContent = "Add Comparison File";
  } else {
    document.getElementById("browser-title").textContent = "Select Output Folder";
  }

  // Show or hide "Select this folder" button
  document.getElementById("btn-select").style.display =
    mode === "dir" ? "inline-block" : "none";

  // Start at the last data folder for file selection, otherwise server default
  const startPath = (mode === "file" || mode === "compare-file")
    ? _getLastDataDir()
    : "";
  _fetchBrowserListing(startPath);

  browserModalInstance =
    browserModalInstance ||
    new bootstrap.Modal(document.getElementById("browserModal"));
  browserModalInstance.show();
}

function _fetchBrowserListing(path) {
  const endpoint = (browserMode === "file" || browserMode === "compare-file")
    ? "/api/browse-files"
    : "/api/browse-dirs";
  const params = new URLSearchParams();
  if (path) params.set("path", path);
  if (browserMode === "file" || browserMode === "compare-file") {
    params.set("ext", REFLECTIVITY_EXTS);
  }

  fetch(`${endpoint}?${params}`)
    .then((r) => r.json())
    .then((data) => {
      if (data.error) {
        alert(data.error);
        return;
      }
      browserCurrentPath = data.current;
      browserParentPath = data.parent;
      document.getElementById("browser-path").textContent = data.current;
      document.getElementById("btn-parent").disabled = !data.parent;

      const list = document.getElementById("browser-list");
      list.innerHTML = "";

      data.entries.forEach((entry) => {
        const a = document.createElement("a");
        a.className = "list-group-item list-group-item-action d-flex align-items-center";
        a.href = "#";

        const icon = document.createElement("i");
        icon.className = entry.is_dir
          ? "bi bi-folder-fill text-warning me-2"
          : "bi bi-file-earmark-text me-2";
        a.appendChild(icon);

        const name = document.createElement("span");
        name.textContent = entry.name;
        a.appendChild(name);

        a.addEventListener("click", (e) => {
          e.preventDefault();
          if (entry.is_dir || entry.is_dir === undefined) {
            // Navigate into directory
            _fetchBrowserListing(entry.path);
          } else {
            // File selected
            if (browserMode === "compare-file") {
              _addComparisonFile(entry.path);
            } else {
              _setFittingDataFile(entry.path);
            }
            _saveLastDataDirFromFile(entry.path);
            browserModalInstance.hide();
          }
        });

        list.appendChild(a);
      });

      if (data.entries.length === 0) {
        const empty = document.createElement("div");
        empty.className = "list-group-item text-muted text-center";
        empty.textContent = browserMode === "file"
          ? "No matching files in this directory"
          : "No sub-folders";
        list.appendChild(empty);
      }
    })
    .catch((err) => console.error("Browse error:", err));
}

function browserUp() {
  if (browserParentPath) {
    _fetchBrowserListing(browserParentPath);
  }
}

function browserSelect() {
  // Folder mode – select the current directory
  if (browserMode === "dir" && browserCurrentPath) {
    document.getElementById("output-dir").value = browserCurrentPath;
    browserModalInstance.hide();
  }
}

function _setFittingDataFile(path) {
  document.getElementById("data-file").value = path;

  plottedFiles.forEach(function (f) { f.isFit = false; });
  const existing = plottedFiles.find(function (f) { return f.path === path; });
  if (existing) {
    existing.isFit = true;
    existing.visible = true;
    _renderPlottedFilesList();
    _renderSetupReflectivityPlot();
    return;
  }

  _loadReflectivityFile(path)
    .then(function (payload) {
      plottedFiles.push({
        path: path,
        Q: payload.Q || [],
        R: payload.R || [],
        dR: payload.dR || [],
        visible: true,
        isFit: true,
      });
      _renderPlottedFilesList();
      _renderSetupReflectivityPlot();
    })
    .catch(function (err) {
      alert("Could not load selected reflectivity file:\n" + err.message);
    });
}

function _addComparisonFile(path) {
  const existing = plottedFiles.find(function (f) { return f.path === path; });
  if (existing) {
    existing.visible = true;
    _renderPlottedFilesList();
    _renderSetupReflectivityPlot();
    return;
  }

  _loadReflectivityFile(path)
    .then(function (payload) {
      plottedFiles.push({
        path: path,
        Q: payload.Q || [],
        R: payload.R || [],
        dR: payload.dR || [],
        visible: true,
        isFit: false,
      });
      _renderPlottedFilesList();
      _renderSetupReflectivityPlot();
    })
    .catch(function (err) {
      alert("Could not load reflectivity file:\n" + err.message);
    });
}

function _loadReflectivityFile(path) {
  const params = new URLSearchParams({ path: path });
  return fetch("/api/reflectivity-file?" + params.toString())
    .then(function (r) {
      return r.json().then(function (data) {
        if (!r.ok || data.error) {
          throw new Error(data.error || "Unknown file loading error");
        }
        return data;
      });
    });
}

function _basename(path) {
  if (!path) return "";
  const parts = path.split("/");
  return parts[parts.length - 1] || path;
}

function _renderPlottedFilesList() {
  const list = document.getElementById("setup-plotted-files");
  if (!list) return;

  if (!plottedFiles.length) {
    list.innerHTML = '<div class="list-group-item text-muted small">No files plotted yet.</div>';
    return;
  }

  list.innerHTML = "";
  plottedFiles.forEach(function (entry, idx) {
    const row = document.createElement("div");
    row.className = "list-group-item d-flex justify-content-between align-items-start gap-2";

    const left = document.createElement("div");
    left.className = "small";

    const title = document.createElement("div");
    title.textContent = _basename(entry.path);
    if (entry.isFit) title.classList.add("fit-file");
    left.appendChild(title);

    const pathLabel = document.createElement("div");
    pathLabel.className = "text-muted";
    pathLabel.textContent = entry.path;
    left.appendChild(pathLabel);

    const right = document.createElement("div");
    right.className = "d-flex align-items-center gap-1";

    const toggleBtn = document.createElement("button");
    toggleBtn.type = "button";
    toggleBtn.className = "btn btn-sm btn-outline-secondary";
    toggleBtn.textContent = entry.visible ? "Hide" : "Show";
    toggleBtn.addEventListener("click", function () {
      plottedFiles[idx].visible = !plottedFiles[idx].visible;
      _renderPlottedFilesList();
      _renderSetupReflectivityPlot();
    });
    right.appendChild(toggleBtn);

    const removeBtn = document.createElement("button");
    removeBtn.type = "button";
    removeBtn.className = "btn btn-sm btn-outline-danger";
    removeBtn.innerHTML = '<i class="bi bi-trash"></i>';
    if (entry.isFit) {
      removeBtn.disabled = true;
      removeBtn.title = "Current fitting file";
    } else {
      removeBtn.title = "Remove";
      removeBtn.addEventListener("click", function () {
        plottedFiles.splice(idx, 1);
        _renderPlottedFilesList();
        _renderSetupReflectivityPlot();
      });
    }
    right.appendChild(removeBtn);

    row.appendChild(left);
    row.appendChild(right);
    list.appendChild(row);
  });
}

function _renderSetupReflectivityPlot() {
  const el = document.getElementById("setup-rq-chart");
  if (!el) return;

  const traces = [];
  plottedFiles.forEach(function (entry, i) {
    if (!entry.visible) return;
    const color = COLORS[i % COLORS.length];
    traces.push({
      x: entry.Q,
      y: entry.R,
      mode: "markers",
      type: "scatter",
      marker: {
        size: entry.isFit ? 5 : 4,
        color: color,
        symbol: entry.isFit ? "circle" : "diamond",
      },
      name: entry.isFit ? _basename(entry.path) + " (fit file)" : _basename(entry.path),
    });
  });

  if (!traces.length) {
    Plotly.react(el, [], {
      margin: { l: 50, r: 10, t: 10, b: 40 },
      xaxis: { title: "Q (Å⁻¹)" },
      yaxis: { title: "R(Q)" },
      annotations: [
        {
          x: 0.5,
          y: 0.5,
          xref: "paper",
          yref: "paper",
          showarrow: false,
          text: "Select a reflectivity file to preview R(Q)",
          font: { size: 13, color: "#6c757d" },
        },
      ],
    }, { responsive: true, displayModeBar: false });
    return;
  }

  Plotly.react(el, traces, {
    margin: { l: 55, r: 10, t: 10, b: 56 },
    xaxis: { title: "Q (Å⁻¹)", type: "log", exponentformat: "e" },
    yaxis: { title: "R(Q)", type: "log", exponentformat: "e" },
    legend: { orientation: "h", y: -0.3 },
    hovermode: "closest",
  }, { responsive: true, displayModeBar: false, scrollZoom: true });
}

/* ---- analysis launch ----------------------------------------- */

function startAnalysis() {
  const dataFile = document.getElementById("data-file").value.trim();
  const sampleDesc = document.getElementById("sample-desc").value.trim();
  const hypothesis = document.getElementById("hypothesis").value.trim();
  const outputDir = document.getElementById("output-dir").value.trim();

  if (!dataFile) { alert("Please select a data file."); return; }
  if (!sampleDesc) { alert("Please enter a sample description."); return; }
  if (!outputDir) { alert("Please select an output directory."); return; }

  _saveFormValues();

  const btn = document.getElementById("btn-start");
  btn.disabled = true;
  btn.innerHTML = '<span class="spinner-border spinner-border-sm"></span> Starting…';

  const maxIter = parseInt(document.getElementById("max-iterations").value, 10) || 5;

  fetch("/api/start-analysis", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      data_file: dataFile,
      sample_description: sampleDesc,
      hypothesis: hypothesis || null,
      output_dir: outputDir,
      interactive: document.getElementById("interactive-mode").checked,
      max_iterations: maxIter,
    }),
  })
    .then((r) => r.json().then((d) => ({ ok: r.ok, data: d })))
    .then(({ ok, data }) => {
      if (!ok) {
        const msg = data.errors ? data.errors.join("\n") : data.error || "Unknown error";
        alert("Could not start analysis:\n" + msg);
        btn.disabled = false;
        btn.innerHTML = '<i class="bi bi-play-fill"></i> Start Analysis';
        return;
      }
      // Switch to progress view
      document.getElementById("setup-section").style.display = "none";
      document.getElementById("progress-section").style.display = "";
      document.getElementById("checkpoint-table").querySelector("tbody").innerHTML = "";
      pollStatus();
    })
    .catch((err) => {
      alert("Network error: " + err);
      btn.disabled = false;
      btn.innerHTML = '<i class="bi bi-play-fill"></i> Start Analysis';
    });
}

/* ---- status polling ------------------------------------------ */

function pollStatus() {
  if (pollTimer) clearTimeout(pollTimer);

  fetch("/api/analysis-status")
    .then((r) => r.json())
    .then((st) => {
      const nodeLabel = st.current_node
        ? st.current_node.charAt(0).toUpperCase() + st.current_node.slice(1)
        : "Starting…";
      document.getElementById("progress-node").textContent =
        st.status === "complete"
          ? "Analysis complete!"
          : st.status === "error"
          ? "Error: " + (st.error || "unknown")
          : st.status === "waiting_for_user"
          ? "Waiting for your feedback…"
          : `Step: ${nodeLabel}  (iteration ${st.iteration})`;

      // Progress bar
      const nSteps = st.checkpoints ? st.checkpoints.length : 0;
      const pct = st.status === "complete" ? 100 : Math.min(95, (nSteps / 8) * 100);
      const bar = document.getElementById("progress-bar");
      bar.style.width = pct + "%";

      if (st.status === "complete") {
        bar.classList.remove("progress-bar-animated", "progress-bar-striped");
        bar.classList.add("bg-success");
      } else if (st.status === "error") {
        bar.classList.remove("progress-bar-animated", "progress-bar-striped");
        bar.classList.add("bg-danger");
      } else if (st.status === "waiting_for_user") {
        bar.classList.remove("progress-bar-animated");
      }

      // Status badge
      const badge = document.getElementById("progress-status");
      badge.textContent = st.status === "waiting_for_user" ? "waiting" : st.status;
      badge.className =
        "badge " +
        (st.status === "complete"
          ? "bg-success"
          : st.status === "error"
          ? "bg-danger"
          : st.status === "waiting_for_user"
          ? "bg-warning text-dark"
          : "bg-primary");

      // Checkpoint table
      if (st.checkpoints && st.checkpoints.length) {
        const tbody = document.getElementById("checkpoint-table").querySelector("tbody");
        tbody.innerHTML = "";
        st.checkpoints.forEach((cp, i) => {
          const tr = document.createElement("tr");
          tr.innerHTML =
            `<td>${i + 1}</td>` +
            `<td>${cp.node}</td>` +
            `<td>${cp.chi2 != null ? cp.chi2.toFixed(2) : "—"}</td>` +
            `<td>${_llmBadges(cp.llm_calls)}</td>`;
          tbody.appendChild(tr);
        });
        _renderChi2Chart(st.checkpoints);
      }

      // Chat panel (interactive mode)
      const chatPanel = document.getElementById("chat-panel");
      if (st.status === "waiting_for_user") {
        chatPanel.style.display = "";
        // Populate checkpoint dropdown with evaluation checkpoints
        _populateFeedbackCheckpoints(st.checkpoints || []);
        if (!_liveResultsFetched) {
          _liveResultsFetched = true;
          _fetchLiveResults();
        }
      } else {
        chatPanel.style.display = "none";
        _liveResultsFetched = false;
      }

      // Show live results panel when we have fit data (even while running)
      if (st.status === "complete" || st.status === "error") {
        _fetchLiveResults();  // final update
      }

      // Footer buttons
      if (st.status === "complete" || st.status === "error") {
        document.getElementById("progress-footer").style.display = "";
        chatPanel.style.display = "none";
        return; // stop polling
      }

      // Don't poll while waiting — user action will resume
      if (st.status !== "waiting_for_user") {
        pollTimer = setTimeout(pollStatus, 2000);
      }
    })
    .catch(() => {
      pollTimer = setTimeout(pollStatus, 3000);
    });
}

/* ---- reset --------------------------------------------------- */

function resetSetup() {
  document.getElementById("setup-section").style.display = "";
  document.getElementById("progress-section").style.display = "none";

  // Re-enable start button
  const btn = document.getElementById("btn-start");
  btn.disabled = false;
  btn.innerHTML = '<i class="bi bi-play-fill"></i> Start Analysis';

  // Clear progress
  document.getElementById("progress-bar").style.width = "5%";
  document.getElementById("progress-bar").className =
    "progress-bar progress-bar-striped progress-bar-animated";
  document.getElementById("progress-footer").style.display = "none";
  document.getElementById("checkpoint-table").querySelector("tbody").innerHTML = "";
  document.getElementById("chat-panel").style.display = "none";
  document.getElementById("chat-messages").innerHTML = "";
  document.getElementById("chat-input").value = "";
  // Clear live results
  document.getElementById("live-results").style.display = "none";
  Plotly.purge(document.getElementById("live-rq-chart"));
  Plotly.purge(document.getElementById("live-sld-chart"));
  Plotly.purge(document.getElementById("chi2-mini-chart"));
  document.getElementById("live-param-table").querySelector("tbody").innerHTML = "";
  document.getElementById("live-fit-summary").textContent = "";
  _liveResultsFetched = false;

  _renderPlottedFilesList();
  _renderSetupReflectivityPlot();
}

/* ---- chat / feedback helpers --------------------------------- */

function _renderChatMessages(messages) {
  const container = document.getElementById("chat-messages");
  container.innerHTML = "";
  messages.forEach(function (m) {
    const div = document.createElement("div");
    div.className = m.role === "user" ? "mb-2" : "mb-2 pb-2 border-bottom";
    const label = m.role === "user"
      ? '<strong class="text-primary">You:</strong> '
      : '<strong class="text-secondary">AuRE:</strong>';
    const body = m.role === "user"
      ? " " + _escapeHtml(m.content)
      : '<div class="mt-1">' + marked.parse(m.content) + "</div>";
    div.innerHTML = label + body;
    container.appendChild(div);
  });
  container.scrollTop = container.scrollHeight;
}

function _escapeHtml(text) {
  const d = document.createElement("div");
  d.textContent = text;
  return d.innerHTML;
}

function _postFeedback(action, feedback) {
  var payload = { action: action, feedback: feedback || null };

  // Include advanced options if set
  var dreamStepsEl = document.getElementById("fb-dream-steps");
  var checkpointEl = document.getElementById("fb-checkpoint");
  if (dreamStepsEl && dreamStepsEl.value) {
    payload.dream_steps = parseInt(dreamStepsEl.value, 10);
  }
  if (checkpointEl && checkpointEl.value) {
    payload.restart_checkpoint = checkpointEl.value;
  }

  fetch("/api/user-feedback", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  })
    .then(function (r) { return r.json(); })
    .then(function (data) {
      if (data.error) {
        alert(data.error);
        return;
      }
      // Re-enable progress bar animation and resume polling
      const bar = document.getElementById("progress-bar");
      bar.classList.add("progress-bar-animated");
      document.getElementById("chat-panel").style.display = "none";
      // Reset advanced options for next pause
      if (dreamStepsEl) dreamStepsEl.value = "";
      if (checkpointEl) checkpointEl.value = "";
      pollStatus();
    })
    .catch(function (err) { alert("Network error: " + err); });
}

function _populateFeedbackCheckpoints(checkpoints) {
  var sel = document.getElementById("fb-checkpoint");
  if (!sel) return;
  // Preserve current selection
  var prev = sel.value;
  sel.innerHTML = '<option value="">— continue normally —</option>';
  // Build options from evaluation checkpoints that contain an iteration
  var seen = {};
  checkpoints.forEach(function (cp, i) {
    if (cp.node !== "evaluation") return;
    var iter = cp.iteration || i;
    if (seen[iter]) return;
    seen[iter] = true;
    var chi2Label = cp.chi2 != null ? " (χ²=" + cp.chi2.toFixed(2) + ")" : "";
    var opt = document.createElement("option");
    opt.value = String(iter);
    opt.textContent = "Iteration " + iter + chi2Label;
    sel.appendChild(opt);
  });
  // Restore selection if still valid
  if (prev) sel.value = prev;
}

function sendFeedback() {
  const input = document.getElementById("chat-input");
  const text = input.value.trim();
  if (!text) { alert("Please type some feedback or click Continue."); return; }
  input.value = "";
  _postFeedback("continue", text);
}

function continueWithoutFeedback() {
  _postFeedback("continue", null);
}

function stopAnalysis() {
  _postFeedback("stop", null);
}

/* ---- χ² mini-chart ------------------------------------------- */

function _renderChi2Chart(checkpoints) {
  const el = document.getElementById("chi2-mini-chart");
  const chi2Values = [];
  const labels = [];
  checkpoints.forEach(function (cp, i) {
    if (cp.chi2 != null) {
      chi2Values.push(cp.chi2);
      labels.push(cp.node);
    }
  });
  if (chi2Values.length < 1) { el.innerHTML = ""; return; }

  const trace = {
    y: chi2Values,
    x: chi2Values.map(function (_, i) { return i + 1; }),
    text: labels,
    mode: "lines+markers",
    marker: { size: 7, color: "#0d6efd" },
    line: { width: 2, color: "#0d6efd" },
    hovertemplate: "%{text}<br>χ² = %{y:.2f}<extra></extra>",
  };
  const layout = {
    margin: { l: 45, r: 10, t: 5, b: 30 },
    xaxis: { title: { text: "Step", font: { size: 11 } }, dtick: 1 },
    yaxis: { title: { text: "χ²", font: { size: 11 } },
             type: Math.max.apply(null, chi2Values) > 100 ? "log" : "linear" },
    hovermode: "closest",
  };
  Plotly.react(el, [trace], layout, { responsive: true, displayModeBar: false });
}

/* ---- live results -------------------------------------------- */

function _fetchLiveResults() {
  fetch("/api/live/results")
    .then(function (r) { return r.json(); })
    .then(function (data) {
      if (!data.models || data.models.length === 0) return;
      document.getElementById("live-results").style.display = "";
      _renderLiveRQ(data);
      _renderLiveSLD(data);
      _renderLiveParams(data);
      _renderEvalSummary(data);
    })
    .catch(function (err) { console.error("Live results error:", err); });
}

function _renderLiveRQ(data) {
  const el = document.getElementById("live-rq-chart");
  const traces = [];

  if (data.Q && data.Q.length) {
    traces.push({
      x: data.Q, y: data.R,
      error_y: data.dR && data.dR.length
        ? { type: "data", array: data.dR, visible: true, thickness: 1 }
        : undefined,
      mode: "markers", marker: { size: 3, color: "#6c757d" },
      name: "Data", type: "scatter",
    });
  }
  (data.models || []).forEach(function (m, i) {
    var isFinal = i === data.models.length - 1;
    traces.push({
      x: m.Q, y: m.R, mode: "lines",
      line: { width: isFinal ? 3.5 : 1.5, color: COLORS[(i + 1) % COLORS.length] },
      name: m.label,
    });
  });
  var layout = {
    margin: { l: 50, r: 10, t: 5, b: 40 },
    xaxis: { title: "Q (Å⁻¹)", type: "log", exponentformat: "e" },
    yaxis: { title: "R(Q)", type: "log", exponentformat: "e" },
    legend: { x: 0, y: 0, bgcolor: "rgba(255,255,255,0.7)", font: { size: 10 } },
    hovermode: "closest",
  };
  Plotly.react(el, traces, layout, { responsive: true, scrollZoom: true });
}

function _renderLiveSLD(data) {
  var el = document.getElementById("live-sld-chart");
  var profiles = data.profiles || [];
  if (profiles.length === 0) {
    el.innerHTML = '<p class="text-muted text-center py-4" style="font-size:0.85rem">SLD profile not yet available.</p>';
    return;
  }
  var traces = profiles.map(function (p, i) {
    var isFinal = i === profiles.length - 1;
    return {
      x: p.z, y: p.sld, mode: "lines",
      line: { width: isFinal ? 3.5 : 1.5, color: COLORS[(i + 1) % COLORS.length] },
      name: p.label,
    };
  });
  var layout = {
    margin: { l: 50, r: 10, t: 5, b: 40 },
    xaxis: { title: "Depth z (Å)" },
    yaxis: { title: "SLD (×10⁻⁶ Å⁻²)" },
    legend: { x: 1, xanchor: "right", y: 1, bgcolor: "rgba(255,255,255,0.7)", font: { size: 10 } },
    hovermode: "closest",
  };
  Plotly.react(el, traces, layout, { responsive: true, scrollZoom: true });
}

function _renderLiveParams(data) {
  var tbody = document.getElementById("live-param-table").querySelector("tbody");
  tbody.innerHTML = "";
  var summary = document.getElementById("live-fit-summary");
  if (!data.parameters || data.parameters.length === 0) return;

  var parts = [];
  if (data.chi_squared != null) parts.push("χ² = " + data.chi_squared.toFixed(2));
  if (data.method) parts.push(data.method);
  if (data.converged != null) parts.push(data.converged ? "converged ✓" : "not converged ✗");
  summary.textContent = parts.join("  ·  ");

  data.parameters.forEach(function (p) {
    var val = typeof p.value === "number" ? p.value.toPrecision(5) : p.value;
    var unc = p.uncertainty != null ? "± " + p.uncertainty.toPrecision(3) : "—";
    tbody.insertAdjacentHTML("beforeend",
      '<tr><td><code>' + _escapeHtml(p.name) + '</code></td>' +
      '<td class="text-end">' + val + '</td>' +
      '<td class="text-end">' + unc + '</td></tr>');
  });
}

function _renderEvalSummary(data) {
  var container = document.getElementById("chat-messages");
  container.innerHTML = "";
  var issues = data.issues || [];
  var suggestions = data.suggestions || [];
  if (issues.length === 0 && suggestions.length === 0) return;

  var html = "";
  if (issues.length) {
    html += '<div class="mb-2"><strong>Issues Identified:</strong><ul class="mb-1">';
    issues.forEach(function (issue) {
      html += "<li>⚠️ " + _escapeHtml(issue) + "</li>";
    });
    html += "</ul></div>";
  }
  if (suggestions.length) {
    html += '<div class="mb-2"><strong>Suggested Improvements:</strong><ol class="mb-1">';
    suggestions.forEach(function (s) {
      html += "<li>" + _escapeHtml(s) + "</li>";
    });
    html += "</ol></div>";
  }
  container.innerHTML = html;
}
