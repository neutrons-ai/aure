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

let plottedFiles = []; // [{ path, Q, R, dR, visible, isFit, state }]
let groupIntoStates = false;

function _collectStateNames() {
  // Distinct, non-empty state names currently assigned to plotted files.
  var seen = Object.create(null);
  var out = [];
  plottedFiles.forEach(function (f) {
    var s = (f.state || "").trim();
    if (s && !seen[s]) { seen[s] = true; out.push(s); }
  });
  return out;
}

function _colorForState(name) {
  // Deterministic palette index based on state-name order.
  var names = _collectStateNames();
  var i = names.indexOf(name);
  if (i < 0) return COLORS[0];
  return COLORS[i % COLORS.length];
}

function _refreshStateDatalist() {
  var dl = document.getElementById("state-name-suggestions");
  if (!dl) return;
  var names = _collectStateNames();
  dl.innerHTML = "";
  names.forEach(function (n) {
    var opt = document.createElement("option");
    opt.value = n;
    dl.appendChild(opt);
  });
}

function _extractRunNumber(path) {
  // Extract a run number from a filename: look for 6-digit number first,
  // then any trailing digits before the extension.
  var name = _basename(path).replace(/\.[^.]+$/, "");
  var m = name.match(/(\d{6})/);
  if (m) return parseInt(m[1], 10);
  m = name.match(/(\d+)/);
  if (m) return parseInt(m[1], 10);
  return Infinity;  // no number found → sort last
}

function _sortPlottedFilesByRunNumber() {
  plottedFiles.sort(function (a, b) {
    return _extractRunNumber(a.path) - _extractRunNumber(b.path);
  });
}

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

  // Derive from first plotted file
  var first = plottedFiles.length ? plottedFiles[0].path : "";
  return _parentDir(first);
}

function _syncDataFileInput() {
  // Keep hidden data-file input in sync with first isFit file
  var firstFit = plottedFiles.find(function (f) { return f.isFit; });
  document.getElementById("data-file").value = firstFit ? firstFit.path : "";
}

/* ---- persist / restore form values --------------------------- */

let _restoringFiles = false;  // guard against DOMContentLoaded race

function _restorePlottedFiles(savedFiles) {
  var remaining = savedFiles.slice();
  plottedFiles = [];
  _restoringFiles = true;

  function _loadNext() {
    if (!remaining.length) {
      _sortPlottedFilesByRunNumber();
      _syncDataFileInput();
      _renderPlottedFilesList();
      _renderSetupReflectivityPlot();
      _restoringFiles = false;
      return;
    }
    var entry = remaining.shift();
    _loadReflectivityFile(entry.path)
      .then(function (payload) {
        plottedFiles.push({
          path: entry.path,
          Q: payload.Q || [],
          R: payload.R || [],
          dR: payload.dR || [],
          visible: true,
          isFit: entry.isFit,
          state: entry.state || null,
        });
      })
      .catch(function () {
        // skip files that can no longer be loaded
      })
      .then(_loadNext);
  }

  _loadNext();
}

function _saveFormValues() {
  const vals = {
    sample_desc: document.getElementById("sample-desc").value,
    hypothesis: document.getElementById("hypothesis").value,
    output_dir: document.getElementById("output-dir").value,
    interactive: document.getElementById("interactive-mode").checked,
    max_iterations: parseInt(document.getElementById("max-iterations").value, 10) || 5,
    plotted_files: plottedFiles.map(function (f) {
      return { path: f.path, isFit: f.isFit, state: f.state || null };
    }),
    group_into_states: groupIntoStates,
    ties_mode: tiesMode,
    ties_text: tiesText,
    state_overrides: stateOverrides,
  };
  try { localStorage.setItem(STORAGE_KEY, JSON.stringify(vals)); } catch (_) {}
}

function _restoreFormValues() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    let vals = raw ? JSON.parse(raw) : null;

    // Fall back to server-provided previous run data
    if (!vals && typeof _prevRun !== "undefined" && _prevRun) {
      // Convert server data_files [{file, label}] to plotted_files [{path, isFit}]
      var pf = [];
      var prevStates = _prevRun.states || [];
      var pathToState = {};
      if (prevStates.length) {
        prevStates.forEach(function (st) {
          (st.data_files || []).forEach(function (df) {
            if (df && df.file) pathToState[df.file] = st.name;
          });
        });
      }
      if (_prevRun.data_files && _prevRun.data_files.length) {
        pf = _prevRun.data_files.map(function (df) {
          return {
            path: df.file,
            isFit: true,
            state: pathToState[df.file] || null,
          };
        });
      } else if (_prevRun.data_file) {
        pf = [{
          path: _prevRun.data_file,
          isFit: true,
          state: pathToState[_prevRun.data_file] || null,
        }];
      }
      var prefTiesMode = "auto";
      var prefTiesText = "";
      if (_prevRun.shared_parameters && _prevRun.shared_parameters.length) {
        prefTiesMode = "shared";
        prefTiesText = _prevRun.shared_parameters.join("\n");
      } else if (_prevRun.unshared_parameters && _prevRun.unshared_parameters.length) {
        prefTiesMode = "unshared";
        prefTiesText = _prevRun.unshared_parameters.join("\n");
      }
      var prefOverrides = {};
      prevStates.forEach(function (st) {
        var ov = {};
        if (st.ambient && typeof st.ambient.rho === "number") ov.ambient = st.ambient.rho;
        ["intensity", "theta_offset", "sample_broadening"].forEach(function (k) {
          if (st[k] && typeof st[k] === "object") {
            ov[k] = {};
            ["init", "min", "max"].forEach(function (sub) {
              if (typeof st[k][sub] === "number") ov[k][sub] = st[k][sub];
            });
            if (!Object.keys(ov[k]).length) delete ov[k];
          }
        });
        if (st.back_reflection === true) ov.back_reflection = true;
        if (st.extra_description) ov.extra_description = st.extra_description;
        if (Object.keys(ov).length) prefOverrides[st.name] = ov;
      });
      vals = {
        sample_desc: _prevRun.sample_description || "",
        hypothesis: _prevRun.hypothesis || "",
        output_dir: _prevRun.output_dir || "",
        plotted_files: pf,
        group_into_states: prevStates.length >= 2,
        ties_mode: prefTiesMode,
        ties_text: prefTiesText,
        state_overrides: prefOverrides,
      };
    }

    if (!vals) return;

    if (vals.sample_desc) document.getElementById("sample-desc").value = vals.sample_desc;
    if (vals.hypothesis)  document.getElementById("hypothesis").value = vals.hypothesis;
    if (vals.output_dir)  document.getElementById("output-dir").value = vals.output_dir;
    if (vals.interactive)  document.getElementById("interactive-mode").checked = vals.interactive;
    if (vals.max_iterations) document.getElementById("max-iterations").value = vals.max_iterations;
    if (vals.group_into_states) {
      groupIntoStates = true;
      var gtoggle = document.getElementById("group-into-states");
      if (gtoggle) gtoggle.checked = true;
    }
    if (vals.ties_mode && ["auto", "shared", "unshared"].indexOf(vals.ties_mode) >= 0) {
      tiesMode = vals.ties_mode;
    }
    if (typeof vals.ties_text === "string") {
      tiesText = vals.ties_text;
    }
    if (vals.state_overrides && typeof vals.state_overrides === "object") {
      stateOverrides = vals.state_overrides;
    }
    // Restore file list (single source of truth for data files)
    if (vals.plotted_files && vals.plotted_files.length) {
      _restorePlottedFiles(vals.plotted_files);
    }
  } catch (_) {}
}

document.addEventListener("DOMContentLoaded", function () {
  _restoreFormValues();
  var gtoggle = document.getElementById("group-into-states");
  if (gtoggle) {
    gtoggle.addEventListener("change", function () {
      groupIntoStates = gtoggle.checked;
      _renderPlottedFilesList();
      _renderSetupReflectivityPlot();
      _saveFormValues();
    });
  }
  _wireTiesPanel();
  // Only render empty state if no restore is in progress
  if (!_restoringFiles && !plottedFiles.length) {
    _renderPlottedFilesList();
    _renderSetupReflectivityPlot();
  }
});

// Persist form state (including multi-file list) on page leave
window.addEventListener("beforeunload", function () {
  _saveFormValues();
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
    document.getElementById("browser-title").textContent = "Load Data";
  } else {
    document.getElementById("browser-title").textContent = "Select Output Folder";
  }

  // Show or hide "Select this folder" button
  document.getElementById("btn-select").style.display =
    mode === "dir" ? "inline-block" : "none";

  // Start at the last data folder for file selection, otherwise server default
  const startPath = mode === "file" ? _getLastDataDir() : "";
  _fetchBrowserListing(startPath);

  browserModalInstance =
    browserModalInstance ||
    new bootstrap.Modal(document.getElementById("browserModal"));
  browserModalInstance.show();
}

function _fetchBrowserListing(path) {
  const endpoint = browserMode === "file"
    ? "/api/browse-files"
    : "/api/browse-dirs";
  const params = new URLSearchParams();
  if (path) params.set("path", path);
  if (browserMode === "file") {
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
            // File selected – add to plotted files (fit by default)
            _addDataFile(entry.path);
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

function _addDataFile(path) {
  const existing = plottedFiles.find(function (f) { return f.path === path; });
  if (existing) {
    existing.visible = true;
    _syncDataFileInput();
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
        state: null,
      });
      _sortPlottedFilesByRunNumber();
      _syncDataFileInput();
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

  _refreshStateDatalist();
  _renderTiesPanel();
  _renderOverridesPanel();

  if (!plottedFiles.length) {
    list.innerHTML = '<div class="list-group-item text-muted small">No files loaded yet.</div>';
    return;
  }

  list.innerHTML = "";
  plottedFiles.forEach(function (entry, idx) {
    const row = document.createElement("div");
    row.className = "list-group-item d-flex justify-content-between align-items-start gap-2";

    const left = document.createElement("div");
    left.className = "d-flex align-items-start gap-2 small";

    // "Include in fit" checkbox
    const fitCheck = document.createElement("input");
    fitCheck.type = "checkbox";
    fitCheck.className = "form-check-input mt-1";
    fitCheck.checked = entry.isFit;
    fitCheck.title = entry.isFit ? "Included in fit" : "Include in fit";
    fitCheck.addEventListener("change", function () {
      plottedFiles[idx].isFit = fitCheck.checked;
      // Keep at least one fit file – re-check the first if none remain
      const anyFit = plottedFiles.some(function (f) { return f.isFit; });
      if (!anyFit) {
        plottedFiles[0].isFit = true;
      }
      _syncDataFileInput();
      _renderPlottedFilesList();
      _renderSetupReflectivityPlot();
    });
    left.appendChild(fitCheck);

    const labels = document.createElement("div");
    const title = document.createElement("div");
    title.textContent = _basename(entry.path);
    if (entry.isFit) title.classList.add("fit-file");
    labels.appendChild(title);

    const pathLabel = document.createElement("div");
    pathLabel.className = "text-muted";
    pathLabel.textContent = entry.path;
    labels.appendChild(pathLabel);

    // State name input (only shown when grouping is on)
    if (groupIntoStates) {
      const stateRow = document.createElement("div");
      stateRow.className = "d-flex align-items-center gap-2 mt-1";
      const stateLabel = document.createElement("label");
      stateLabel.className = "text-muted small mb-0";
      stateLabel.textContent = "State:";
      const stateInput = document.createElement("input");
      stateInput.type = "text";
      stateInput.className = "form-control form-control-sm";
      stateInput.style.maxWidth = "160px";
      stateInput.setAttribute("list", "state-name-suggestions");
      stateInput.placeholder = "e.g. D2O";
      stateInput.value = entry.state || "";
      stateInput.addEventListener("input", function () {
        plottedFiles[idx].state = stateInput.value.trim() || null;
        _refreshStateDatalist();
      });
      stateInput.addEventListener("change", function () {
        // On blur / commit: re-render so badges, ties panel, and trace
        // colours update.
        _renderPlottedFilesList();
        _renderSetupReflectivityPlot();
        _saveFormValues();
      });
      stateRow.appendChild(stateLabel);
      stateRow.appendChild(stateInput);

      if (entry.isFit && !(entry.state && entry.state.trim())) {
        const badge = document.createElement("span");
        badge.className = "badge bg-warning text-dark";
        badge.textContent = "ungrouped";
        badge.title =
          "Assign this file a state name, or untick its fit checkbox.";
        stateRow.appendChild(badge);
      } else if (entry.state) {
        const swatch = document.createElement("span");
        swatch.className = "badge";
        swatch.textContent = entry.state;
        swatch.style.backgroundColor = _colorForState(entry.state);
        swatch.style.color = "#fff";
        stateRow.appendChild(swatch);
      }
      labels.appendChild(stateRow);
    }

    left.appendChild(labels);

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
    // Only allow removal if more than one fit file or this one isn't a fit file
    const fitCount = plottedFiles.filter(function (f) { return f.isFit; }).length;
    if (entry.isFit && fitCount <= 1) {
      removeBtn.disabled = true;
      removeBtn.title = "Cannot remove the only fitting file";
    } else {
      removeBtn.title = "Remove";
      removeBtn.addEventListener("click", function () {
        plottedFiles.splice(idx, 1);
        _syncDataFileInput();
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
  const useStateColours = groupIntoStates && _collectStateNames().length >= 2;
  plottedFiles.forEach(function (entry, i) {
    if (!entry.visible) return;
    const color = (useStateColours && entry.state)
      ? _colorForState(entry.state)
      : COLORS[i % COLORS.length];
    const stateSuffix = (useStateColours && entry.state) ? " [" + entry.state + "]" : "";
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
      name: (entry.isFit
        ? _basename(entry.path) + " (fit)"
        : _basename(entry.path)) + stateSuffix,
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

/* ---- cross-state ties panel ---------------------------------- */

let tiesMode = "auto";   // "auto" | "shared" | "unshared"
let tiesText = "";

const TIES_HELP_URL =
  "https://github.com/neutrons-ai/aure/blob/main/src/aure/skills/multi-state-corefinement/SKILL.md";

const TIES_PRESETS = {
  structural: [
    "Cu.thickness",
    "Cu.material.rho",
    "Cu.interface",
    "substrate.interface",
  ],
  substrate: ["substrate.interface"],
  "all-but-ambient": [
    "Cu.thickness",
    "Cu.material.rho",
    "Cu.interface",
    "substrate.interface",
    "intensity",
  ],
};

function _parseTiesText(text) {
  if (!text) return [];
  return text
    .split(/[\n,]+/)
    .map(function (s) { return s.trim(); })
    .filter(function (s) { return s.length > 0; });
}

function _renderTiesPanel() {
  const card = document.getElementById("ties-card");
  if (!card) return;

  const stateNames = _collectStateNames();
  const fitFiles = plottedFiles.filter(function (f) { return f.isFit; });
  // Only show when grouping is on, ≥2 distinct state names, and every
  // fit file is assigned a state (mirrors server validation).
  const allGrouped = fitFiles.length > 0 && fitFiles.every(function (f) {
    return f.state && f.state.trim();
  });
  const visible = groupIntoStates && stateNames.length >= 2 && allGrouped;
  card.style.display = visible ? "" : "none";
  if (!visible) return;

  const radios = card.querySelectorAll("input.ties-mode");
  radios.forEach(function (r) { r.checked = r.value === tiesMode; });

  const presets = document.getElementById("ties-presets");
  if (presets) presets.style.display = (tiesMode === "shared") ? "" : "none";

  const ta = document.getElementById("ties-text");
  if (ta) {
    ta.value = tiesText || "";
    ta.disabled = (tiesMode === "auto");
    ta.setAttribute("list", "ties-param-suggestions");
  }

  const help = document.getElementById("ties-help");
  if (help) help.href = TIES_HELP_URL;
}

function _refreshTiesParamDatalist(extraParams) {
  const dl = document.getElementById("ties-param-suggestions");
  if (!dl) return;
  const merged = new Set();
  // Skill-bundled presets contribute their canonical names.
  Object.values(TIES_PRESETS).forEach(function (arr) {
    arr.forEach(function (p) { merged.add(p); });
  });
  (_knownSharedParamsCache || []).forEach(function (p) { merged.add(p); });
  if (Array.isArray(extraParams)) {
    extraParams.forEach(function (p) { if (p) merged.add(p); });
  }
  dl.innerHTML = "";
  Array.from(merged).sort().forEach(function (p) {
    const opt = document.createElement("option");
    opt.value = p;
    dl.appendChild(opt);
  });
}

let _knownSharedParamsCache = [];
let _knownSharedParamsFetched = false;
function _fetchKnownSharedParamsOnce() {
  if (_knownSharedParamsFetched) return;
  _knownSharedParamsFetched = true;
  fetch("/api/known-shared-params")
    .then(function (r) { return r.ok ? r.json() : { parameters: [] }; })
    .then(function (j) {
      _knownSharedParamsCache = (j && j.parameters) || [];
      _refreshTiesParamDatalist();
    })
    .catch(function () { /* silent */ });
}

function _wireTiesPanel() {
  const card = document.getElementById("ties-card");
  if (!card) return;

  card.querySelectorAll("input.ties-mode").forEach(function (r) {
    r.addEventListener("change", function () {
      if (r.checked) {
        tiesMode = r.value;
        _renderTiesPanel();
        _saveFormValues();
      }
    });
  });

  const ta = document.getElementById("ties-text");
  if (ta) {
    ta.addEventListener("input", function () {
      tiesText = ta.value;
      _saveFormValues();
    });
  }

  card.querySelectorAll("[data-ties-preset]").forEach(function (btn) {
    btn.addEventListener("click", function () {
      const key = btn.getAttribute("data-ties-preset");
      const items = TIES_PRESETS[key] || [];
      tiesText = items.join("\n");
      _renderTiesPanel();
      _saveFormValues();
    });
  });

  _refreshTiesParamDatalist();
  _fetchKnownSharedParamsOnce();

  const previewBtn = document.getElementById("ties-preview-btn");
  if (previewBtn) {
    previewBtn.addEventListener("click", _previewStructure);
  }
}

function _setPreviewStatus(msg, isError) {
  const el = document.getElementById("ties-preview-status");
  if (!el) return;
  el.textContent = msg || "";
  el.classList.toggle("text-danger", !!isError);
  el.classList.toggle("text-muted", !isError);
}

function _previewStructure() {
  const btn = document.getElementById("ties-preview-btn");
  if (btn) btn.disabled = true;
  _setPreviewStatus("Running intake → analysis → modeling…", false);
  const built = _buildAnalysisBody({ skipOutputDir: true });
  if (built.errors && built.errors.length) {
    _setPreviewStatus(built.errors.join("; "), true);
    if (btn) btn.disabled = false;
    return;
  }
  fetch("/api/preview-structure", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(built.body),
  })
    .then(function (r) { return r.json().then(function (j) { return { ok: r.ok, body: j }; }); })
    .then(function (res) {
      if (!res.ok) {
        const errs = (res.body && res.body.errors) || ["Preview failed."];
        _setPreviewStatus(errs.join("; "), true);
        return;
      }
      const params = (res.body.parameters || []).slice();
      _refreshTiesParamDatalist(params);
      _renderPreviewChecklist(res.body.layers || [], params);
      _setPreviewStatus(
        (res.body.layers || []).length + " layers · " + params.length + " parameters",
        false
      );
    })
    .catch(function (e) { _setPreviewStatus(String(e), true); })
    .finally(function () { if (btn) btn.disabled = false; });
}

function _renderPreviewChecklist(layers, params) {
  const host = document.getElementById("ties-checklist-host");
  if (!host) return;
  host.innerHTML = "";
  if (!params.length) return;

  // Split into structural (layer.* / substrate.interface) and ambient/intensity columns.
  const structural = params.filter(function (p) {
    return !/^(intensity|theta_offset|sample_broadening|ambient)/i.test(p);
  });
  const ambient = params.filter(function (p) {
    return /^(intensity|theta_offset|sample_broadening|ambient)/i.test(p);
  });
  const selected = new Set(
    tiesText.split(/\r?\n/).map(function (s) { return s.trim(); }).filter(Boolean)
  );

  function column(title, items) {
    const col = document.createElement("div");
    col.className = "col-md-6";
    const h = document.createElement("div");
    h.className = "small fw-semibold mb-1";
    h.textContent = title;
    col.appendChild(h);
    items.forEach(function (p) {
      const div = document.createElement("div");
      div.className = "form-check";
      const cb = document.createElement("input");
      cb.type = "checkbox";
      cb.className = "form-check-input";
      cb.id = "ties-cb-" + p.replace(/[^a-z0-9]/gi, "_");
      cb.checked = selected.has(p);
      cb.addEventListener("change", function () {
        const cur = new Set(
          tiesText.split(/\r?\n/).map(function (s) { return s.trim(); }).filter(Boolean)
        );
        if (cb.checked) cur.add(p); else cur.delete(p);
        tiesText = Array.from(cur).join("\n");
        const ta = document.getElementById("ties-text");
        if (ta) ta.value = tiesText;
        _saveFormValues();
      });
      const lbl = document.createElement("label");
      lbl.className = "form-check-label small font-monospace";
      lbl.htmlFor = cb.id;
      lbl.textContent = p;
      div.appendChild(cb);
      div.appendChild(lbl);
      col.appendChild(div);
    });
    return col;
  }

  const row = document.createElement("div");
  row.className = "row g-2 mb-2";
  row.appendChild(column("Structural", structural));
  if (ambient.length) row.appendChild(column("Ambient / intensity", ambient));
  host.appendChild(row);
}

/* ---- per-state overrides accordion --------------------------- */

let stateOverrides = {};  // { [stateName]: { ambient, intensity:{init,min,max},
                          //                  theta_offset:{...},
                          //                  sample_broadening:{...},
                          //                  back_reflection: bool,
                          //                  extra_description: str } }

function _ensureOverride(name) {
  if (!stateOverrides[name]) stateOverrides[name] = {};
  return stateOverrides[name];
}

const SINGLE_STATE_DEFAULT = "state0";

function _singleStateName() {
  // Canonical name for the implicit single state: the one distinct state name
  // carried by the files (e.g. from a loaded setup), else "state0".
  const names = _collectStateNames();
  return names.length === 1 ? names[0] : SINGLE_STATE_DEFAULT;
}

// Mirror the server's partials detection (config.py _PARTIAL_RE / _detect_kind):
// theta_offset and sample_broadening are only valid for partial-data states.
const _PARTIAL_FILE_RE = /_\d+_\d+_partial\.txt$/i;

function _filesArePartials(files) {
  return (files || []).some(function (f) {
    return _PARTIAL_FILE_RE.test(_basename(f.path || f.file || ""));
  });
}

function _fitFilesForState(name) {
  return plottedFiles.filter(function (f) {
    return f.isFit && (f.state || "").trim() === name;
  });
}

function _triplet(name, key, ov) {
  // Render an {init, min, max} triplet of numeric inputs.
  const wrap = document.createElement("div");
  wrap.className = "row g-2 mb-2";
  ["init", "min", "max"].forEach(function (sub) {
    const col = document.createElement("div");
    col.className = "col-auto";
    const lbl = document.createElement("label");
    lbl.className = "form-label small mb-0";
    lbl.textContent = sub;
    const inp = document.createElement("input");
    inp.type = "number";
    inp.step = "any";
    inp.className = "form-control form-control-sm";
    inp.style.maxWidth = "120px";
    inp.value = (ov[key] && ov[key][sub] != null) ? ov[key][sub] : "";
    inp.addEventListener("change", function () {
      const o = _ensureOverride(name);
      const cur = o[key] || {};
      const v = inp.value.trim();
      if (v === "") { delete cur[sub]; }
      else { cur[sub] = parseFloat(v); }
      // Drop empty triplet sub-objects.
      if (Object.keys(cur).length) { o[key] = cur; }
      else { delete o[key]; }
      _validateTriplet(o[key]);
      _saveFormValues();
    });
    col.appendChild(lbl);
    col.appendChild(inp);
    wrap.appendChild(col);
  });
  return wrap;
}

function _validateTriplet(t) {
  if (!t) return true;
  if (t.init != null && t.min != null && t.init < t.min) return false;
  if (t.init != null && t.max != null && t.init > t.max) return false;
  if (t.min != null && t.max != null && t.min > t.max) return false;
  return true;
}

function _renderOverrideFields(body, name, ov, isPartials) {
  // Render the editable override widgets (ambient SLD; intensity / theta_offset
  // / sample_broadening triplets; back-reflection; extra description) for one
  // state into `body`. Shared by the multi-state accordion and the single-state
  // panel. `ov` must be the stored stateOverrides[name] object so the change
  // handlers mutate the persisted state. `isPartials` gates the partials-only
  // θ-offset / sample-broadening fields.
  const safeId = name.replace(/[^\w-]/g, "_");

  // ambient (single number)
  const ambWrap = document.createElement("div");
  ambWrap.className = "mb-2";
  ambWrap.innerHTML = '<label class="form-label small mb-0">Ambient SLD ' +
    '<span class="text-muted">(10⁻⁶ Å⁻²)</span></label>';
  const ambIn = document.createElement("input");
  ambIn.type = "number";
  ambIn.step = "any";
  ambIn.className = "form-control form-control-sm";
  ambIn.style.maxWidth = "200px";
  ambIn.value = (ov.ambient != null) ? ov.ambient : "";
  ambIn.addEventListener("change", function () {
    const v = ambIn.value.trim();
    if (v === "") delete ov.ambient;
    else ov.ambient = parseFloat(v);
    _saveFormValues();
  });
  ambWrap.appendChild(ambIn);
  body.appendChild(ambWrap);

  // intensity (any state) + theta_offset / sample_broadening (partials only)
  const tripletDefs = [["intensity", "Intensity"]];
  if (isPartials) {
    tripletDefs.push(
      ["theta_offset", "θ offset"],
      ["sample_broadening", "Sample broadening"],
    );
  }
  tripletDefs.forEach(function (pair) {
    const wrap = document.createElement("div");
    wrap.className = "mb-1";
    const lbl = document.createElement("div");
    lbl.className = "form-label small mb-0";
    lbl.textContent = pair[1];
    wrap.appendChild(lbl);
    wrap.appendChild(_triplet(name, pair[0], ov));
    body.appendChild(wrap);
  });
  if (!isPartials) {
    const hint = document.createElement("div");
    hint.className = "form-text mb-2";
    hint.textContent =
      "θ-offset and sample broadening apply to partial-data states only.";
    body.appendChild(hint);
  }

  // back_reflection checkbox
  const brWrap = document.createElement("div");
  brWrap.className = "form-check form-switch mb-2";
  brWrap.innerHTML =
    '<input class="form-check-input" type="checkbox" id="ov-br-' + safeId + '">' +
    '<label class="form-check-label" for="ov-br-' + safeId + '">' +
    'Back reflection</label>';
  const brIn = brWrap.querySelector("input");
  brIn.checked = !!ov.back_reflection;
  brIn.addEventListener("change", function () {
    if (brIn.checked) ov.back_reflection = true;
    else delete ov.back_reflection;
    _saveFormValues();
  });
  body.appendChild(brWrap);

  // extra_description textarea
  const descWrap = document.createElement("div");
  descWrap.className = "mb-1";
  descWrap.innerHTML = '<label class="form-label small mb-0">Extra description</label>';
  const descIn = document.createElement("textarea");
  descIn.className = "form-control form-control-sm";
  descIn.rows = 2;
  descIn.value = ov.extra_description || "";
  descIn.addEventListener("change", function () {
    const v = descIn.value.trim();
    if (v) ov.extra_description = v;
    else delete ov.extra_description;
    _saveFormValues();
  });
  descWrap.appendChild(descIn);
  body.appendChild(descWrap);
}

function _renderOverridesPanel() {
  const card = document.getElementById("overrides-card");
  const acc = document.getElementById("overrides-accordion");
  const single = document.getElementById("overrides-single");
  if (!card || !acc) return;

  const stateNames = _collectStateNames();
  const multi = groupIntoStates && stateNames.length >= 2;
  const hasFitFiles = plottedFiles.some(function (f) { return f.isFit; });

  // The card shows in multi-state mode, or whenever there are fit files to
  // attach single-state overrides to.
  card.style.display = (multi || hasFitFiles) ? "" : "none";
  if (!multi && !hasFitFiles) {
    acc.innerHTML = "";
    if (single) single.innerHTML = "";
    return;
  }

  if (multi) {
    // Multi-state: one collapsible accordion item per state.
    if (single) single.innerHTML = "";
    acc.style.display = "";

    // Drop stale overrides for renamed/removed states.
    Object.keys(stateOverrides).forEach(function (n) {
      if (stateNames.indexOf(n) < 0) delete stateOverrides[n];
    });

    acc.innerHTML = "";
    stateNames.forEach(function (name, idx) {
      const ov = _ensureOverride(name);
      const item = document.createElement("div");
      item.className = "accordion-item";
      const hid = "ov-h-" + idx;
      const cid = "ov-c-" + idx;
      item.innerHTML =
        '<h2 class="accordion-header" id="' + hid + '">' +
          '<button class="accordion-button collapsed" type="button" ' +
          'data-bs-toggle="collapse" data-bs-target="#' + cid + '">' +
          '<span class="badge me-2" style="background-color:' +
            _colorForState(name) + ';color:#fff">' + name + '</span>' +
          '<span class="small text-muted">overrides</span>' +
          '</button>' +
        '</h2>' +
        '<div id="' + cid + '" class="accordion-collapse collapse" ' +
          'data-bs-parent="#overrides-accordion"><div class="accordion-body small"></div></div>';
      const body = item.querySelector(".accordion-body");
      _renderOverrideFields(body, name, ov, _filesArePartials(_fitFilesForState(name)));
      acc.appendChild(item);
    });
    return;
  }

  // Single-state: one flat panel keyed by the implicit single state. Lets a
  // single (ungrouped) state carry ambient / intensity / theta_offset /
  // sample_broadening / back_reflection / extra_description without faking a
  // second state — and round-trips a single-state setup loaded from YAML.
  acc.innerHTML = "";
  if (!single) return;
  const name = _singleStateName();
  const ov = _ensureOverride(name);
  single.innerHTML = "";
  const hdr = document.createElement("div");
  hdr.className = "small text-muted mb-2";
  const badge = document.createElement("span");
  badge.className = "badge";
  badge.style.backgroundColor = _colorForState(name);
  badge.style.color = "#fff";
  badge.textContent = name;
  hdr.appendChild(document.createTextNode("State "));
  hdr.appendChild(badge);
  single.appendChild(hdr);
  const fitFiles = plottedFiles.filter(function (f) { return f.isFit; });
  _renderOverrideFields(single, name, ov, _filesArePartials(fitFiles));
}

/* ---- analysis launch ----------------------------------------- */

function _buildStateEntry(name, files, errors) {
  // Build one `states[]` entry from a state name + its files, applying any
  // per-state overrides tracked in `stateOverrides`. Shared by the multi-state
  // path and the single-state path so a loaded setup (ambient / intensity /
  // theta_offset / sample_broadening / back_reflection / extra_description)
  // round-trips faithfully on Save Setup and carries into Start Analysis.
  const dataFiles = files.map(function (f) {
    const stem = _basename(f.path).replace(/\.[^.]+$/, "");
    return { file: f.path, label: stem };
  });
  const entry = { name: name, data_files: dataFiles };
  const ov = stateOverrides[name] || {};
  if (ov.ambient != null && !Number.isNaN(ov.ambient)) {
    entry.ambient = { rho: ov.ambient };
  }
  // theta_offset / sample_broadening are partials-only (the server rejects
  // them on combined states); intensity applies to any state.
  const tripletKeys = _filesArePartials(files)
    ? ["intensity", "theta_offset", "sample_broadening"]
    : ["intensity"];
  tripletKeys.forEach(function (k) {
    if (ov[k] && Object.keys(ov[k]).length) {
      if (!_validateTriplet(ov[k])) {
        errors.push(
          "State '" + name + "' " + k +
          ": init/min/max must satisfy min ≤ init ≤ max."
        );
      }
      entry[k] = Object.assign({}, ov[k]);
    }
  });
  if (ov.back_reflection === true) entry.back_reflection = true;
  if (ov.extra_description) entry.extra_description = ov.extra_description;
  return entry;
}

function _buildAnalysisBody(opts) {
  // Compose the JSON body for /api/start-analysis (or /api/preview-structure).
  // Returns { body, errors }; if errors is non-empty the caller should abort.
  const sampleDesc = (document.getElementById("sample-desc").value || "").trim();
  const hypothesis = (document.getElementById("hypothesis").value || "").trim();
  const outputDir = (document.getElementById("output-dir").value || "").trim();
  const fitFiles = plottedFiles.filter(function (f) { return f.isFit; });
  const errors = [];

  if (!fitFiles.length) errors.push("Please load at least one data file.");
  if (!sampleDesc) errors.push("Please enter a sample description.");
  if (!opts || !opts.skipOutputDir) {
    if (!outputDir) errors.push("Please select an output directory.");
  }

  const maxIter = parseInt(document.getElementById("max-iterations").value, 10) || 5;
  const body = {
    data_file: fitFiles.length ? fitFiles[0].path : "",
    sample_description: sampleDesc,
    hypothesis: hypothesis || null,
    output_dir: outputDir,
    interactive: document.getElementById("interactive-mode").checked,
    max_iterations: maxIter,
  };

  const stateNames = _collectStateNames();
  const fitStates = fitFiles.map(function (f) {
    return (f.state || "").trim();
  });
  const hasGrouped = fitStates.some(function (s) { return s.length > 0; });
  const allGrouped = fitStates.every(function (s) { return s.length > 0; });

  if (groupIntoStates && hasGrouped) {
    if (!allGrouped) {
      errors.push(
        "Some fit files have no state assigned. Either tag every fit file " +
        "with a state name or untick its fit checkbox."
      );
    }
    if (errors.length) return { body: body, errors: errors };

    // Group fit files by state name (order = first-seen order).
    const order = [];
    const byState = Object.create(null);
    fitFiles.forEach(function (f) {
      const name = f.state.trim();
      if (!byState[name]) { byState[name] = []; order.push(name); }
      byState[name].push(f);
    });

    if (order.length < 2) {
      errors.push(
        "Multi-state co-refinement requires at least 2 distinct state names."
      );
      return { body: body, errors: errors };
    }

    body.states = order.map(function (name) {
      return _buildStateEntry(name, byState[name], errors);
    });

    const tieParams = _parseTiesText(tiesText);
    if (tiesMode === "shared" && tieParams.length) {
      body.shared_parameters = tieParams;
    } else if (tiesMode === "unshared" && tieParams.length) {
      body.unshared_parameters = tieParams;
    }
    if (body.shared_parameters && body.unshared_parameters) {
      // Radio prevents this, but be defensive.
      errors.push(
        "shared_parameters and unshared_parameters are mutually exclusive."
      );
    }
    // Multi-state path drops top-level data_files.
  } else {
    // Single-state or ad-hoc multi-file co-refinement. Emit a one-entry
    // `states:` array when the fit files carry a named state (e.g. a
    // single-state setup loaded from YAML) OR the user set any single-state
    // overrides, so the state name and its per-state overrides survive Save
    // Setup / Start Analysis. Otherwise fall back to the flat data_files shape
    // for truly ad-hoc runs (the server synthesizes a state0 on export).
    const distinct = _collectStateNames();
    const singleName = distinct.length === 1 ? distinct[0] : SINGLE_STATE_DEFAULT;
    const ov = stateOverrides[singleName];
    const hasOverrides = ov && Object.keys(ov).length > 0;
    if (distinct.length === 1 || hasOverrides) {
      body.states = [_buildStateEntry(singleName, fitFiles, errors)];
    } else {
      const dataFiles = fitFiles.map(function (f) {
        const stem = _basename(f.path).replace(/\.[^.]+$/, "");
        return { file: f.path, label: stem };
      });
      if (dataFiles.length > 1) body.data_files = dataFiles;
    }
  }

  return { body: body, errors: errors };
}

/* ---- Setup YAML import/export ------------------------------- */

function loadSetupFromFile(file) {
  // Triggered by the hidden <input type="file"> in the card header.
  if (!file) return;
  const fd = new FormData();
  fd.append("file", file);
  fetch("/api/setup/load", { method: "POST", body: fd })
    .then((r) => r.json().then((d) => ({ ok: r.ok, data: d })))
    .then(({ ok, data }) => {
      if (!ok) {
        const msg = (data && data.errors) ? data.errors.join("\n") : "Unknown error";
        alert("Could not load setup:\n" + msg);
        return;
      }
      _applySetupPrefill(data);
    })
    .catch((err) => alert("Network error: " + err))
    .finally(() => {
      // Reset the input so re-loading the same file fires `change` again.
      const inp = document.getElementById("setup-load-input");
      if (inp) inp.value = "";
    });
}

function _applySetupPrefill(payload) {
  // Reuse the existing _restoreFormValues path by stuffing the payload
  // into the global `_prevRun` shape and re-running the restore logic
  // for the relevant fields.
  if (payload.sample_description != null) {
    document.getElementById("sample-desc").value = payload.sample_description || "";
  }
  if (payload.hypothesis != null) {
    document.getElementById("hypothesis").value = payload.hypothesis || "";
  }
  if (payload.max_refinements != null) {
    document.getElementById("max-iterations").value = payload.max_refinements;
  }

  // Build a path → state-name map for file checkboxes.
  const states = payload.states || [];
  const pathToState = {};
  states.forEach(function (st) {
    (st.data_files || []).forEach(function (df) {
      if (df && df.file) pathToState[df.file] = st.name;
    });
  });

  // Reset plotted files to the new list.
  plottedFiles = (payload.data_files || []).map(function (df) {
    return {
      path: df.file,
      isFit: true,
      state: pathToState[df.file] || null,
    };
  });

  // Multi-state mode: turn it on when >= 2 states are declared.
  groupIntoStates = states.length >= 2;
  const gtoggle = document.getElementById("group-into-states");
  if (gtoggle) gtoggle.checked = groupIntoStates;

  // Per-state overrides.
  stateOverrides = {};
  states.forEach(function (st) {
    var ov = {};
    if (st.ambient && typeof st.ambient.rho === "number") ov.ambient = st.ambient.rho;
    ["intensity", "theta_offset", "sample_broadening"].forEach(function (k) {
      if (st[k] && typeof st[k] === "object") {
        ov[k] = {};
        ["init", "min", "max"].forEach(function (sub) {
          if (typeof st[k][sub] === "number") ov[k][sub] = st[k][sub];
        });
        if (!Object.keys(ov[k]).length) delete ov[k];
      }
    });
    if (st.back_reflection === true) ov.back_reflection = true;
    if (st.extra_description) ov.extra_description = st.extra_description;
    if (Object.keys(ov).length) stateOverrides[st.name] = ov;
  });

  // Ties mode.
  if (payload.shared_parameters && payload.shared_parameters.length) {
    tiesMode = "shared";
    tiesText = payload.shared_parameters.join("\n");
  } else if (payload.unshared_parameters && payload.unshared_parameters.length) {
    tiesMode = "unshared";
    tiesText = payload.unshared_parameters.join("\n");
  } else {
    tiesMode = "auto";
    tiesText = "";
  }

  // Re-render UI.
  if (typeof _renderPlottedFilesList === "function") _renderPlottedFilesList();
  if (typeof _renderSetupReflectivityPlot === "function") _renderSetupReflectivityPlot();
  if (typeof _wireTiesPanel === "function") _wireTiesPanel();
  _saveFormValues();
}

function saveSetupToFile() {
  const { body, errors } = _buildAnalysisBody({ skipOutputDir: true });
  if (errors.length) {
    alert(errors.join("\n"));
    return;
  }
  fetch("/api/setup/export", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  })
    .then(function (r) {
      if (!r.ok) {
        return r.json().then(function (d) {
          const msg = (d && d.errors) ? d.errors.join("\n") : "Unknown error";
          throw new Error(msg);
        });
      }
      // Filename from the Content-Disposition header (server provides it).
      var fname = "setup.yaml";
      const cd = r.headers.get("Content-Disposition") || "";
      const m = cd.match(/filename="([^"]+)"/);
      if (m) fname = m[1];
      return r.text().then(function (text) {
        const blob = new Blob([text], { type: "text/yaml" });
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = fname;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
      });
    })
    .catch(function (err) {
      alert("Could not export setup:\n" + err.message);
    });
}

function startAnalysis() {
  const { body, errors } = _buildAnalysisBody({ skipOutputDir: false });
  if (errors.length) {
    alert(errors.join("\n"));
    return;
  }
  _saveFormValues();

  const btn = document.getElementById("btn-start");
  btn.disabled = true;
  btn.innerHTML = '<span class="spinner-border spinner-border-sm"></span> Starting…';

  fetch("/api/start-analysis", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
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
  const DATA_COLORS = ["#6c757d", "#0d6efd", "#198754", "#dc3545", "#fd7e14", "#6610f2", "#20c997", "#d63384"];
  const DATA_SYMBOLS = ["circle", "diamond", "square", "cross", "triangle-up"];

  // Experimental data – per-file traces for co-refinement, single trace otherwise
  if (data.data_files && data.data_files.length > 1) {
    data.data_files.forEach(function (df, i) {
      traces.push({
        x: df.Q, y: df.R,
        error_y: df.dR && df.dR.length
          ? { type: "data", array: df.dR, visible: true, thickness: 1 }
          : undefined,
        mode: "markers", marker: { size: 3, color: DATA_COLORS[i % DATA_COLORS.length], symbol: DATA_SYMBOLS[i % DATA_SYMBOLS.length] },
        name: df.label || ("File " + (i + 1)), type: "scatter",
      });
    });
  } else if (data.Q && data.Q.length) {
    traces.push({
      x: data.Q, y: data.R,
      error_y: data.dR && data.dR.length
        ? { type: "data", array: data.dR, visible: true, thickness: 1 }
        : undefined,
      mode: "markers", marker: { size: 3, color: "#6c757d" },
      name: "Data", type: "scatter",
    });
  }
  var allModels = data.models || [];
  var lastIter = allModels.length ? allModels[allModels.length - 1].iteration : null;
  allModels.forEach(function (m, i) {
    var mIter = m.iteration != null ? m.iteration : i;
    var isFinal = (lastIter != null) ? mIter === lastIter : i === allModels.length - 1;
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
