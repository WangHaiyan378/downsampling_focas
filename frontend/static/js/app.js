let metadata = null;
let selectedTask = "history2era5";
let activeModule = "train";
let pollTimer = null;
let currentFrames = [];
let currentFrameIndex = 0;
let animTimer = null;
let lastEvalRunning = false;
let lastInferRunning = false;
let lastTrainingError = null;
let lastEvaluationError = null;
let lastInferenceError = null;

const $ = (id) => document.getElementById(id);
const setText = (id, value) => {
    const el = $(id);
    if (el) el.textContent = value;
};

document.addEventListener("DOMContentLoaded", async () => {
    await clearServerErrors();
    await refreshMetadata();
    $("configSelect").addEventListener("change", applyConfigDefaults);
    selectTask("history2era5");
    switchModule("train");
    drawEmpty($("lossChart"), "选择配置后显示历史训练曲线；启动训练后实时更新");
    appendLog("训练监控已就绪");
    startPolling();
    loadDatasetHeatmap();
});

async function refreshMetadata() {
    const data = await fetchJson("/api/metadata");
    metadata = data;
    setText("statConfigs", data.configs.length);
    setText("statCkpts", data.checkpoints.length);
    setText("statReports", data.reports.length);
    setText("statPreds", data.predictions.length);
    populateSelects();
}

function populateSelects() {
    if (!metadata) return;
    populateConfigSelect();
    fillSelect("evalCheckpoint", checkpointOptions(), "不指定 checkpoint");
    fillSelect("inferCheckpoint", checkpointOptions(), "不指定 checkpoint");
    fillSelect("reportSelect", metadata.reports.map(r => ({
        value: r.path,
        label: `${r.task_type || "report"} / ${r.name}`,
    })), "选择评估报告");
    fillSelect("rmseSelect", metadata.rmse_images.map(i => ({
        value: i.path,
        label: `${i.run} / ${i.name}`,
    })), "选择 RMSE 图");
    fillSelect("predictionSelect", metadata.predictions.map(p => ({
        value: p.path,
        label: `${p.run} / ${p.name}`,
    })), "选择推理输出");

    if ($("reportSelect").value) loadReport($("reportSelect").value);
    if ($("rmseSelect").value) showRmseImage($("rmseSelect").value);
    if ($("predictionSelect").value) loadPredictionFrames($("predictionSelect").value);
}

function populateConfigSelect() {
    const configs = metadata.configs
        .filter(c => !c.error && c.task_type === selectedTask)
        .map(c => ({
            value: c.name,
            label: `${c.name}  |  ${displayVars(c.target_vars)}`,
        }));
    fillSelect("configSelect", configs, "没有匹配配置");
    applyConfigDefaults();
}

function checkpointOptions() {
    return metadata.checkpoints.map(c => ({
        value: c.path,
        label: `${c.run} / ${c.name} (${c.size_mb} MB)`,
    }));
}

function fillSelect(id, options, emptyLabel) {
    const el = $(id);
    const old = el.value;
    el.innerHTML = "";
    const empty = document.createElement("option");
    empty.value = "";
    empty.textContent = emptyLabel;
    el.appendChild(empty);
    options.forEach(opt => {
        const node = document.createElement("option");
        node.value = opt.value;
        node.textContent = opt.label;
        el.appendChild(node);
    });
    if ([...el.options].some(o => o.value === old)) {
        el.value = old;
    } else if (options.length) {
        el.value = options[0].value;
    }
}

function selectTask(task) {
    selectedTask = task;
    $("taskHistory").classList.toggle("selected", task === "history2era5");
    $("taskForecast").classList.toggle("selected", task === "fcst2era5fit");
    setText("metricTask", task === "history2era5" ? "Task A" : "Task B");
    populateConfigSelect();
}

function switchModule(name) {
    activeModule = name;
    document.querySelectorAll(".tab").forEach(btn => {
        btn.classList.toggle("active", btn.dataset.module === name);
    });
    document.querySelectorAll(".module-panel").forEach(panel => {
        panel.classList.toggle("active", panel.id === `module-${name}`);
    });
    const visible = {
        train: ["trainView"],
        evaluate: ["evalView", "rmseView"],
        infer: ["inferView"],
        dataset: ["datasetView"],
    }[name] || [];
    ["trainView", "evalView", "rmseView", "inferView", "datasetView"].forEach(id => {
        const el = $(id);
        if (el) el.style.display = visible.includes(id) ? "" : "none";
    });
}

function applyConfigDefaults() {
    const cfg = currentConfigMeta();
    if (!cfg) return;
    $("trainEpochs").value = cfg.epochs || 5;
    $("trainBatch").value = cfg.batch_size || 1;
    $("trainLR").value = cfg.lr || "1e-4";
    loadTrainingHistory();
}

function currentConfigMeta() {
    if (!metadata) return null;
    const name = $("configSelect").value;
    return metadata.configs.find(c => c.name === name);
}

async function runTrain() {
    switchModule("train");
    const payload = basePayload();
    payload.epochs = Number($("trainEpochs").value);
    payload.batch_size = Number($("trainBatch").value);
    payload.lr = Number($("trainLR").value);
    payload.device = $("trainDevice").value;
    payload.num_workers = 0;
    await postJson("/api/train", payload);
    clearLog();
    lastTrainingError = null;
    drawLossChart([]);
    appendLog(`启动训练：${payload.config} / ${payload.dataset_mode}`);
    startPolling();
}

async function stopTrain() {
    await postJson("/api/train/stop", {});
}

async function runEvaluate() {
    switchModule("evaluate");
    const payload = basePayload();
    payload.checkpoint = $("evalCheckpoint").value || null;
    payload.max_samples = Number($("evalSamples").value);
    payload.device = $("evalDevice").value;
    payload.lead_min = Number($("leadMin").value);
    payload.lead_max = Number($("leadMax").value);
    payload.lead_groups = $("leadGroups").value;
    payload.save_rmse_maps = true;
    await postJson("/api/evaluate", payload);
    startPolling();
}

async function runInfer() {
    switchModule("infer");
    const payload = basePayload();
    payload.checkpoint = $("inferCheckpoint").value || null;
    payload.max_samples = Number($("inferSamples").value);
    payload.device = $("inferDevice").value;
    payload.output_dir = $("inferOutput").value;
    await postJson("/api/infer", payload);
    startPolling();
}

function basePayload() {
    return {
        task_type: selectedTask,
        config: $("configSelect").value,
        dataset_mode: $("datasetMode").value,
    };
}

function startPolling() {
    if (pollTimer) clearInterval(pollTimer);
    pollStatus();
    pollTimer = setInterval(pollStatus, 1500);
}

async function pollStatus() {
    try {
        const data = await fetchJson("/api/status");
        updateGpu(data.gpu_available);
        updateRunState(data);
        updateTraining(data.training);
        updateEvaluation(data.evaluation);
        updateInference(data.inference);
    } catch (err) {
        appendLog(`status error: ${err.message}`);
    }
}

function updateGpu(ok) {
    const dot = $("gpuDot");
    if (dot) dot.className = `status-dot ${ok ? "active" : "inactive"}`;
    setText("gpuLabel", ok ? "GPU 可用" : "CPU / 无 CUDA");
}

function updateRunState(data) {
    const running = data.training.running || data.evaluation.running || data.inference.running;
    const state = data.training.running ? "训练中" : data.evaluation.running ? "评估中" : data.inference.running ? "推理中" : "待命";
    setText("metricState", state);
    if (!running) setText("metricProgress", "0%");
}

function updateTraining(state) {
    if (!state) return;
    if (state.running || state.history?.length) {
        const pct = state.total_epochs ? Math.round((state.epoch / state.total_epochs) * 100) : 0;
        setText("metricProgress", `${pct}%`);
        setText("metricElapsed", elapsedText(state.start_time));
        setText("trainSummary", `${state.phase || "training"} | Epoch ${state.epoch || 0}/${state.total_epochs || 0}  Train ${fmt(state.train_loss)}  Val ${fmt(state.val_loss)}`);
        updateTrainKpis(state.epoch || 0, state.total_epochs || 0, state.train_loss, state.val_loss);
        drawLossChart(state.history || []);
        renderTrainLog(state);
    } else if (!state.error) {
        setText("trainSummary", "未运行训练；已显示所选配置的历史日志（如果存在）");
    }
    if (state.error && state.error !== lastTrainingError) {
        lastTrainingError = state.error;
        appendLog(`training error: ${state.error}`);
        if (state.traceback) {
            appendLog("traceback:");
            state.traceback.split("\n").slice(-14).forEach(line => {
                if (line.trim()) appendLog(line);
            });
        }
    }
}

function updateTrainKpis(epoch, total, trainLoss, valLoss) {
    setText("trainEpochMetric", total ? `${epoch} / ${total}` : `${epoch || "--"}`);
    setText("trainLossMetric", fmt(trainLoss));
    setText("valLossMetric", fmt(valLoss));
}

async function loadTrainingHistory() {
    const config = $("configSelect").value;
    if (!config) {
        drawEmpty($("lossChart"), "没有可用训练配置");
        return;
    }
    try {
        const data = await fetchJson(`/api/train/history?config=${encodeURIComponent(config)}`);
        if (data.history && data.history.length) {
            drawLossChart(data.history);
            renderHistoryLog(data.history, data.path);
            const last = data.history[data.history.length - 1] || {};
            updateTrainKpis(last.epoch, data.history.length, last.train_loss, last.val_loss);
            setText("trainSummary", `历史训练日志：${data.history.length} epochs`);
        } else {
            drawEmpty($("lossChart"), "该配置还没有训练日志");
            $("runLog").innerHTML = "";
            updateTrainKpis("--", 0, NaN, NaN);
            appendLog(`当前配置 ${config} 暂无 training_log.json`);
            setText("trainSummary", "暂无历史训练日志");
        }
    } catch (err) {
        drawEmpty($("lossChart"), "训练日志读取失败");
        appendLog(`history error: ${err.message}`);
    }
}

async function updateEvaluation(state) {
    if (!state) return;
    if (state.running) {
        setText("metricProgress", "运行中");
        setText("metricElapsed", elapsedText(state.start_time));
        lastEvalRunning = true;
    } else if (lastEvalRunning) {
        lastEvalRunning = false;
        await refreshMetadata();
        if (state.report_path) loadReport(state.report_path);
    }
    if (state.report) renderReport(state.report);
    if (state.error && state.error !== lastEvaluationError) {
        lastEvaluationError = state.error;
        appendLog(`evaluation error: ${state.error}`);
    }
}

async function updateInference(state) {
    if (!state) return;
    if (state.running) {
        setText("metricProgress", "运行中");
        setText("metricElapsed", elapsedText(state.start_time));
        lastInferRunning = true;
    } else if (lastInferRunning) {
        lastInferRunning = false;
        await refreshMetadata();
        if (metadata.predictions.length) {
            loadPredictionFrames(metadata.predictions[metadata.predictions.length - 1].path);
        }
    }
    if (state.results?.length) {
        appendLog(`inference samples: ${state.results.length}`);
    }
    if (state.error && state.error !== lastInferenceError) {
        lastInferenceError = state.error;
        appendLog(`inference error: ${state.error}`);
    }
}

function renderTrainLog(state) {
    const log = $("runLog");
    log.innerHTML = "";
    (state.history || []).forEach(entry => {
        const line = document.createElement("div");
        line.className = "log-line";
        line.textContent = `Epoch ${entry.epoch}: train=${fmt(entry.train_loss)} val=${fmt(entry.val_loss)} lr=${sci(entry.lr)} time=${entry.time}s`;
        log.appendChild(line);
    });
    log.scrollTop = log.scrollHeight;
}

function renderHistoryLog(history, path) {
    const log = $("runLog");
    log.innerHTML = "";
    const title = document.createElement("div");
    title.className = "log-line";
    title.textContent = path ? `Loaded ${path}` : "No training log";
    log.appendChild(title);
    history.forEach(entry => {
        const line = document.createElement("div");
        line.className = "log-line";
        line.textContent = `Epoch ${entry.epoch}: train=${fmt(entry.train_loss)} val=${fmt(entry.val_loss)} lr=${sci(entry.lr)} time=${entry.time}s`;
        log.appendChild(line);
    });
    log.scrollTop = log.scrollHeight;
}

function appendLog(text) {
    const log = $("runLog");
    if (!log) return;
    const line = document.createElement("div");
    line.className = "log-line";
    line.textContent = `[${new Date().toLocaleTimeString()}] ${text}`;
    log.appendChild(line);
    log.scrollTop = log.scrollHeight;
}

function clearLog() {
    $("runLog").innerHTML = "";
}

async function loadReport(path) {
    if (!path) return;
    const data = await fetchJson(`/api/eval/report?path=${encodeURIComponent(path)}`);
    renderReport(data.report);
}

function renderReport(report) {
    const metrics = flattenMetrics(report.overall_metrics || {});
    const box = $("evalMetrics");
    box.innerHTML = "";
    metrics.slice(0, 8).forEach(([key, value]) => {
        const cell = document.createElement("div");
        cell.className = "metric-cell";
        cell.innerHTML = `<small>${escapeHtml(key)}</small><strong>${fmt(value)}</strong>`;
        box.appendChild(cell);
    });
    drawLeadChart(report);
}

function showRmseImage(path) {
    if (!path) return;
    $("rmseImage").src = `/api/rmse-image?path=${encodeURIComponent(path)}`;
}

async function loadDatasetHeatmap() {
    const params = new URLSearchParams({
        source: $("heatSource").value,
        task: $("heatTask").value,
        var: $("heatVar").value,
        field: $("heatField").value,
        lead: $("heatLead").value,
        model: $("heatModel").value,
    });
    const data = await fetchJson(`/api/dataset/heatmap?${params.toString()}`);
    $("datasetLabel").textContent = `${data.label} / ${data.sample} / lead ${data.lead}`;
    drawHeatmap($("datasetCanvas"), data, `${data.label}  mean=${fmt(data.mean)}`);
}

async function loadPredictionFrames(path) {
    if (!path) return;
    const data = await fetchJson(`/api/inference/frames?path=${encodeURIComponent(path)}`);
    currentFrames = data.frames || [];
    currentFrameIndex = 0;
    $("frameSlider").max = Math.max(0, currentFrames.length - 1);
    $("frameSlider").value = 0;
    showFrame(0);
}

function showFrame(index) {
    if (!currentFrames.length) {
        drawEmpty($("windCanvas"), "暂无推理风速帧");
        return;
    }
    currentFrameIndex = Math.max(0, Math.min(index, currentFrames.length - 1));
    const frame = currentFrames[currentFrameIndex];
    $("frameSlider").value = currentFrameIndex;
    $("frameLabel").textContent = `Lead ${frame.lead}`;
    drawHeatmap($("windCanvas"), frame, `预测风速动态  Lead ${frame.lead}d`);
}

function toggleAnimation() {
    if (animTimer) {
        clearInterval(animTimer);
        animTimer = null;
        $("animButton").textContent = "播放";
        return;
    }
    $("animButton").textContent = "暂停";
    animTimer = setInterval(() => {
        const next = (currentFrameIndex + 1) % Math.max(1, currentFrames.length);
        showFrame(next);
    }, 650);
}

function drawLossChart(history) {
    const train = (history || []).map(h => Number(h.train_loss)).filter(Number.isFinite);
    const val = (history || []).map(h => Number(h.val_loss)).filter(Number.isFinite);
    drawLineChart($("lossChart"), [
        { label: "Train", values: train, color: "#57d5d2" },
        { label: "Val", values: val, color: "#f3bd5b" },
    ], "Loss");
}

function drawLeadChart(report) {
    const leads = [];
    const vals = [];
    Object.entries(report.per_lead_metrics || {}).forEach(([key, metric]) => {
        const lead = Number((key.match(/\d+/) || [leads.length + 1])[0]);
        const value = leadMetricValue(metric);
        if (Number.isFinite(value)) {
            leads.push(lead);
            vals.push(value);
        }
    });
    drawLineChart($("leadChart"), [{ label: "RMSE", values: vals, labels: leads, color: "#7bd88f" }], "RMSE by Lead");
}

function drawLineChart(canvas, series, title) {
    const { ctx, width, height } = canvasContext(canvas);
    ctx.clearRect(0, 0, width, height);
    paintPanelGrid(ctx, width, height);
    const all = series.flatMap(s => s.values).filter(Number.isFinite);
    if (all.length < 2) {
        drawCanvasText(ctx, width, height, "暂无可绘制曲线：loss 为空或为 NaN");
        return;
    }
    const pad = { l: 48, r: 18, t: 28, b: 34 };
    const minY = Math.min(...all);
    const maxY = Math.max(...all);
    const range = Math.max(1e-9, maxY - minY);
    ctx.fillStyle = "#9fb7b5";
    ctx.font = "12px Segoe UI";
    ctx.fillText(title, pad.l, 18);

    for (let i = 0; i <= 4; i++) {
        const y = pad.t + (i / 4) * (height - pad.t - pad.b);
        ctx.strokeStyle = "rgba(255,255,255,.08)";
        ctx.beginPath();
        ctx.moveTo(pad.l, y);
        ctx.lineTo(width - pad.r, y);
        ctx.stroke();
        const tick = maxY - (i / 4) * range;
        ctx.fillStyle = "#688280";
        ctx.textAlign = "right";
        ctx.fillText(fmt(tick), pad.l - 7, y + 4);
    }

    series.forEach((s, si) => {
        if (s.values.length < 2) return;
        const n = s.values.length;
        const plotW = width - pad.l - pad.r;
        const plotH = height - pad.t - pad.b;
        const x = i => pad.l + (i / Math.max(1, n - 1)) * plotW;
        const y = v => pad.t + (1 - (v - minY) / range) * plotH;
        ctx.strokeStyle = s.color;
        ctx.lineWidth = 2.2;
        ctx.beginPath();
        s.values.forEach((v, i) => {
            if (i === 0) ctx.moveTo(x(i), y(v));
            else ctx.lineTo(x(i), y(v));
        });
        ctx.stroke();
        ctx.fillStyle = s.color;
        ctx.textAlign = "left";
        ctx.fillText(s.label, width - 92, 18 + si * 16);
    });
}

function drawHeatmap(canvas, payload, title) {
    const { ctx, width, height } = canvasContext(canvas);
    ctx.clearRect(0, 0, width, height);
    paintPanelGrid(ctx, width, height);
    const values = payload.values || [];
    if (!values.length) {
        drawCanvasText(ctx, width, height, "暂无热力图数据");
        return;
    }
    const rows = values.length;
    const cols = values[0].length;
    const pad = { l: 52, r: 22, t: 34, b: 40 };
    const plotW = width - pad.l - pad.r;
    const plotH = height - pad.t - pad.b;
    const min = Number.isFinite(payload.min) ? payload.min : 0;
    const max = Number.isFinite(payload.max) ? payload.max : min + 1;
    const span = Math.max(1e-9, max - min);

    for (let r = 0; r < rows; r++) {
        for (let c = 0; c < cols; c++) {
            const v = values[r][c];
            if (!Number.isFinite(v)) continue;
            ctx.fillStyle = heatColor((v - min) / span);
            ctx.fillRect(
                pad.l + (c / cols) * plotW,
                pad.t + (r / rows) * plotH,
                Math.ceil(plotW / cols) + 1,
                Math.ceil(plotH / rows) + 1,
            );
        }
    }

    ctx.strokeStyle = "rgba(238,247,244,.72)";
    ctx.lineWidth = 1;
    ctx.strokeRect(pad.l, pad.t, plotW, plotH);
    drawMapAxes(ctx, payload, pad, plotW, plotH, width, height);
    drawLegend(ctx, min, max, width - 118, pad.t + 8, 86, 10);
    ctx.fillStyle = "#eef7f4";
    ctx.font = "13px Segoe UI";
    ctx.textAlign = "left";
    ctx.fillText(title, pad.l, 20);
}

function drawMapAxes(ctx, payload, pad, plotW, plotH, width, height) {
    ctx.strokeStyle = "rgba(255,255,255,.12)";
    ctx.fillStyle = "#9fb7b5";
    ctx.font = "11px Segoe UI";
    ctx.textAlign = "center";
    for (let i = 0; i <= 4; i++) {
        const x = pad.l + (i / 4) * plotW;
        ctx.beginPath();
        ctx.moveTo(x, pad.t);
        ctx.lineTo(x, pad.t + plotH);
        ctx.stroke();
        const lon = payload.region.lon_min + (i / 4) * (payload.region.lon_max - payload.region.lon_min);
        ctx.fillText(`${Math.round(lon)}E`, x, height - 16);
    }
    ctx.textAlign = "right";
    for (let i = 0; i <= 4; i++) {
        const y = pad.t + (i / 4) * plotH;
        ctx.beginPath();
        ctx.moveTo(pad.l, y);
        ctx.lineTo(pad.l + plotW, y);
        ctx.stroke();
        const lat = payload.region.lat_max - (i / 4) * (payload.region.lat_max - payload.region.lat_min);
        ctx.fillText(`${Math.round(lat)}N`, pad.l - 8, y + 4);
    }
    ctx.textAlign = "left";
    ctx.fillStyle = "rgba(238,247,244,.78)";
    ctx.fillText("China domain", pad.l + 8, pad.t + plotH - 10);
}

function drawLegend(ctx, min, max, x, y, w, h) {
    for (let i = 0; i < w; i++) {
        ctx.fillStyle = heatColor(i / Math.max(1, w - 1));
        ctx.fillRect(x + i, y, 1, h);
    }
    ctx.strokeStyle = "rgba(255,255,255,.34)";
    ctx.strokeRect(x, y, w, h);
    ctx.fillStyle = "#9fb7b5";
    ctx.font = "10px Segoe UI";
    ctx.textAlign = "left";
    ctx.fillText(fmt(min), x, y + h + 14);
    ctx.textAlign = "right";
    ctx.fillText(fmt(max), x + w, y + h + 14);
}

function heatColor(t) {
    const stops = [
        [9, 19, 24],
        [31, 91, 120],
        [87, 213, 210],
        [123, 216, 143],
        [243, 189, 91],
        [255, 107, 107],
    ];
    t = Math.max(0, Math.min(1, t));
    const p = t * (stops.length - 1);
    const i = Math.min(stops.length - 2, Math.floor(p));
    const f = p - i;
    const a = stops[i], b = stops[i + 1];
    const rgb = a.map((v, idx) => Math.round(v + (b[idx] - v) * f));
    return `rgb(${rgb[0]},${rgb[1]},${rgb[2]})`;
}

function canvasContext(canvas) {
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    const cssHeight = Number(canvas.getAttribute("height")) || rect.height || 260;
    const width = Math.max(320, rect.width);
    const height = cssHeight;
    canvas.width = Math.floor(width * dpr);
    canvas.height = Math.floor(height * dpr);
    const ctx = canvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    return { ctx, width, height };
}

function paintPanelGrid(ctx, width, height) {
    ctx.fillStyle = "rgba(2,8,10,.64)";
    ctx.fillRect(0, 0, width, height);
    ctx.strokeStyle = "rgba(87,213,210,.06)";
    for (let x = 0; x < width; x += 28) {
        ctx.beginPath();
        ctx.moveTo(x, 0);
        ctx.lineTo(x, height);
        ctx.stroke();
    }
    for (let y = 0; y < height; y += 28) {
        ctx.beginPath();
        ctx.moveTo(0, y);
        ctx.lineTo(width, y);
        ctx.stroke();
    }
}

function drawEmpty(canvas, text) {
    const { ctx, width, height } = canvasContext(canvas);
    paintPanelGrid(ctx, width, height);
    drawCanvasText(ctx, width, height, text);
}

function drawCanvasText(ctx, width, height, text) {
    ctx.fillStyle = "#9fb7b5";
    ctx.font = "14px Segoe UI";
    ctx.textAlign = "center";
    ctx.fillText(text, width / 2, height / 2);
}

function flattenMetrics(obj, prefix = "") {
    const out = [];
    Object.entries(obj).forEach(([key, value]) => {
        const name = prefix ? `${prefix}.${key}` : key;
        if (value && typeof value === "object" && !Array.isArray(value)) {
            out.push(...flattenMetrics(value, name));
        } else if (typeof value === "number") {
            out.push([name, value]);
        }
    });
    return out;
}

function leadMetricValue(metric) {
    if (typeof metric.rmse === "number") return metric.rmse;
    if (metric.wind_speed?.rmse) return metric.wind_speed.rmse;
    if (metric["wind_speed.rmse"]) return metric["wind_speed.rmse"];
    if (metric.u10?.rmse && metric.v10?.rmse) return (metric.u10.rmse + metric.v10.rmse) / 2;
    if (metric["u10.rmse"] && metric["v10.rmse"]) return (metric["u10.rmse"] + metric["v10.rmse"]) / 2;
    return NaN;
}

async function fetchJson(url) {
    const resp = await fetch(url);
    const data = await resp.json();
    if (!resp.ok || data.error) throw new Error(data.error || resp.statusText);
    return data;
}

async function clearServerErrors() {
    try {
        await postJson("/api/errors/clear", {});
    } catch (_) {
        // Older server process may not have this endpoint yet; restarting the server fixes that.
    }
}

async function postJson(url, payload) {
    const resp = await fetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
    });
    const data = await resp.json();
    if (!resp.ok || data.error) throw new Error(data.error || resp.statusText);
    return data;
}

function displayVars(vars) {
    return Array.isArray(vars) && vars.length ? vars.join(",") : "u/v wind";
}

function fmt(v) {
    if (!Number.isFinite(Number(v))) return "--";
    const n = Number(v);
    if (Math.abs(n) >= 1000 || Math.abs(n) < 0.001 && n !== 0) return n.toExponential(2);
    return n.toFixed(Math.abs(n) >= 10 ? 2 : 4);
}

function sci(v) {
    return Number.isFinite(Number(v)) ? Number(v).toExponential(2) : "--";
}

function elapsedText(start) {
    if (!start) return "0s";
    const s = Math.max(0, Date.now() / 1000 - start);
    const m = Math.floor(s / 60);
    const sec = Math.floor(s % 60);
    return m ? `${m}m ${sec}s` : `${sec}s`;
}

function escapeHtml(text) {
    return String(text).replace(/[&<>"']/g, ch => ({
        "&": "&amp;",
        "<": "&lt;",
        ">": "&gt;",
        "\"": "&quot;",
        "'": "&#039;",
    }[ch]));
}
