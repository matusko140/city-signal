const el = id => document.getElementById(id);
const series = ["actual", "forecast", "baseline"];
const labels = {actual: "Observed", forecast: "CNN forecast", baseline: "Last week"};
const fmt = new Intl.NumberFormat("en-US", {maximumFractionDigits: 1});
const isoHour = hour => String(hour).padStart(2, "0") + ":00";
const timeLabel = hour => (hour % 12 || 12) + (hour < 12 ? " AM" : " PM");
let data, dayStarts, day, selected = null, timer, cells, colorMax, errorMax;

function svg(tag, attributes, value) {
  const node = document.createElementNS("http://www.w3.org/2000/svg", tag);
  Object.entries(attributes).forEach(([key, attribute]) => node.setAttribute(key, attribute));
  if (value !== undefined) node.textContent = value;
  return node;
}

async function readJSON(path) {
  const response = await fetch(path);
  if (!response.ok) throw new Error(path + " returned " + response.status);
  return response.json();
}

function buildMap(geojson) {
  const { west, east, north, south } = data.bounds;
  const x = lng => 480 * (lng - west) / (east - west);
  const y = lat => 480 * (north - lat) / (north - south);
  if (geojson) {
    for (const { geometry } of geojson.features) {
      const polygons = geometry.type === "MultiPolygon" ? geometry.coordinates : [geometry.coordinates];
      const rings = polygons.flat();
      const path = rings.map(ring => "M" + ring.map(
        ([lng, lat]) => x(lng).toFixed(1) + " " + y(lat).toFixed(1)
      ).join(" L") + " Z").join(" ");
      el("land").append(svg("path", {d: path, "fill-rule": "evenodd"}));
    }
    for (const [name, lng, lat] of [
      ["Bronx", -73.855, 40.864], ["Manhattan", -73.972, 40.783],
      ["Queens", -73.804, 40.724], ["Brooklyn", -73.962, 40.643],
      ["Staten Island", -74.154, 40.568],
    ]) {
      el("borough-labels").append(svg("text", {x: x(lng), y: y(lat)}, name));
    }
  }
  const step = 480 / data.size;
  cells = Array.from({length: data.size ** 2}, (_, i) => {
    const rect = svg("rect", {
      x: i % data.size * step, y: Math.floor(i / data.size) * step,
      width: step, height: step, tabindex: i === 0 ? 0 : -1, role: "button",
    });
    rect.append(svg("title", {}));
    rect.addEventListener("click", () => selectCell(i));
    rect.addEventListener("focus", () => {
      cells.forEach((cell, j) => cell.setAttribute("tabindex", i === j ? 0 : -1));
    });
    rect.addEventListener("keydown", event => {
      const {key} = event;
      if (key === "Enter" || key === " ") {
        event.preventDefault();
        selectCell(i);
      }
      const offset = {ArrowLeft: -1, ArrowRight: 1, ArrowUp: -data.size, ArrowDown: data.size}[key];
      if (offset !== undefined) {
        event.preventDefault();
        cells[Math.min(cells.length - 1, Math.max(0, i + offset))].focus();
      }
    });
    el("cells").append(rect);
    return rect;
  });
}

function selectCell(i) {
  selected = selected === i ? null : i;
  draw();
}

function center(i) {
  const { west, east, north, south } = data.bounds;
  const lat = north - (Math.floor(i / data.size) + .5) * (north - south) / data.size;
  const lon = west + (i % data.size + .5) * (east - west) / data.size;
  return [lat.toFixed(4), lon.toFixed(4)];
}

function color(value, layer, missing) {
  if (missing || value === 0) return "transparent";
  if (layer === "error") {
    const intensity = Math.min(1, Math.abs(value) / errorMax);
    return "hsl(" + (value < 0 ? 215 : 24) + " 88% " + (38 + 29 * intensity) + "% / " + (.35 + .6 * intensity) + ")";
  }
  const ratio = Math.min(1, Math.sqrt(value / colorMax));
  return "hsl(" + (166 - 124 * ratio) + " " + (48 + 44 * ratio) + "% " + (30 + 44 * ratio) + "% / " + (.52 + .42 * ratio) + ")";
}

function draw() {
  const hour = Number(el("hour").value);
  const start = dayStarts.get(day);
  const index = start + hour;
  const present = data.timestamps[index]?.startsWith(day);
  const layer = el("layer").value;
  const errors = layer === "error";
  el("date").value = day;
  el("hour-label").value = timeLabel(hour);
  el("map-time").textContent = day + " · " + timeLabel(hour) + " ET";
  el("export").disabled = !present;
  el("scale").classList.toggle("error", errors);
  el("legend-low").textContent = errors ? "−" + errorMax : "0";
  el("legend-high").textContent = errors ? "+" + errorMax : colorMax;
  el("legend-unit").textContent = errors ? "forecast minus observed" : "complaints";
  cells.forEach((cell, i) => {
    const observed = present ? data.actual[index][i] : null;
    const forecast = present ? data.forecast[index][i] : null;
    const baseline = present ? data.baseline[index][i] : null;
    const value = present ? (errors ? forecast - observed : data[layer][index][i]) : 0;
    const [lat, lon] = center(i);
    const description = lat + "°, " + lon + "°" + (present
      ? " · Observed " + observed + " · CNN " + fmt.format(forecast) + " · Last week " + baseline
      : " · No data for this hour");
    cell.setAttribute("fill", color(value, layer, !present));
    cell.setAttribute("aria-label", description);
    cell.setAttribute("aria-pressed", selected === i);
    cell.firstChild.textContent = description;
    cell.classList.toggle("selected", selected === i);
  });
  el("chart-title").textContent = selected === null ? "Citywide counts" : "Selected cell counts";
  el("clear-cell").hidden = selected === null;
  el("cell-details").textContent = selected === null
    ? "Select a map cell to see its hourly counts."
    : "Cell center " + center(selected).join("°, ") + "°";
  const values = Object.fromEntries(series.map(key => [key, Array.from({length: 24}, (_, h) => {
    const i = start + h;
    if (!data.timestamps[i]?.startsWith(day)) return null;
    const frame = data[key][i];
    return selected === null ? frame.reduce((total, count) => total + count, 0) : frame[selected];
  })]));
  for (const key of series) {
    el(key + "-count").textContent = present ? fmt.format(values[key][hour]) : "—";
  }
  drawChart(values, hour);
}

function drawChart(values, hour) {
  const chart = el("chart");
  chart.replaceChildren();
  const top = 12, bottom = 192, right = 464, left = 40;
  const max = Math.max(1, ...Object.values(values).flat());
  const x = i => left + i * (right - left) / 23;
  const y = value => bottom - value * (bottom - top) / max;
  for (const fraction of [0, .25, .5, .75, 1]) {
    const height = y(max * fraction);
    chart.append(svg("line", {x1: left, x2: right, y1: height, y2: height, stroke: "#354457"}));
    chart.append(svg("text", {x: left - 7, y: height + 4, "text-anchor": "end"}, fmt.format(max * fraction)));
  }
  for (const i of [0, 6, 12, 18, 23]) {
    chart.append(svg("text", {x: x(i), y: 211, "text-anchor": "middle"}, isoHour(i)));
  }
  chart.append(svg("line", {
    x1: x(hour), x2: x(hour), y1: top, y2: bottom, stroke: "#e4eaf1", "stroke-dasharray": "3 5",
  }));
  for (const key of ["baseline", "forecast", "actual"]) {
    let path = "", open = false;
    values[key].forEach((value, i) => {
      if (value === null) {
        open = false;
        return;
      }
      path += (open ? " L" : " M") + x(i) + " " + y(value);
      open = true;
    });
    chart.append(svg("path", {
      d: path, stroke: "var(--" + key + ")", fill: "none", "stroke-width": 2,
      "stroke-dasharray": key === "baseline" ? "4 4" : "none",
    }));
    if (values[key][hour] !== null) {
      chart.append(svg("circle", {cx: x(hour), cy: y(values[key][hour]), r: 3, fill: "var(--" + key + ")"}));
    }
  }
}

function playDay() {
  if (timer) {
    clearInterval(timer);
    timer = undefined;
    el("play").textContent = "▶ Play day";
    return;
  }
  el("play").textContent = "Ⅱ Pause";
  timer = setInterval(() => {
    el("hour").value = (Number(el("hour").value) + 1) % 24;
    draw();
  }, 450);
}

function exportHour() {
  const index = dayStarts.get(day) + Number(el("hour").value);
  const rows = cells.map((_, i) => [
    data.timestamps[index], i + 1, ...center(i), ...series.map(key => data[key][index][i]),
  ].join(","));
  const csv = ["timestamp_nyc,cell,latitude,longitude,observed,cnn_forecast,previous_week", ...rows].join("\n");
  const link = document.createElement("a");
  link.href = URL.createObjectURL(new Blob([csv], {type: "text/csv;charset=utf-8"}));
  link.download = "nyc-complaints-" + data.timestamps[index].replace(":", "-") + ".csv";
  link.click();
  setTimeout(() => URL.revokeObjectURL(link.href), 1000);
}

async function init() {
  const geojson = readJSON("assets/boroughs.geojson").catch(() => null);
  try {
    data = await readJSON("artifacts/dashboard_data.json");
    dayStarts = new Map();
    data.timestamps.forEach((timestamp, i) => {
      const date = timestamp.slice(0, 10);
      if (!dayStarts.has(date)) dayStarts.set(date, i - Number(timestamp.slice(11, 13)));
    });
    const dates = Array.from(dayStarts.keys());
    el("date").min = dates[0];
    el("date").max = dates.at(-1);
    day = dates.at(-1);
    colorMax = errorMax = 1;
    data.actual.forEach((frame, t) => {
      for (let i = 0; i < frame.length; i++) {
        colorMax = Math.max(colorMax, frame[i], data.forecast[t][i], data.baseline[t][i]);
        errorMax = Math.max(errorMax, Math.abs(data.forecast[t][i] - frame[i]));
      }
    });
    colorMax = Math.ceil(colorMax);
    errorMax = Math.ceil(errorMax);
    const metrics = data.metrics;
    el("holdout").textContent = "Evaluated on " + metrics.hours.toLocaleString()
      + " held-out hours from " + data.holdout_start.slice(0, 10) + " through " + data.period_end.slice(0, 10) + ".";
    el("cnn-mae").textContent = metrics.mae.toFixed(2);
    el("baseline-mae").textContent = metrics.baseline_mae.toFixed(2);
    el("cnn-city").textContent = metrics.city_mae.toFixed(2);
    el("baseline-city").textContent = metrics.baseline_city_mae.toFixed(2);
    el("model-note").textContent = "The CNN trained for " + data.epochs
      + (data.epochs === 1 ? " epoch. It uses the previous " : " epochs. It uses the previous ") + data.steps
      + " hours, the same hour last week, and calendar information. The explorer shows the final "
      + dates.length + " days of the holdout.";
    buildMap(await geojson);
    el("status").textContent = data.complaints.toLocaleString() + " geocoded complaints · "
      + data.period_start.slice(0, 10) + " to " + data.period_end.slice(0, 10);
    el("date").addEventListener("change", event => {
      if (dayStarts.has(event.target.value)) day = event.target.value;
      draw();
    });
    el("hour").addEventListener("input", draw);
    el("layer").addEventListener("change", draw);
    el("play").addEventListener("click", playDay);
    el("export").addEventListener("click", exportHour);
    el("clear-cell").addEventListener("click", () => selectCell(selected));
    el("dashboard").hidden = false;
    draw();
  } catch (error) {
    el("status").textContent = "Model results are unavailable. Run the Python pipeline and open this page through the local server.";
    console.error(error);
  }
}

init();
