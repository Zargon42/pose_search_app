const pageInput = document.getElementById("page-input");
const segmentButton = document.getElementById("segment-btn");
const searchButton = document.getElementById("search-btn");
const indexButton = document.getElementById("index-btn");
const statusMessage = document.getElementById("page-status");
const previewSection = document.getElementById("page-preview-section");
const pagePreview = document.getElementById("page-preview");
const overlay = document.getElementById("page-overlay");
const resultsSection = document.getElementById("page-results");
const matchList = document.getElementById("match-list");
const errorSection = document.getElementById("error");

let pageObjectUrl = null;
let imageDimensions = null;

pageInput.addEventListener("change", () => {
  clearError();
  clearResults();
  statusMessage.textContent = "";
  if (pageObjectUrl) {
    URL.revokeObjectURL(pageObjectUrl);
  }
  if (!pageInput.files.length) {
    previewSection.classList.add("hidden");
    return;
  }

  pageObjectUrl = URL.createObjectURL(pageInput.files[0]);
  pagePreview.onload = () => {
    imageDimensions = { width: pagePreview.naturalWidth, height: pagePreview.naturalHeight };
    overlay.width = imageDimensions.width;
    overlay.height = imageDimensions.height;
    previewSection.classList.remove("hidden");
    drawBoxes([], "#2563eb");
  };
  pagePreview.src = pageObjectUrl;
});

function clearError() {
  errorSection.textContent = "";
  errorSection.classList.add("hidden");
}

function clearResults() {
  resultsSection.classList.add("hidden");
  matchList.replaceChildren();
  if (imageDimensions) {
    drawBoxes([], "#2563eb");
  }
}

function drawBoxes(boxes, color) {
  if (!imageDimensions) return;
  const context = overlay.getContext("2d");
  context.clearRect(0, 0, overlay.width, overlay.height);
  context.lineWidth = Math.max(3, imageDimensions.width / 400);
  context.strokeStyle = color;
  context.fillStyle = color;
  context.font = `${Math.max(18, imageDimensions.width / 45)}px sans-serif`;
  boxes.forEach(({ bounds, label }) => {
    const [left, top, right, bottom] = bounds;
    context.strokeRect(left, top, right - left, bottom - top);
    if (label) context.fillText(label, left + 4, top + Math.max(20, imageDimensions.width / 45));
  });
}

function showMatches(matches) {
  matchList.replaceChildren();
  matches.forEach((match) => {
    const item = document.createElement("li");
    const image = document.createElement("img");
    image.src = match.match;
    image.alt = "Matching indexed pose";
    image.loading = "lazy";
    const description = document.createElement("p");
    description.textContent = `Query panel ${match.query_panel_index + 1} — distance ${match.distance.toFixed(4)}`;
    item.append(image, description);
    matchList.append(item);
  });
  resultsSection.classList.remove("hidden");
}

async function submitPage(endpoint, action) {
  clearError();
  clearResults();
  if (!pageInput.files.length) {
    errorSection.textContent = "Choose a page image first.";
    errorSection.classList.remove("hidden");
    return;
  }

  [segmentButton, searchButton, indexButton].forEach((button) => {
    button.disabled = true;
  });
  statusMessage.textContent = action === "segment"
    ? "Finding panel boundaries..."
    : action === "search"
      ? "Detecting poses and searching..."
      : "Detecting poses and updating the index...";

  try {
    const formData = new FormData();
    formData.append("image", pageInput.files[0]);
    const response = await fetch(endpoint, { method: "POST", body: formData });
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || "Page processing failed.");

    if (action === "segment") {
      drawBoxes(data.panels.map((panel) => ({
        bounds: panel.bounds,
        label: `Panel ${panel.panel_index + 1}`,
      })), "#2563eb");
      statusMessage.textContent = `Found ${data.panel_count} panel(s). Segmentation is an automatic estimate; review the boundaries before using the page.`;
    } else if (action === "search") {
      drawBoxes(data.matches.flatMap((match) => [
        { bounds: match.query_panel_bounds, label: `Panel ${match.query_panel_index + 1}` },
        { bounds: match.query_pose_bounds, label: "" },
      ]), "#dc2626");
      showMatches(data.matches);
      statusMessage.textContent = `Detected ${data.pose_count} pose(s) across ${data.panel_count} panel(s).`;
    } else {
      drawBoxes(data.panels.map((panel) => ({
        bounds: panel.bounds,
        label: `Panel ${panel.panel_index + 1}`,
      })), "#2563eb");
      statusMessage.textContent = data.message;
    }
  } catch (error) {
    errorSection.textContent = error.message || "Page processing failed.";
    errorSection.classList.remove("hidden");
    statusMessage.textContent = "";
  } finally {
    [segmentButton, searchButton, indexButton].forEach((button) => {
      button.disabled = false;
    });
  }
}

segmentButton.addEventListener("click", () => submitPage("/api/segment-page", "segment"));
searchButton.addEventListener("click", () => submitPage("/api/full-page/search", "search"));
indexButton.addEventListener("click", () => submitPage("/api/full-page/upload", "index"));

window.addEventListener("beforeunload", () => {
  if (pageObjectUrl) URL.revokeObjectURL(pageObjectUrl);
});
