const form = document.querySelector("#uploadForm");
const mediaInput = document.querySelector("#mediaInput");
const fileName = document.querySelector("#fileName");
const button = document.querySelector("#processButton");
const resultImage = document.querySelector("#resultImage");
const emptyState = document.querySelector("#emptyState");
const labelsList = document.querySelector("#labelsList");
const historyList = document.querySelector("#historyList");
const modelStatus = document.querySelector("#modelStatus");
const countValue = document.querySelector("#countValue");
const confidenceValue = document.querySelector("#confidenceValue");

function renderLabels(labels) {
  labelsList.innerHTML = "";
  if (!labels.length) {
    labelsList.innerHTML = '<div class="meta">Объекты не найдены.</div>';
    return;
  }

  labels.forEach((item) => {
    const node = document.createElement("div");
    node.className = "label-item";
    const box = item.box && item.box.length ? `box: ${item.box.join(", ")}` : "без координат";
    node.innerHTML = `
      <strong>${item.label}</strong>
      <span class="meta">confidence: ${Number(item.confidence || 0).toFixed(3)} · ${box}</span>
    `;
    labelsList.appendChild(node);
  });
}

function renderHistory(items) {
  historyList.innerHTML = "";
  if (!items.length) {
    historyList.innerHTML = '<div class="meta">История пока пустая.</div>';
    return;
  }

  items.forEach((item) => {
    const node = document.createElement("button");
    node.type = "button";
    node.className = "history-item";
    node.innerHTML = `
      <strong>#${item.id} · ${item.filename}</strong>
      <span class="meta">${item.timestamp} · Детекция · объектов: ${item.count}</span>
    `;
    node.addEventListener("click", () => {
      resultImage.src = `${item.result_url}?${Date.now()}`;
      resultImage.style.display = "block";
      emptyState.style.display = "none";
      countValue.textContent = item.count;
      renderLabels(item.result);
    });
    historyList.appendChild(node);
  });
}

async function loadHealth() {
  const response = await fetch("/api/health");
  const data = await response.json();
  modelStatus.textContent = data.model_available ? "YOLOv8" : data.opencv ? "OpenCV" : "Demo";
}

async function loadHistory() {
  const response = await fetch("/api/history");
  const data = await response.json();
  renderHistory(data);
}

mediaInput.addEventListener("change", () => {
  fileName.textContent = mediaInput.files[0]?.name || "JPG, PNG, WEBP, MP4, MOV";
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const file = mediaInput.files[0];
  if (!file) {
    labelsList.innerHTML = '<div class="error">Сначала выберите файл.</div>';
    return;
  }

  const formData = new FormData(form);
  button.disabled = true;
  button.textContent = "Детекция...";
  labelsList.innerHTML = '<div class="meta">Файл передан на сервер.</div>';

  try {
    const response = await fetch("/api/process", {
      method: "POST",
      body: formData,
    });
    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.error || "Ошибка детекции.");
    }

    resultImage.src = `${data.result_url}?${Date.now()}`;
    resultImage.style.display = "block";
    emptyState.style.display = "none";
    modelStatus.textContent = data.model;
    countValue.textContent = data.count;
    confidenceValue.textContent = Number(data.confidence_avg || 0).toFixed(2);
    renderLabels(data.labels || []);
    await loadHistory();
  } catch (error) {
    labelsList.innerHTML = `<div class="error">${error.message}</div>`;
  } finally {
    button.disabled = false;
    button.textContent = "Запустить детекцию";
  }
});

loadHealth().catch(() => {
  modelStatus.textContent = "Неизвестно";
});
loadHistory().catch(() => {
  historyList.innerHTML = '<div class="error">Не удалось загрузить историю.</div>';
});
