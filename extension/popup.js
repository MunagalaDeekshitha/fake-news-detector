// Change this to your deployed backend URL once hosted (e.g. Render/HF Spaces).
const API_BASE_URL = "http://localhost:8000";

const inputText = document.getElementById("inputText");
const useSelectionBtn = document.getElementById("useSelection");
const analyzeBtn = document.getElementById("analyzeBtn");
const resultDiv = document.getElementById("result");
const noteDiv = document.getElementById("note");
const articlesDiv = document.getElementById("articles");
const articlesListDiv = document.getElementById("articlesList");
const statusDiv = document.getElementById("status");

// Grab whatever text is currently selected on the active tab.
useSelectionBtn.addEventListener("click", async () => {
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });

  const [{ result }] = await chrome.scripting.executeScript({
    target: { tabId: tab.id },
    func: () => window.getSelection().toString(),
  });

  if (result && result.trim().length > 0) {
    inputText.value = result.trim();
    statusDiv.textContent = "Selected text loaded.";
  } else {
    statusDiv.textContent = "No text selected on the page.";
  }
});

analyzeBtn.addEventListener("click", async () => {
  const text = inputText.value.trim();
  if (!text) {
    statusDiv.textContent = "Please enter or select some text first.";
    return;
  }

  analyzeBtn.disabled = true;
  analyzeBtn.textContent = "Analyzing...";
  resultDiv.style.display = "none";
  noteDiv.style.display = "none";
  articlesDiv.style.display = "none";
  statusDiv.textContent = "";

  try {
    // Use /verify instead of /predict: it returns the model's prediction
    // PLUS live related news articles, so the model's guess is never the
    // only thing the user sees -- especially important for recent events
    // the model was never trained on.
    const res = await fetch(`${API_BASE_URL}/verify`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || `Server error (${res.status})`);
    }

    const data = await res.json();
    showResult(data);
  } catch (e) {
    statusDiv.textContent = `Error: ${e.message}. Is the backend running at ${API_BASE_URL}?`;
  } finally {
    analyzeBtn.disabled = false;
    analyzeBtn.textContent = "Analyze & Verify";
  }
});

function showResult(data) {
  const { prediction, related_articles, note } = data;
  const { label, confidence } = prediction;

  resultDiv.className = label === "real" ? "real" : "fake";
  resultDiv.style.display = "block";
  const emoji = label === "real" ? "✅" : "⚠️";
  const pct = (confidence * 100).toFixed(1);
  resultDiv.textContent = `${emoji} Model says: likely ${label.toUpperCase()} (${pct}% confidence)`;

  if (note) {
    noteDiv.textContent = note;
    noteDiv.style.display = "block";
  }

  articlesListDiv.innerHTML = "";
  if (related_articles && related_articles.length > 0) {
    articlesDiv.style.display = "block";
    related_articles.forEach((a) => {
      const link = document.createElement("a");
      link.className = "article";
      link.href = a.url;
      link.target = "_blank";
      link.rel = "noopener noreferrer";

      const titleDiv = document.createElement("div");
      titleDiv.className = "a-title";
      titleDiv.textContent = a.title;

      const sourceDiv = document.createElement("div");
      sourceDiv.className = "a-source";
      sourceDiv.textContent = a.source || "";

      link.appendChild(titleDiv);
      link.appendChild(sourceDiv);
      articlesListDiv.appendChild(link);
    });
  }
}
