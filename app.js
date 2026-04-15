const state = {
  prompts: {
    search: [
      "IDF が高い単語ほど、どの文書の識別に効いているかを観察してみましょう。",
      "クエリに頻出語だけを入れたときと、珍しい語を入れたときで順位はどう変わるでしょうか。",
      "TF を『文書内での重要さ』、IDF を『集合全体での珍しさ』として読むと直感がつかみやすくなります。"
    ],
    recommend: [
      "中心化をオン・オフして、似ているユーザーの順番や予測値がどう変わるか比べてみましょう。",
      "高評価をつけがちな人と辛口な人を、平均値の差としてどう扱うかが中心化のポイントです。",
      "類似度が負になるユーザーは、逆方向の嗜好を持つと読めます。"
    ],
    vision: [
      "カーネルの符号を反転すると、どんな特徴が強調されるか見てみましょう。",
      "Softmax は logit の差に敏感です。1つだけ logit を大きくしたとき確率はどこまで偏るでしょうか。",
      "交差エントロピーは『正解クラスにどれだけ確率を置けたか』を測っています。"
    ],
    attention: [
      "注目トークン index を切り替えて、どのトークンへの重みが増えるか見てみましょう。",
      "Q と K の内積が大きい組み合わせほど、参照の強さが大きくなります。",
      "V は『どの情報を運ぶか』、Softmax の重みは『どこから集めるか』に対応します。"
    ],
    signal: [
      "時間波形が複雑でも、周波数側では少数の強い成分に分解できることがあります。",
      "正弦波プリセットでは、特定の周波数成分だけが強く立つはずです。",
      "混合波では『どの周波数が足し合わされていたか』をスペクトルから逆読みできます。"
    ]
  }
};

const els = {
  tabs: [...document.querySelectorAll(".tab")],
  panels: [...document.querySelectorAll(".lab-panel")],
  promptBox: document.getElementById("prompt-box"),
  notes: document.getElementById("study-notes"),
  saveStatus: document.getElementById("save-status"),
  clearNotes: document.getElementById("clear-notes"),
  search: {
    query: document.getElementById("search-query"),
    docs: document.getElementById("search-docs"),
    summary: document.getElementById("search-summary"),
    terms: document.getElementById("search-terms"),
    ranking: document.getElementById("search-ranking")
  },
  recommend: {
    target: document.getElementById("recommend-target"),
    peers: document.getElementById("recommend-peers"),
    centered: document.getElementById("recommend-centered"),
    summary: document.getElementById("recommend-summary"),
    similarities: document.getElementById("recommend-similarities"),
    explain: document.getElementById("recommend-explain")
  },
  vision: {
    image: document.getElementById("vision-image"),
    kernel: document.getElementById("vision-kernel"),
    logits: document.getElementById("vision-logits"),
    target: document.getElementById("vision-target"),
    summary: document.getElementById("vision-summary"),
    conv: document.getElementById("vision-conv"),
    softmax: document.getElementById("vision-softmax")
  },
  attention: {
    q: document.getElementById("attention-q"),
    k: document.getElementById("attention-k"),
    v: document.getElementById("attention-v"),
    focus: document.getElementById("attention-focus"),
    summary: document.getElementById("attention-summary"),
    scores: document.getElementById("attention-scores"),
    context: document.getElementById("attention-context")
  },
  signal: {
    values: document.getElementById("signal-values"),
    presets: [...document.querySelectorAll(".preset-btn")],
    summary: document.getElementById("signal-summary"),
    spectrum: document.getElementById("signal-spectrum"),
    explain: document.getElementById("signal-explain")
  }
};

function tokenize(text) {
  return text
    .toLowerCase()
    .replace(/[.,/#!$%^&*;:{}=\-_`~()]/g, " ")
    .split(/\s+/)
    .filter(Boolean);
}

function parseNumberList(value) {
  return value
    .split(",")
    .map((item) => item.trim())
    .filter(Boolean)
    .map((item) => {
      if (item === "?") {
        return null;
      }
      const parsed = Number(item);
      return Number.isFinite(parsed) ? parsed : null;
    });
}

function parseNamedVectors(text) {
  return text
    .split("\n")
    .map((line) => line.trim())
    .filter(Boolean)
    .map((line, index) => {
      const [rawName, rawValues] = line.includes(":")
        ? line.split(":")
        : [`User ${index + 1}`, line];
      return {
        name: rawName.trim(),
        values: parseNumberList(rawValues)
      };
    });
}

function parseMatrix(text) {
  return text
    .trim()
    .split("\n")
    .map((line) =>
      line
        .trim()
        .split(/[\s,]+/)
        .map(Number)
        .filter((value) => Number.isFinite(value))
    )
    .filter((row) => row.length > 0);
}

function formatNumber(value, digits = 3) {
  if (!Number.isFinite(value)) {
    return "—";
  }
  return Number(value).toFixed(digits).replace(/\.?0+$/, "");
}

function buildMetric(label, value) {
  return `<div class="metric"><span class="metric-label">${label}</span><span class="metric-value">${value}</span></div>`;
}

function buildTable(headers, rows) {
  const head = headers.map((header) => `<th>${header}</th>`).join("");
  const body = rows
    .map(
      (row) =>
        `<tr>${row
          .map((cell) => `<td>${cell}</td>`)
          .join("")}</tr>`
    )
    .join("");
  return `<table><thead><tr>${head}</tr></thead><tbody>${body}</tbody></table>`;
}

function sum(values) {
  return values.reduce((acc, value) => acc + value, 0);
}

function dot(a, b) {
  return a.reduce((acc, value, index) => acc + value * b[index], 0);
}

function norm(values) {
  return Math.sqrt(dot(values, values));
}

function softmax(values) {
  const max = Math.max(...values);
  const exps = values.map((value) => Math.exp(value - max));
  const total = sum(exps);
  return exps.map((value) => value / total);
}

function transpose(matrix) {
  return matrix[0].map((_, colIndex) => matrix.map((row) => row[colIndex]));
}

function multiplyMatrices(a, b) {
  return a.map((row) =>
    transpose(b).map((column) => dot(row, column))
  );
}

function updateSearchLab() {
  const queryTokens = tokenize(els.search.query.value);
  const documents = els.search.docs.value
    .split("\n")
    .map((line) => line.trim())
    .filter(Boolean);

  const tokenizedDocs = documents.map((doc) => tokenize(doc));
  const uniqueTerms = [...new Set(queryTokens)];
  const documentCount = documents.length || 1;

  const termStats = uniqueTerms.map((term) => {
    const df = tokenizedDocs.filter((tokens) => tokens.includes(term)).length;
    const idf = Math.log((documentCount + 1) / (df + 1)) + 1;
    return { term, df, idf };
  });

  const rankedDocs = documents.map((doc, index) => {
    const tokens = tokenizedDocs[index];
    const score = termStats.reduce((acc, { term, idf }) => {
      const count = tokens.filter((token) => token === term).length;
      const tf = tokens.length ? count / tokens.length : 0;
      return acc + tf * idf;
    }, 0);
    return {
      document: `Doc ${index + 1}`,
      text: doc,
      score
    };
  }).sort((a, b) => b.score - a.score);

  els.search.summary.innerHTML = [
    buildMetric("文書数", String(documents.length)),
    buildMetric("クエリ語数", String(uniqueTerms.length)),
    buildMetric("上位文書", rankedDocs[0]?.document ?? "—")
  ].join("");

  els.search.terms.innerHTML = buildTable(
    ["Term", "DF", "IDF"],
    termStats.map(({ term, df, idf }) => [term, df, formatNumber(idf)])
  );

  els.search.ranking.innerHTML = buildTable(
    ["Rank", "Document", "Score", "Text"],
    rankedDocs.map((doc, index) => [
      index + 1,
      doc.document,
      formatNumber(doc.score),
      doc.text
    ])
  );
}

function cosineSimilarity(a, b, centered = false) {
  const paired = a
    .map((value, index) => [value, b[index]])
    .filter(([left, right]) => left !== null && right !== null);

  if (paired.length === 0) {
    return 0;
  }

  let left = paired.map(([value]) => value);
  let right = paired.map(([, value]) => value);

  if (centered) {
    const leftMean = sum(left) / left.length;
    const rightMean = sum(right) / right.length;
    left = left.map((value) => value - leftMean);
    right = right.map((value) => value - rightMean);
  }

  const denominator = norm(left) * norm(right);
  if (!denominator) {
    return 0;
  }

  return dot(left, right) / denominator;
}

function meanKnown(values) {
  const known = values.filter((value) => value !== null);
  return known.length ? sum(known) / known.length : 0;
}

function updateRecommendLab() {
  const target = parseNumberList(els.recommend.target.value);
  const peers = parseNamedVectors(els.recommend.peers.value);
  const centered = els.recommend.centered.checked;
  const missingIndex = target.findIndex((value) => value === null);

  const sims = peers.map((peer) => ({
    ...peer,
    similarity: cosineSimilarity(target, peer.values, centered)
  })).sort((a, b) => b.similarity - a.similarity);

  const eligible = sims.filter((peer) => missingIndex >= 0 && peer.values[missingIndex] !== null);
  const denominator = sum(eligible.map((peer) => Math.abs(peer.similarity)));
  const targetMean = meanKnown(target);

  let prediction = null;
  if (missingIndex >= 0 && denominator) {
    if (centered) {
      const numerator = sum(
        eligible.map((peer) => peer.similarity * (peer.values[missingIndex] - meanKnown(peer.values)))
      );
      prediction = targetMean + numerator / denominator;
    } else {
      prediction = sum(eligible.map((peer) => peer.similarity * peer.values[missingIndex])) / denominator;
    }
  }

  els.recommend.summary.innerHTML = [
    buildMetric("欠損 index", missingIndex >= 0 ? String(missingIndex) : "なし"),
    buildMetric("最も近い人", sims[0]?.name ?? "—"),
    buildMetric("予測評価", prediction === null ? "—" : formatNumber(prediction))
  ].join("");

  els.recommend.similarities.innerHTML = buildTable(
    ["User", "Similarity", "予測対象の評価"],
    sims.map((peer) => [
      peer.name,
      formatNumber(peer.similarity),
      missingIndex >= 0 && peer.values[missingIndex] !== null ? peer.values[missingIndex] : "—"
    ])
  );

  els.recommend.explain.innerHTML = missingIndex < 0
    ? "ターゲット評価に <code>?</code> がありません。未評価アイテムを1つ入れると予測値を計算できます。"
    : `予測対象は <strong>item ${missingIndex}</strong> です。${centered ? "中心化あり" : "中心化なし"}で、
      類似度の絶対値を重みとして近いユーザーの評価を集約しています。`;
}

function convolve(image, kernel) {
  const rows = image.length;
  const cols = image[0]?.length ?? 0;
  const kRows = kernel.length;
  const kCols = kernel[0]?.length ?? 0;
  const out = [];

  for (let i = 0; i <= rows - kRows; i += 1) {
    const row = [];
    for (let j = 0; j <= cols - kCols; j += 1) {
      let total = 0;
      for (let ki = 0; ki < kRows; ki += 1) {
        for (let kj = 0; kj < kCols; kj += 1) {
          total += image[i + ki][j + kj] * kernel[ki][kj];
        }
      }
      row.push(total);
    }
    out.push(row);
  }

  return out;
}

function updateVisionLab() {
  const image = parseMatrix(els.vision.image.value);
  const kernel = parseMatrix(els.vision.kernel.value);
  const logits = parseNumberList(els.vision.logits.value).filter((value) => value !== null);
  const targetIndex = Number(els.vision.target.value);

  const conv = image.length && kernel.length ? convolve(image, kernel) : [];
  const relu = conv.map((row) => row.map((value) => Math.max(0, value)));
  const probs = logits.length ? softmax(logits) : [];
  const loss = probs[targetIndex] ? -Math.log(probs[targetIndex]) : null;
  const strongest = probs.length ? probs.indexOf(Math.max(...probs)) : null;

  els.vision.summary.innerHTML = [
    buildMetric("特徴マップ", `${conv.length}×${conv[0]?.length ?? 0}`),
    buildMetric("予測クラス", strongest === null ? "—" : String(strongest)),
    buildMetric("交差エントロピー", loss === null ? "—" : formatNumber(loss))
  ].join("");

  els.vision.conv.innerHTML = buildTable(
    ["段階", "値"],
    [
      ["Convolution", matrixToInline(conv)],
      ["ReLU", matrixToInline(relu)]
    ]
  );

  els.vision.softmax.innerHTML = buildTable(
    ["Class", "Logit", "Probability"],
    logits.map((logit, index) => [
      index,
      formatNumber(logit),
      formatNumber(probs[index])
    ])
  );
}

function scaleMatrix(matrix, factor) {
  return matrix.map((row) => row.map((value) => value / factor));
}

function updateAttentionLab() {
  const q = parseMatrix(els.attention.q.value);
  const k = parseMatrix(els.attention.k.value);
  const v = parseMatrix(els.attention.v.value);
  const focus = Number(els.attention.focus.value);

  if (!q.length || !k.length || !v.length) {
    return;
  }

  const kT = transpose(k);
  const rawScores = multiplyMatrices(q, kT);
  const scaledScores = scaleMatrix(rawScores, Math.sqrt(k[0].length || 1));
  const focusRow = scaledScores[focus] ?? scaledScores[0];
  const weights = softmax(focusRow);
  const context = transpose(v).map((column) => dot(weights, column));

  els.attention.summary.innerHTML = [
    buildMetric("token数", String(q.length)),
    buildMetric("次元数", String(q[0].length)),
    buildMetric("最大重み先", String(weights.indexOf(Math.max(...weights))))
  ].join("");

  els.attention.scores.innerHTML = buildTable(
    ["Token", "Scaled score", "Attention weight"],
    focusRow.map((score, index) => [
      index,
      formatNumber(score),
      formatNumber(weights[index])
    ])
  );

  els.attention.context.innerHTML = `
    注目トークン <strong>${focus}</strong> の文脈ベクトルは
    <strong>[${context.map((value) => formatNumber(value)).join(", ")}]</strong> です。
    これは V 行列の各行を Attention weight で重み付け平均した結果です。
  `;
}

function dft(values) {
  const N = values.length;
  return Array.from({ length: N }, (_, k) => {
    let real = 0;
    let imag = 0;
    for (let n = 0; n < N; n += 1) {
      const angle = (2 * Math.PI * k * n) / N;
      real += values[n] * Math.cos(angle);
      imag -= values[n] * Math.sin(angle);
    }
    return {
      k,
      real,
      imag,
      magnitude: Math.sqrt(real ** 2 + imag ** 2)
    };
  });
}

function polylinePoints(values, width, height, padding) {
  if (!values.length) {
    return "";
  }
  const min = Math.min(...values);
  const max = Math.max(...values);
  const range = max - min || 1;
  return values
    .map((value, index) => {
      const x = padding + (index * (width - padding * 2)) / Math.max(values.length - 1, 1);
      const y = height - padding - ((value - min) / range) * (height - padding * 2);
      return `${x},${y}`;
    })
    .join(" ");
}

function renderSignalChart(values, title) {
  const points = polylinePoints(values, 420, 180, 18);
  return `
    <div class="chart-card">
      <h4>${title}</h4>
      <svg viewBox="0 0 420 180" aria-label="${title}">
        <rect x="0" y="0" width="420" height="180" rx="16" fill="#fff7ed"></rect>
        <polyline fill="none" stroke="#0f766e" stroke-width="3" points="${points}"></polyline>
      </svg>
    </div>
  `;
}

function renderSpectrumChart(values) {
  const max = Math.max(...values, 1);
  const barWidth = 26;
  return `
    <div class="chart-card">
      <h4>スペクトル</h4>
      <svg viewBox="0 0 420 180" aria-label="スペクトル">
        <rect x="0" y="0" width="420" height="180" rx="16" fill="#fff7ed"></rect>
        ${values
          .map((value, index) => {
            const height = (value / max) * 120;
            const x = 20 + index * 48;
            const y = 150 - height;
            return `
              <rect x="${x}" y="${y}" width="${barWidth}" height="${height}" rx="6" fill="#d97757"></rect>
              <text x="${x + 13}" y="168" text-anchor="middle" font-size="12" fill="#5e6d67">k=${index}</text>
            `;
          })
          .join("")}
      </svg>
    </div>
  `;
}

function updateSignalLab() {
  const values = parseNumberList(els.signal.values.value).filter((value) => value !== null);
  if (!values.length) {
    return;
  }

  const spectrum = dft(values);
  const magnitudes = spectrum.map((entry) => entry.magnitude);
  const dominant = spectrum.reduce((best, current) => current.magnitude > best.magnitude ? current : best, spectrum[0]);

  els.signal.summary.innerHTML = [
    buildMetric("サンプル数", String(values.length)),
    buildMetric("支配周波数", `k = ${dominant.k}`),
    buildMetric("最大振幅", formatNumber(dominant.magnitude))
  ].join("");

  els.signal.spectrum.innerHTML = `
    ${renderSignalChart(values, "時間領域の信号")}
    ${renderSpectrumChart(magnitudes)}
  `;

  els.signal.explain.innerHTML = buildTable(
    ["k", "Real", "Imag", "Magnitude"],
    spectrum.map((entry) => [
      entry.k,
      formatNumber(entry.real),
      formatNumber(entry.imag),
      formatNumber(entry.magnitude)
    ])
  );
}

function matrixToInline(matrix) {
  if (!matrix.length) {
    return "—";
  }
  return `[${matrix.map((row) => `[${row.map((value) => formatNumber(value)).join(", ")}]`).join(", ")}]`;
}

function setPrompt(tabName) {
  const items = state.prompts[tabName] ?? [];
  els.promptBox.innerHTML = `<ul>${items.map((item) => `<li>${item}</li>`).join("")}</ul>`;
}

function setActiveTab(tabName) {
  els.tabs.forEach((tab) => {
    const isActive = tab.dataset.tab === tabName;
    tab.classList.toggle("is-active", isActive);
    tab.setAttribute("aria-selected", String(isActive));
  });

  els.panels.forEach((panel) => {
    panel.classList.toggle("is-active", panel.id === `tab-${tabName}`);
  });

  setPrompt(tabName);
}

function setupTabs() {
  els.tabs.forEach((tab) => {
    tab.addEventListener("click", () => {
      setActiveTab(tab.dataset.tab);
    });
  });
}

function setupNotes() {
  const saved = localStorage.getItem("math-for-ai-notes");
  if (saved) {
    els.notes.value = saved;
  }

  const save = () => {
    localStorage.setItem("math-for-ai-notes", els.notes.value);
    els.saveStatus.textContent = "ローカルに保存しました";
    window.clearTimeout(setupNotes.timer);
    setupNotes.timer = window.setTimeout(() => {
      els.saveStatus.textContent = "ローカル保存中";
    }, 1400);
  };

  els.notes.addEventListener("input", save);
  els.clearNotes.addEventListener("click", () => {
    els.notes.value = "";
    localStorage.removeItem("math-for-ai-notes");
    els.saveStatus.textContent = "メモを消去しました";
  });
}

function setupSignalPresets() {
  const presets = {
    sine: [0, 1, 0, -1, 0, 1, 0, -1],
    mix: [0, 1.7, 1, -0.7, 0, 0.7, -1, -1.7],
    pulse: [2, 0, 0, 0, 2, 0, 0, 0]
  };

  els.signal.presets.forEach((button) => {
    button.addEventListener("click", () => {
      const preset = presets[button.dataset.signal];
      if (!preset) {
        return;
      }
      els.signal.values.value = preset.join(", ");
      updateSignalLab();
    });
  });
}

function bindInputs() {
  [
    els.search.query,
    els.search.docs,
    els.recommend.target,
    els.recommend.peers,
    els.recommend.centered,
    els.vision.image,
    els.vision.kernel,
    els.vision.logits,
    els.vision.target,
    els.attention.q,
    els.attention.k,
    els.attention.v,
    els.attention.focus,
    els.signal.values
  ].forEach((element) => {
    element.addEventListener("input", renderAll);
    element.addEventListener("change", renderAll);
  });
}

function renderAll() {
  updateSearchLab();
  updateRecommendLab();
  updateVisionLab();
  updateAttentionLab();
  updateSignalLab();
}

setupTabs();
setupNotes();
setupSignalPresets();
bindInputs();
setActiveTab("search");
renderAll();
