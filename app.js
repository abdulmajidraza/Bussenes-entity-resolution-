document.addEventListener("DOMContentLoaded", () => {
    initTabs();
    loadStats();
    loadMatchingResults();
    loadEntities();
    initPairResolver();
    initSimulator();
    initValidatorModal();
});

function initTabs() {
    const btns = document.querySelectorAll(".tab-btn");
    btns.forEach(btn => {
        btn.addEventListener("click", () => {
            btns.forEach(b => b.classList.remove("active"));
            document.querySelectorAll(".tab-content").forEach(c => c.classList.remove("active"));

            btn.classList.add("active");
            const tabId = btn.getAttribute("data-tab");
            document.getElementById(tabId).classList.add("active");
        });
    });
}

async function loadStats() {
    try {
        const res = await fetch("/api/stats");
        if (!res.ok) return;
        const data = await res.json();

        document.getElementById("val-f05").textContent = data.validation_f05.toFixed(4);
        document.getElementById("val-precision").textContent = data.precision.toFixed(4);
        document.getElementById("val-recall").textContent = data.recall.toFixed(4);
        document.getElementById("val-blocking").textContent = data.blocking_recall.toFixed(4);
        document.getElementById("val-threshold").textContent = data.threshold.toFixed(4);

        document.getElementById("sim-threshold-slider").value = data.threshold;
        document.getElementById("sim-threshold-val").textContent = data.threshold.toFixed(4);
        document.getElementById("sim-matched-count").textContent = data.predicted_matched;
        document.getElementById("sim-singleton-count").textContent = data.predicted_singletons;
        document.getElementById("sim-total-count").textContent = data.test_s1_count;
    } catch (e) {
        console.error("Error loading stats:", e);
    }
}

async function loadMatchingResults() {
    try {
        const res = await fetch("/api/matching-results");
        if (!res.ok) return;
        const data = await res.json();

        const tbody = document.querySelector("#outputs-table tbody");
        tbody.innerHTML = "";

        const matches = data.matching_results || [];
        const candMap = {};
        (data.candidate_pairs || []).forEach(c => {
            candMap[c.source1_entity_id] = c.candidate_entity_ids;
        });

        matches.forEach(m => {
            const tr = document.createElement("tr");
            const sid = m.source1_entity_id;
            const matchIds = m.matched_entity_ids.trim();
            const candIds = candMap[sid] || "";

            const isSingleton = !matchIds;

            tr.innerHTML = `
                <td><strong>${sid}</strong></td>
                <td>${matchIds ? `<span class="chip" style="background:rgba(16,185,129,0.2); border-color:#10B981; color:#10B981;">${matchIds}</span>` : '<span class="text-muted">None (Singleton)</span>'}</td>
                <td><span style="font-family: monospace; font-size: 0.8rem;">${candIds || 'None'}</span></td>
                <td><span class="status-badge" style="${isSingleton ? 'background:rgba(245,158,11,0.15); border-color:#F59E0B; color:#F59E0B;' : ''}">${isSingleton ? 'Singleton' : 'Matched'}</span></td>
            `;
            tbody.appendChild(tr);
        });
    } catch (e) {
        console.error("Error loading matching results:", e);
    }
}

async function loadEntities() {
    try {
        const res = await fetch("/api/entities");
        if (!res.ok) return;
        const data = await res.json();

        const tbody = document.querySelector("#explorer-table tbody");
        tbody.innerHTML = "";

        const all = [
            ...(data.source1 || []),
            ...(data.source2 || []),
            ...(data.source3 || [])
        ];

        all.forEach(item => {
            const tr = document.createElement("tr");
            tr.innerHTML = `
                <td><strong>${item.entity_id}</strong></td>
                <td>${item.business_name}</td>
                <td>${item.business_address}</td>
                <td><span class="chip">${item.country}</span></td>
            `;
            tbody.appendChild(tr);
        });
    } catch (e) {
        console.error("Error loading entities:", e);
    }
}

function initPairResolver() {
    document.getElementById("template-us").addEventListener("click", () => {
        document.getElementById("name1").value = "Apex Technologies Corp";
        document.getElementById("addr1").value = "100 Industrial Pkwy Suite 400 Austin TX 78701";
        document.getElementById("c1").value = "United States";

        document.getElementById("name2").value = "APEX TECH CORPORATION INC";
        document.getElementById("addr2").value = "100 Industrial Parkway Ste 400, Austin TX 78701";
        document.getElementById("c2").value = "United States";
    });

    document.getElementById("template-in").addEventListener("click", () => {
        document.getElementById("name1").value = "Tata Consultancy Services Pvt. Ltd.";
        document.getElementById("addr1").value = "Nirmal Building 9th Flr Nariman Pt Mumbai 400021";
        document.getElementById("c1").value = "India";

        document.getElementById("name2").value = "TATA CONSULTANCY SERVICES LIMITED";
        document.getElementById("addr2").value = "Nirmal Bldg, Nariman Point, Mumbai 400021";
        document.getElementById("c2").value = "India";
    });

    document.getElementById("template-fr").addEventListener("click", () => {
        document.getElementById("name1").value = "Société Générale SA";
        document.getElementById("addr1").value = "29 Boulevard Haussmann, 75009 Paris";
        document.getElementById("c1").value = "France";

        document.getElementById("name2").value = "Societe Generale Paris";
        document.getElementById("addr2").value = "29 Bd Haussmann 75009 Paris";
        document.getElementById("c2").value = "France";
    });

    document.getElementById("btn-run-pair").addEventListener("click", async () => {
        const name1 = document.getElementById("name1").value;
        const addr1 = document.getElementById("addr1").value;
        const c1 = document.getElementById("c1").value;

        const name2 = document.getElementById("name2").value;
        const addr2 = document.getElementById("addr2").value;
        const c2 = document.getElementById("c2").value;

        try {
            const res = await fetch("/api/match-custom", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ name1, address1: addr1, country1: c1, name2, address2: addr2, country2: c2 })
            });

            if (!res.ok) return;
            const data = await res.json();
            displayPairResult(data);
        } catch (e) {
            console.error("Error matching pair:", e);
        }
    });
}

function displayPairResult(data) {
    const panel = document.getElementById("pair-result-panel");
    panel.classList.remove("hidden");

    const prob = data.probability;
    const threshold = data.threshold;
    const isMatch = data.is_match;

    document.getElementById("prob-display").textContent = prob.toFixed(4);

    const verdict = document.getElementById("match-verdict");
    verdict.textContent = isMatch ? "MATCH" : "NON-MATCH";
    verdict.className = `match-badge-large ${isMatch ? 'match' : 'non-match'}`;

    document.getElementById("meter-fill").style.width = `${(prob * 100).toFixed(1)}%`;
    document.getElementById("threshold-marker").style.left = `${(threshold * 100).toFixed(1)}%`;
    document.getElementById("threshold-label-text").textContent = `Threshold: ${threshold.toFixed(2)}`;

    const tbody = document.querySelector("#features-table tbody");
    tbody.innerHTML = "";

    const feats = data.features || {};
    Object.entries(feats).forEach(([fname, val]) => {
        const tr = document.createElement("tr");
        tr.innerHTML = `
            <td><code>${fname}</code></td>
            <td style="font-family: monospace; font-weight: 600; color:${val > 0.5 ? '#10B981' : '#94A3B8'}">${typeof val === 'number' ? val.toFixed(4) : val}</td>
            <td style="color:#94A3B8; font-size:0.8rem;">${getFeatureDesc(fname)}</td>
        `;
        tbody.appendChild(tr);
    });
}

function getFeatureDesc(fname) {
    const map = {
        "name_ratio": "Levenshtein distance similarity ratio on normalized name",
        "name_wratio": "RapidFuzz weighted string similarity ratio",
        "name_token_sort": "Token sort ratio (robust to word order variations)",
        "name_token_set": "Token set ratio (robust to extra words / legal suffixes)",
        "name_jaccard": "Jaccard token intersection over union similarity",
        "name_exact": "1.0 if normalized names are identical, 0.0 otherwise",
        "name_len_diff": "Absolute difference in character lengths",
        "name_token_count_diff": "Absolute difference in word counts",
        "addr_ratio": "Address Levenshtein distance similarity ratio",
        "addr_wratio": "RapidFuzz address weighted similarity ratio",
        "addr_token_sort": "Address token sort ratio",
        "addr_token_set": "Address token set ratio",
        "addr_jaccard": "Address token Jaccard similarity",
        "addr_exact": "1.0 if normalized addresses are identical, 0.0 otherwise",
        "addr_numeric_overlap": "Count of shared digits / house / postal code tokens",
        "country_exact": "1.0 if normalized countries match, 0.0 otherwise",
        "combined_sim_score": "Composite weighted similarity metric"
    };
    return map[fname] || "Tabular similarity feature";
}

function initSimulator() {
    const slider = document.getElementById("sim-threshold-slider");
    slider.addEventListener("input", async () => {
        const val = parseFloat(slider.value);
        document.getElementById("sim-threshold-val").textContent = val.toFixed(4);

        try {
            const res = await fetch("/api/adjust-threshold", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ threshold: val })
            });

            if (!res.ok) return;
            const data = await res.json();

            document.getElementById("sim-matched-count").textContent = data.predicted_matched;
            document.getElementById("sim-singleton-count").textContent = data.predicted_singletons;
        } catch (e) {
            console.error("Error adjusting threshold:", e);
        }
    });
}

function initValidatorModal() {
    const modal = document.getElementById("validator-modal");
    const btn = document.getElementById("btn-run-validator");
    const closeBtn = document.getElementById("modal-close");

    btn.addEventListener("click", async () => {
        modal.classList.remove("hidden");
        const consoleEl = document.getElementById("validator-console");
        consoleEl.textContent = "Executing official validator against test dataset...";

        try {
            const res = await fetch("/api/validate");
            if (!res.ok) return;
            const data = await res.json();
            consoleEl.textContent = data.output || "Validation finished.";
        } catch (e) {
            consoleEl.textContent = "Error running validator: " + e;
        }
    });

    closeBtn.addEventListener("click", () => {
        modal.classList.add("hidden");
    });
}
