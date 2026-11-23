import { useState, useEffect } from "react";
import { api } from "../api.js";
import { Card, Button, Spinner, Alert, Stat, Highlight, Priority } from "../components.jsx";

// Helper component for "Show code" toggle
function CodeToggle({ name, label = "Show code" }) {
  const [open, setOpen] = useState(false);
  const [code, setCode] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const toggle = async () => {
    if (open) { setOpen(false); return; }
    setOpen(true);
    if (!code && !loading) {
      setLoading(true); setError(null);
      try {
        const res = await api.source(name);
        setCode(res.source);
      } catch (e) {
        setError(e.message);
      } finally {
        setLoading(false);
      }
    }
  };

  return (
    <div style={{ marginTop: "12px" }}>
      <Button variant="ghost" onClick={toggle}>
        {open ? `Hide code (${name})` : `${label} (${name})`}
      </Button>
      {open && (
        <div className="code-box">
          {loading && <Spinner small label="Loading code" />}
          {error && <div style={{ color: "#F87171" }}>Failed to load code: {error}</div>}
          {code && <pre><code>{code}</code></pre>}
        </div>
      )}
    </div>
  );
}

const pct = (n) => `${(n * 100).toFixed(1)}%`;

export default function Walkthrough() {
  const [m, setM] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  // Step 2: Extraction sample
  const [samples, setSamples] = useState([]);
  const [selectedSample, setSelectedSample] = useState("");
  const [extractedSample, setExtractedSample] = useState(null);
  const [loadingSample, setLoadingSample] = useState(false);

  // Step 3: Cleaning interactive
  const [cleanInput, setCleanInput] = useState("There is no hot water in Cedar Hall and requests were ignored!");
  const [cleanResult, setCleanResult] = useState(null);
  const [loadingClean, setLoadingClean] = useState(false);

  // Step 5: Active tab for evaluation
  const [evalTab, setEvalTab] = useState("category");

  // Step 6: Sentiment self-check
  const [quizItems, setQuizItems] = useState([]);
  const [userChoices, setUserChoices] = useState({});
  const [loadingQuiz, setLoadingQuiz] = useState(false);

  // Step 7: Entity interactive
  const [entityInput, setEntityInput] = useState("I have CHEM 102 in Cedar Hall at 9:30 AM this Friday.");
  const [entityResult, setEntityResult] = useState(null);

  // Step 8: Topic modelling k
  const [topicK, setTopicK] = useState("7");

  // Step 9: Multilingual interactive
  const [selectedLangExample, setSelectedLangExample] = useState(0);
  const [multiResult, setMultiResult] = useState(null);
  const [loadingMulti, setLoadingMulti] = useState(false);

  // Step 10: Prompt comparison
  const [compareData, setCompareData] = useState(null);
  const [loadingCompare, setLoadingCompare] = useState(false);
  const [compareError, setCompareError] = useState(null);

  // Load metrics & initial data on mount
  useEffect(() => {
    Promise.all([api.metrics(), api.samples()])
      .then(([metricsData, samplesData]) => {
        setM(metricsData);
        setSamples(samplesData);
        if (samplesData.length > 0) {
          const first = samplesData[0].name;
          setSelectedSample(first);
          api.sample(first).then(setExtractedSample).catch(() => {});
        }
      })
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
  }, []);

  // Run initial cleaning demo
  useEffect(() => {
    if (cleanInput) {
      api.steps(cleanInput).then((s) => {
        setCleanResult(s.clean_steps);
      }).catch(() => {});
    }
  }, []);

  // Run initial entity demo
  useEffect(() => {
    if (entityInput) {
      api.steps(entityInput).then((s) => {
        setEntityResult(s);
      }).catch(() => {});
    }
  }, []);

  // Load Step 6 sentiment check items
  const loadQuiz = () => {
    setLoadingQuiz(true);
    api.sentimentCheck(20)
      .then((items) => {
        setQuizItems(items);
        setUserChoices({});
      })
      .catch((e) => setError(e.message))
      .finally(() => setLoadingQuiz(false));
  };

  // Step 2 sample selection
  const handleSampleChange = (name) => {
    setSelectedSample(name);
    setLoadingSample(true);
    api.sample(name)
      .then(setExtractedSample)
      .catch((e) => setError(e.message))
      .finally(() => setLoadingSample(false));
  };

  // Step 3 cleaning submit
  const handleClean = () => {
    if (!cleanInput.trim()) return;
    setLoadingClean(true);
    api.steps(cleanInput)
      .then((s) => setCleanResult(s.clean_steps))
      .catch((e) => setError(e.message))
      .finally(() => setLoadingClean(false));
  };

  // Step 7 entity submit
  const handleEntityCheck = () => {
    if (!entityInput.trim()) return;
    api.steps(entityInput).then((s) => setEntityResult(s)).catch(() => {});
  };

  // Step 9: Hindi/Marathi sample pool
  const MULTI_SAMPLES = [
    { lang: "Hindi", text: "मेरे हॉस्टल में तीन दिन से गर्म पानी नहीं आ रहा है।", ref: "There has been no hot water in my hostel for three days." },
    { lang: "Hindi", text: "कॉलेज का वाई-फाई बहुत धीमा है और बार-बार बंद हो जाता है।", ref: "The college Wi-Fi is very slow and keeps disconnecting." },
    { lang: "Hindi", text: "मेरी स्कॉलरशिप की राशि अभी तक खाते में नहीं आई है।", ref: "My scholarship amount has not yet arrived in my account." },
    { lang: "Marathi", text: "माझ्या वसतिगृहात तीन दिवसांपासून गरम पाणी येत नाही.", ref: "There has been no hot water in my hostel for three days." },
    { lang: "Marathi", text: "महाविद्यालयाचे वाय-फाय खूप हळू आहे आणि सतत बंद होते.", ref: "The college Wi-Fi is very slow and keeps disconnecting." },
    { lang: "Marathi", text: "माझ्या शिष्यवृत्तीची रक्कम अजून खात्यात जमा झाली नाही.", ref: "My scholarship amount has not yet been credited to my account." },
  ];

  const handleTestMultilingual = (idx) => {
    setSelectedLangExample(idx);
    setLoadingMulti(true);
    setMultiResult(null);
    api.steps(MULTI_SAMPLES[idx].text)
      .then(setMultiResult)
      .catch((e) => setError(e.message))
      .finally(() => setLoadingMulti(false));
  };

  // Step 10: Run prompt comparison
  const handlePromptCompare = () => {
    setLoadingCompare(true);
    setCompareError(null);
    api.promptCompare()
      .then(setCompareData)
      .catch((e) => setCompareError(e.message))
      .finally(() => setLoadingCompare(false));
  };

  if (loading) return <div className="center"><Spinner label="Loading metrics" /><p className="muted">Loading walkthrough evidence...</p></div>;
  if (error && !m) return <Alert>{error}</Alert>;

  // Compute sentiment quiz agreement
  const quizAnswered = Object.keys(userChoices).length;
  const quizMatches = quizItems.filter((it, idx) => userChoices[idx] === it.vader_label).length;
  const agreementRate = quizAnswered > 0 ? Math.round((quizMatches / quizAnswered) * 100) : null;

  return (
    <div className="stack">
      <div>
        <h1>Technical Walkthrough</h1>
        <p className="muted">All models and metrics were produced in the notebook NLP_Pipeline.ipynb and exported to this app.</p>
      </div>

      {/* OVERVIEW */}
      <Card title="Project Overview" subtitle="ResolveAI · Student Grievance Classification & Priority Prediction">
        <div className="grid-4">
          <Stat label="Team Members" value="Vedant & Krishna" note="Asawale V. M. · Kushwah K. R." />
          <Stat label="Dataset Origin" value="SCOPE Dataset" note="Open-source university complaints" />
          <Stat label="Dataset Size" value={`${m.dataset.rows} rows`} note="3,000 grievance texts" />
          <Stat label="Stratified Split" value="70 / 15 / 15" note={`${m.dataset.train} train · ${m.dataset.val} val · ${m.dataset.test} test`} />
        </div>
        <div style={{ marginTop: "14px" }} className="panel">
          <p className="small">
            <strong>Honest limitations:</strong> Grievance texts in the SCOPE dataset look machine-generated, so real student phrasing may be messier.
            Urgency classification achieves ~59% test accuracy because urgency labels are inherently subjective (baseline guessing: {pct(m.priority.baseline_acc)}).
            Models are trained on English text; non-English grievances are translated prior to classification.
          </p>
        </div>
      </Card>

      {/* STEP 1: DATASET */}
      <Card className="step-card" title={<span><span className="step-num">1</span> Dataset & Statistics</span>}>
        <p className="step-desc">We use an open-source dataset of 3,000 university grievances across 7 categories and 4 priority levels from the SCOPE project.</p>
        <div className="grid-4">
          <Stat label="Total Records" value={m.dataset.rows} />
          <Stat label="Average Length" value={`${m.dataset.avg_words} words`} note="Per complaint text" />
          <Stat label="Raw Vocabulary" value={m.dataset.vocab_raw} note="Unique tokens before cleaning" />
          <Stat label="Clean Vocabulary" value={m.dataset.vocab_clean} note="Tokens after preprocessing" />
        </div>

        <div className="grid-2" style={{ marginTop: "14px" }}>
          <div className="panel">
            <h3>Category Distribution</h3>
            <table>
              <thead><tr><th>Category</th><th className="right">Count</th><th className="right">Share</th></tr></thead>
              <tbody>
                {Object.entries(m.dataset.category_counts).map(([cat, cnt]) => (
                  <tr key={cat}>
                    <td>{cat}</td>
                    <td className="right">{cnt}</td>
                    <td className="right">{pct(cnt / m.dataset.rows)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <div className="panel">
            <h3>Priority Distribution</h3>
            <table>
              <thead><tr><th>Priority</th><th className="right">Count</th><th className="right">Share</th></tr></thead>
              <tbody>
                {Object.entries(m.dataset.priority_counts).map(([pri, cnt]) => (
                  <tr key={pri}>
                    <td><Priority value={pri} /></td>
                    <td className="right">{cnt}</td>
                    <td className="right">{pct(cnt / m.dataset.rows)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </Card>

      {/* STEP 2: TEXT EXTRACTION */}
      <Card className="step-card" title={<span><span className="step-num">2</span> Text Extraction</span>}>
        <p className="step-desc">Extracts plain grievance text from uploaded TXT and PDF letters by isolating the grievance body after formal headers.</p>
        <div className="row">
          <label htmlFor="sample-select" style={{ fontWeight: 600, fontSize: "0.9rem" }}>Select a sample letter:</label>
          <select id="sample-select" value={selectedSample} onChange={(e) => handleSampleChange(e.target.value)}>
            {samples.map((s) => (
              <option key={s.name} value={s.name}>{s.label}</option>
            ))}
          </select>
        </div>
        {loadingSample && <div className="center"><Spinner small /></div>}
        {extractedSample && !loadingSample && (
          <div className="grid-2" style={{ marginTop: "14px" }}>
            <div className="panel">
              <h4>Sample Document ({extractedSample.name})</h4>
              <p className="small muted">Extracted format: {extractedSample.kind.toUpperCase()}</p>
            </div>
            <div className="panel">
              <h4>Isolated Grievance Body</h4>
              <p style={{ marginTop: "6px", fontStyle: "italic" }}>"{extractedSample.text || (extractedSample.texts && extractedSample.texts[0])}"</p>
            </div>
          </div>
        )}
        <CodeToggle name="extract_file" />
      </Card>

      {/* STEP 3: TEXT PREPROCESSING */}
      <Card className="step-card" title={<span><span className="step-num">3</span> Text Preprocessing</span>}>
        <p className="step-desc">Cleans text through lowercase conversion, punctuation removal, stopword filtering with negation preservation ('not', 'no', 'never'), and light suffix stemming.</p>
        <div>
          <label className="sr" htmlFor="clean-input">Text to clean</label>
          <input
            id="clean-input"
            type="text"
            value={cleanInput}
            onChange={(e) => setCleanInput(e.target.value)}
            style={{ width: "100%", padding: "8px 12px", borderRadius: "8px" }}
            aria-label="Text to clean"
          />
          <div style={{ marginTop: "8px" }}>
            <Button variant="secondary" loading={loadingClean} onClick={handleClean}>Trace Cleaning Steps</Button>
          </div>
        </div>
        {cleanResult && (
          <div style={{ marginTop: "14px" }} className="stack">
            <div className="pipeline-step">
              <h4>1. Lowercase</h4>
              <p className="small">{cleanResult.lowercase}</p>
            </div>
            <div className="pipeline-step">
              <h4>2. Remove Punctuation</h4>
              <p className="small">{cleanResult.no_punctuation}</p>
            </div>
            <div className="pipeline-step">
              <h4>3. Tokenization</h4>
              <p className="small">[{cleanResult.tokens.join(", ")}] ({cleanResult.tokens.length} tokens)</p>
            </div>
            <div className="pipeline-step">
              <h4>4. Remove Stopwords (Negations Preserved)</h4>
              <p className="small">[{cleanResult.no_stopwords.join(", ")}] ({cleanResult.no_stopwords.length} tokens retained)</p>
            </div>
            <div className="pipeline-step">
              <h4>5. Suffix Stemming & Final Text</h4>
              <p className="small" style={{ fontWeight: 600, color: "var(--brand-ink)" }}>{cleanResult.cleaned}</p>
            </div>
          </div>
        )}
        <CodeToggle name="clean_steps" />
      </Card>

      {/* STEP 4: TF-IDF VECTORIZATION */}
      <Card className="step-card" title={<span><span className="step-num">4</span> TF-IDF Feature Representation</span>}>
        <p className="step-desc">Term Frequency-Inverse Document Frequency (TF-IDF) converts text into numerical features using 1-2 word n-grams and sublinear scaling.</p>
        <div className="grid-3">
          <Stat label="Total N-gram Features" value={m.dataset.tfidf_features} note="Unigrams + bigrams (min_df=2)" />
          <Stat label="Sublinear TF" value="Enabled" note="Replaces TF with 1 + log(TF)" />
          <Stat label="Categories Covered" value={Object.keys(m.dataset.top_words).length} note="Distinct administrative units" />
        </div>
        <div style={{ marginTop: "14px" }}>
          <h3>Top Indicative TF-IDF Terms Per Category</h3>
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>Category</th>
                  <th>Top 10 Keywords (Highest Mean TF-IDF)</th>
                </tr>
              </thead>
              <tbody>
                {Object.entries(m.dataset.top_words).map(([cat, words]) => (
                  <tr key={cat}>
                    <td style={{ fontWeight: 600 }}>{cat}</td>
                    <td>
                      <div style={{ display: "flex", gap: "6px", flexWrap: "wrap" }}>
                        {words.map((w) => (
                          <span key={w} style={{ background: "var(--bg)", border: "1px solid var(--line)", padding: "2px 8px", borderRadius: "4px", fontSize: "0.82rem" }}>
                            {w}
                          </span>
                        ))}
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
        <p className="small muted" style={{ marginTop: "12px" }}>Code and outputs: NLP_Pipeline.ipynb, section 5</p>
      </Card>

      {/* STEP 5: CLASSIFICATION & EVALUATION */}
      <Card className="step-card" title={<span><span className="step-num">5</span> Classification & Evaluation</span>}>
        <p className="step-desc">Compares Naive Bayes, Logistic Regression, and Linear SVM on validation F1 score, selecting the best model for category and priority separately.</p>

        <div style={{ display: "flex", gap: "10px", margin: "10px 0 16px" }}>
          <Button variant={evalTab === "category" ? "primary" : "secondary"} onClick={() => setEvalTab("category")}>
            Category Classification
          </Button>
          <Button variant={evalTab === "urgency" ? "primary" : "secondary"} onClick={() => setEvalTab("urgency")}>
            Priority / Urgency Classification
          </Button>
        </div>

        {(() => {
          const target = evalTab === "category" ? m.category : m.priority;
          const targetName = evalTab === "category" ? "Category" : "Priority";
          return (
            <div className="stack">
              {/* Candidate Comparison Table */}
              <div className="panel">
                <h3>Model Comparison (Validation Set)</h3>
                <table>
                  <thead>
                    <tr>
                      <th>Model</th>
                      <th className="right">Train Accuracy</th>
                      <th className="right">Val Accuracy</th>
                      <th className="right">Val Weighted F1</th>
                      <th>Selected</th>
                    </tr>
                  </thead>
                  <tbody>
                    {Object.entries(target.models).map(([name, scores]) => (
                      <tr key={name} style={name === target.best ? { background: "var(--brand-soft)", fontWeight: 600 } : {}}>
                        <td>{name}</td>
                        <td className="right">{pct(scores.train_acc)}</td>
                        <td className="right">{pct(scores.val_acc)}</td>
                        <td className="right">{scores.val_f1.toFixed(3)}</td>
                        <td>{name === target.best ? "Best Model" : ""}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>

              {/* Overfitting Demo */}
              <div className="panel">
                <h3>Regularisation & Overfitting Demo (Logistic Regression)</h3>
                <p className="small muted">High C (100) under-regularises and overfits (100% train accuracy). Low C (1.0 - 0.1) regularises weights to generalize better.</p>
                <div className="bar-chart">
                  {target.overfit_demo.map((d) => (
                    <div key={d.C} className="bar-row">
                      <div className="bar-label">C = {d.C}</div>
                      <div className="bar-track" title={`Train: ${pct(d.train_acc)} | Val: ${pct(d.val_acc)}`}>
                        <div className="bar-fill bar-train" style={{ width: `${d.train_acc * 50}%` }}>
                          Train {pct(d.train_acc)}
                        </div>
                        <div className="bar-fill bar-val" style={{ width: `${d.val_acc * 50}%` }}>
                          Val {pct(d.val_acc)}
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              {/* Test Set Results */}
              <div className="grid-4">
                <Stat label="Selected Best Model" value={target.best} note="Picked by validation F1" />
                <Stat label="Test Accuracy" value={pct(target.test.accuracy)} note="Evaluated on 450 unseen test rows" />
                <Stat label="Test Weighted F1" value={target.test.f1.toFixed(3)} note="Harmonic mean of precision & recall" />
                <Stat label="Majority Baseline" value={pct(target.baseline_acc)} note="Accuracy of guessing most frequent class" />
              </div>

              <div className="panel">
                <h4>Metric Explanations</h4>
                <p className="small muted" style={{ lineHeight: 1.6 }}>
                  &bull; <strong>Accuracy:</strong> Fraction of all predictions that were exactly right ({pct(target.test.accuracy)}).<br />
                  &bull; <strong>Precision:</strong> When the model predicts a class, how often it was correct ({pct(target.test.precision)}).<br />
                  &bull; <strong>Recall:</strong> What fraction of actual instances of a class were successfully detected ({pct(target.test.recall)}).<br />
                  &bull; <strong>F1 Score:</strong> Balance of precision and recall ({target.test.f1.toFixed(3)}).<br />
                  &bull; <strong>Baseline:</strong> Accuracy if a simple dummy model always guessed the majority class ({pct(target.baseline_acc)}).
                </p>
              </div>

              {/* Confusion Matrix Heatmap */}
              <div className="panel">
                <h3>Test Set Confusion Matrix ({targetName})</h3>
                <p className="small muted">Rows = Actual Class &middot; Columns = Predicted Class</p>
                <div className="table-wrap" style={{ marginTop: "10px" }}>
                  <table className="cm-table">
                    <thead>
                      <tr>
                        <th>Actual \ Predicted</th>
                        {target.labels.map((lab) => <th key={lab}>{lab}</th>)}
                      </tr>
                    </thead>
                    <tbody>
                      {target.labels.map((actualLab, rIdx) => (
                        <tr key={actualLab}>
                          <th style={{ textAlign: "left" }}>{actualLab}</th>
                          {target.cm[rIdx].map((cnt, cIdx) => {
                            const isDiagonal = rIdx === cIdx;
                            const bg = isDiagonal
                              ? `rgba(79, 70, 229, ${Math.min(0.85, Math.max(0.12, cnt / 60))})`
                              : cnt > 0
                              ? `rgba(220, 38, 38, ${Math.min(0.6, cnt / 15)})`
                              : "transparent";
                            const color = (isDiagonal && cnt > 20) || (!isDiagonal && cnt > 5) ? "#fff" : "inherit";
                            return (
                              <td key={cIdx} className="cm-cell" style={{ background: bg, color }}>
                                {cnt}
                              </td>
                            );
                          })}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
          );
        })()}
        <p className="small muted" style={{ marginTop: "12px" }}>Code and outputs: NLP_Pipeline.ipynb, section 6</p>
        <div style={{ display: "flex", gap: "10px", marginTop: "10px" }}>
          <CodeToggle name="classify" label="Show prediction code" />
          <CodeToggle name="PortableModel" label="Show model class" />
        </div>
      </Card>

      {/* STEP 6: SENTIMENT ANALYSIS */}
      <Card className="step-card" title={<span><span className="step-num">6</span> Sentiment Analysis (VADER)</span>}>
        <p className="step-desc">VADER (Valence Aware Dictionary and sEntiment Reasoner) scores emotional tone, with 'please' set to neutral to avoid confusing politeness with positivity.</p>
        <div className="grid-2">
          <div className="panel">
            <h3>Average Sentiment Score by Priority</h3>
            <table>
              <thead><tr><th>Priority</th><th className="right">Avg VADER Compound Score</th></tr></thead>
              <tbody>
                {Object.entries(m.sentiment_by_priority).map(([pri, sc]) => (
                  <tr key={pri}>
                    <td><Priority value={pri} /></td>
                    <td className="right" style={{ fontWeight: 600, color: sc < 0 ? "var(--c-critical)" : "inherit" }}>
                      {sc > 0 ? "+" : ""}{sc}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
            <p className="small muted" style={{ marginTop: "8px" }}>Critical grievances are noticeably more negative than Low-urgency ones.</p>
          </div>
          <div className="panel">
            <h3>Sentiment Distribution Across Dataset</h3>
            <table>
              <thead><tr><th>Sentiment Label</th><th className="right">Share</th></tr></thead>
              <tbody>
                {Object.entries(m.sentiment_share).map(([lbl, val]) => (
                  <tr key={lbl}>
                    <td>{lbl}</td>
                    <td className="right">{pct(val)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        {/* 20-item Interactive Self-Check */}
        <div className="panel" style={{ marginTop: "14px" }}>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", flexWrap: "wrap", gap: "10px" }}>
            <div>
              <h3>20-Item Sentiment Agreement Self-Check</h3>
              <p className="small muted">Label these 20 random held-out grievances yourself to measure agreement with VADER in your browser.</p>
            </div>
            <Button variant="secondary" loading={loadingQuiz} onClick={loadQuiz}>
              {quizItems.length > 0 ? "Reload 20 Grievances" : "Start 20-Item Self-Check"}
            </Button>
          </div>

          {agreementRate !== null && (
            <div style={{ margin: "12px 0" }}>
              <Stat label="Human-VADER Agreement" value={`${agreementRate}%`} note={`${quizMatches} of ${quizAnswered} questions agreed`} />
            </div>
          )}

          {quizItems.length > 0 && (
            <div className="stack" style={{ marginTop: "12px", maxHeight: "400px", overflowY: "auto" }}>
              {quizItems.map((item, idx) => (
                <div key={idx} className="quiz-item">
                  <p className="small" style={{ fontWeight: 500 }}>{idx + 1}. "{item.text}"</p>
                  <div className="quiz-options">
                    {["Negative", "Neutral", "Positive"].map((opt) => (
                      <label key={opt}>
                        <input
                          type="radio"
                          name={`quiz-${idx}`}
                          checked={userChoices[idx] === opt}
                          onChange={() => setUserChoices({ ...userChoices, [idx]: opt })}
                        />
                        {opt}
                      </label>
                    ))}
                    {userChoices[idx] && (
                      <span className="small muted" style={{ marginLeft: "auto" }}>
                        VADER: <strong>{item.vader_label}</strong> ({item.vader_score}) {userChoices[idx] === item.vader_label ? "✓" : "≠"}
                      </span>
                    )}
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
        <p className="small muted" style={{ marginTop: "12px" }}>Code and outputs: NLP_Pipeline.ipynb, section 7</p>
        <CodeToggle name="sentiment" />
      </Card>

      {/* STEP 7: ENTITY EXTRACTION */}
      <Card className="step-card" title={<span><span className="step-num">7</span> Entity Extraction (NER)</span>}>
        <p className="step-desc">Custom domain rules on a blank spaCy pipeline extract actionable entities: course codes, campus buildings, specific times, and days of the week.</p>
        <div>
          <label className="sr" htmlFor="entity-input">Text with entities</label>
          <input
            id="entity-input"
            type="text"
            value={entityInput}
            onChange={(e) => setEntityInput(e.target.value)}
            style={{ width: "100%", padding: "8px 12px", borderRadius: "8px" }}
            aria-label="Text with entities"
          />
          <div style={{ marginTop: "8px" }}>
            <Button variant="secondary" onClick={handleEntityCheck}>Extract Entities</Button>
          </div>
        </div>

        {entityResult && (
          <div className="panel" style={{ marginTop: "14px" }}>
            <h4>Live Extraction Output</h4>
            <Highlight text={entityResult.english_text} entities={entityResult.entities} />
            <div style={{ marginTop: "8px" }}>
              {entityResult.entities.length === 0 ? (
                <p className="small muted">No entities matched.</p>
              ) : (
                <div style={{ display: "flex", gap: "8px", flexWrap: "wrap" }}>
                  {entityResult.entities.map((e, idx) => (
                    <span key={idx} style={{ background: "var(--brand-soft)", color: "var(--brand-ink)", padding: "2px 8px", borderRadius: "4px", fontSize: "0.82rem" }}>
                      <strong>{e.label}</strong>: {e.text}
                    </span>
                  ))}
                </div>
              )}
            </div>
          </div>
        )}

        <div className="panel" style={{ marginTop: "14px" }}>
          <h3>Entity Frequency Across All 3,000 Complaints</h3>
          <p className="small muted">Rule-based spaCy EntityRuler engine: {m.ner_engine}</p>
          <div style={{ display: "flex", gap: "12px", flexWrap: "wrap", marginTop: "10px" }}>
            {Object.entries(m.ner_counts).map(([label, cnt]) => (
              <Stat key={label} label={label} value={cnt} note="matches found" />
            ))}
          </div>
        </div>
        <p className="small muted" style={{ marginTop: "12px" }}>Code and outputs: NLP_Pipeline.ipynb, section 8</p>
        <CodeToggle name="extract_entities" />
      </Card>

      {/* STEP 8: TOPIC MODELLING */}
      <Card className="step-card" title={<span><span className="step-num">8</span> Unsupervised Topic Modelling (LDA)</span>}>
        <p className="step-desc">Latent Dirichlet Allocation (LDA) discovers latent themes across unlabelled complaints, revealing natural clusters without human supervision.</p>
        <div className="row">
          <label htmlFor="topic-k" style={{ fontWeight: 600, fontSize: "0.9rem" }}>Select Topic Granularity (k):</label>
          <select id="topic-k" value={topicK} onChange={(e) => setTopicK(e.target.value)}>
            <option value="5">k = 5 topics (high-level themes)</option>
            <option value="7">k = 7 topics (aligned with categories)</option>
            <option value="10">k = 10 topics (fine-grained themes)</option>
          </select>
          <span className="muted small">Perplexity: {m.topics[topicK]?.perplexity}</span>
        </div>

        {m.topics[topicK] && (
          <div className="stack" style={{ marginTop: "14px" }}>
            <div className="panel">
              <h3>Discovered Topics (Top 8 Keywords each)</h3>
              <div className="grid-2">
                {Object.entries(m.topics[topicK].topics).map(([topName, words]) => (
                  <div key={topName} style={{ marginBottom: "8px" }}>
                    <div style={{ fontWeight: 600, fontSize: "0.9rem", color: "var(--brand-ink)" }}>{topName}</div>
                    <div className="small muted">{words.join(", ")}</div>
                  </div>
                ))}
              </div>
            </div>

            <div className="panel">
              <h3>Topic vs Category Cross-Tabulation</h3>
              <p className="small muted">Shows how unsupervised LDA topics naturally group inside administrative categories.</p>
              <div className="table-wrap" style={{ marginTop: "8px" }}>
                <table>
                  <thead>
                    <tr>
                      <th>Topic</th>
                      {m.topics[topicK].crosstab.columns.map((col) => (
                        <th key={col} className="right" style={{ fontSize: "0.72rem" }}>{col}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {m.topics[topicK].crosstab.index.map((topName, rIdx) => (
                      <tr key={topName}>
                        <td style={{ fontWeight: 600 }}>{topName}</td>
                        {m.topics[topicK].crosstab.values[rIdx].map((val, cIdx) => (
                          <td key={cIdx} className="right" style={val > 50 ? { background: "var(--brand-soft)", fontWeight: 600 } : {}}>
                            {val}
                          </td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        )}
        <p className="small muted" style={{ marginTop: "12px" }}>Code and outputs: NLP_Pipeline.ipynb, section 9</p>
      </Card>

      {/* STEP 9: MULTILINGUAL SUPPORT */}
      <Card className="step-card" title={<span><span className="step-num">9</span> Multilingual Language Detection & Translation</span>}>
        <p className="step-desc">Detects non-English input using langdetect and translates Hindi or Marathi to English through a resilient chain (Gemini &rarr; Google Translate &rarr; MyMemory).</p>
        <div className="row">
          <label htmlFor="multi-select" style={{ fontWeight: 600, fontSize: "0.9rem" }}>Pick a regional grievance:</label>
          <select id="multi-select" value={selectedLangExample} onChange={(e) => handleTestMultilingual(Number(e.target.value))}>
            {MULTI_SAMPLES.map((s, idx) => (
              <option key={idx} value={idx}>[{s.lang}] {s.text.slice(0, 45)}...</option>
            ))}
          </select>
        </div>

        <div className="panel" style={{ marginTop: "14px" }}>
          <h4>Original Regional Text</h4>
          <p style={{ fontSize: "1.05rem", margin: "4px 0 8px" }}>{MULTI_SAMPLES[selectedLangExample].text}</p>
          <p className="small muted">Reference meaning: "{MULTI_SAMPLES[selectedLangExample].ref}"</p>
        </div>

        {loadingMulti && <div className="center"><Spinner small /><p className="small muted">Translating and classifying...</p></div>}
        {multiResult && !loadingMulti && (
          <div className="panel" style={{ marginTop: "14px" }}>
            <h4>Pipeline Results on Translated Text</h4>
            <div className="grid-4" style={{ marginTop: "8px" }}>
              <Stat label="Language Detected" value={multiResult.language.toUpperCase()} note={`Translated via ${multiResult.translated_by || "direct"}`} />
              <Stat label="Predicted Category" value={multiResult.category} note={`Confidence ${Math.round(multiResult.confidence * 100)}%`} />
              <Stat label="Predicted Priority" value={<Priority value={multiResult.priority} />} />
              <Stat label="Sentiment Score" value={`${multiResult.sentiment_score > 0 ? "+" : ""}${multiResult.sentiment_score}`} note={multiResult.sentiment} />
            </div>
            <div style={{ marginTop: "10px" }}>
              <strong>English Translation:</strong>
              <p style={{ fontStyle: "italic", marginTop: "4px" }}>"{multiResult.english_text}"</p>
            </div>
          </div>
        )}
        <p className="small muted" style={{ marginTop: "12px" }}>Code and outputs: NLP_Pipeline.ipynb, section 10</p>
        <CodeToggle name="to_english" />
      </Card>

      {/* STEP 10: GENERATIVE AI RESPONSE */}
      <Card className="step-card" title={<span><span className="step-num">10</span> Generative AI (Prompt Engineering)</span>}>
        <p className="step-desc">Drafts a polite administrative response using Gemini with a structured prompt, grounded in our model's classification to prevent hallucinations.</p>
        <div className="grid-2">
          <div className="prompt-box">
            <h4>Prompt V1 (Unstructured Baseline)</h4>
            <pre className="small">Write a reply to this student grievance: "&#123;text&#125;"</pre>
            <p className="small muted" style={{ marginTop: "8px" }}>Issues: Unconstrained tone, produces hallucinated deadlines or promises, unstructured raw text.</p>
          </div>
          <div className="prompt-box">
            <h4>Prompt V2 (Role-Grounded & Strict JSON Schema)</h4>
            <pre className="small" style={{ fontSize: "0.78rem" }}>
{`You are a university grievance-redressal assistant.
Student grievance: """{text}"""
Our system's analysis -> category: {category}; priority: {priority}; sentiment: {sentiment}; entities: {entities}.
Return ONLY valid JSON with exactly these keys:
{"summary": "max 20 words", "department": "office", "action": "one concrete step", "reply": "3 sentences"}
Rules: empathetic; no invented facts; mention response time '{sla}'.`}
            </pre>
            <p className="small muted" style={{ marginTop: "8px" }}>Grounded in trained classification models; strict schema; SLA-aware.</p>
          </div>
        </div>

        <div style={{ marginTop: "14px" }}>
          <Button variant="secondary" loading={loadingCompare} onClick={handlePromptCompare}>
            Run Live Comparison on 3 Held-Out Complaints
          </Button>
          {compareError && <Alert>{compareError}</Alert>}
        </div>

        {compareData && (
          <div className="stack" style={{ marginTop: "14px" }}>
            {compareData.map((res, i) => (
              <div key={i} className="panel">
                <h4>Held-Out Grievance #{i + 1}</h4>
                <p style={{ fontStyle: "italic", marginBottom: "8px" }}>"{res.text}"</p>
                <div className="grid-2">
                  <div style={{ background: "var(--surface)", padding: "10px", borderRadius: "8px", border: "1px solid var(--line)" }}>
                    <div style={{ fontWeight: 600, fontSize: "0.85rem", marginBottom: "4px" }}>Prompt V1 Output</div>
                    <p className="small" style={{ whiteSpace: "pre-wrap" }}>{res.response_v1}</p>
                  </div>
                  <div style={{ background: "var(--surface)", padding: "10px", borderRadius: "8px", border: "1px solid var(--line)" }}>
                    <div style={{ display: "flex", justifyContent: "space-between", marginBottom: "4px" }}>
                      <span style={{ fontWeight: 600, fontSize: "0.85rem" }}>Prompt V2 Output</span>
                      <span className="small" style={{ color: res.v2_valid_json ? "var(--c-low)" : "var(--c-critical)", fontWeight: 700 }}>
                        {res.v2_valid_json ? "Valid JSON ✓" : "Invalid JSON"}
                      </span>
                    </div>
                    <pre className="small" style={{ margin: 0, whiteSpace: "pre-wrap" }}>{res.response_v2}</pre>
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}

        <p className="small muted" style={{ marginTop: "12px" }}>Code and outputs: NLP_Pipeline.ipynb, section 11</p>
        <div style={{ display: "flex", gap: "10px", marginTop: "10px" }}>
          <CodeToggle name="build_prompt" label="Show prompt builder" />
          <CodeToggle name="make_reply" label="Show reply generator" />
        </div>
      </Card>
    </div>
  );
}
