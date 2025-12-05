(function () {
  const outEl = document.getElementById("test-output");

  function log(message) {
    if (outEl) {
      outEl.textContent += message + "\n";
    } else {
      console.log(message);
    }
  }

  const helpers = window.__LLM_UI__;
  if (!helpers) {
    log("ERROR: __LLM_UI__ helpers not found.");
    return;
  }

  const { formatTimestamp, extractTextAndTimestamp } = helpers;

  function runTests() {
    log("Running LLM UI tests...");

    // Test formatTimestamp
    const fixedDate = new Date("2025-12-05T09:03:07Z");
    const ts = formatTimestamp(fixedDate);
    console.assert(typeof ts === "string", "formatTimestamp should return string");
    log(`formatTimestamp output: ${ts}`);

    // Test extractTextAndTimestamp: standard case
    const sampleResponse = {
      text: "Hello from LLM",
      timestamp: "2025-12-05T10:00:00Z",
    };
    const extracted = extractTextAndTimestamp(sampleResponse);
    console.assert(extracted.text === "Hello from LLM", "Should extract text");
    console.assert(
      extracted.timestamp === "2025-12-05T10:00:00Z",
      "Should extract timestamp"
    );
    log(`extractTextAndTimestamp output: ${JSON.stringify(extracted)}`);

    // Fallback text field
    const fallbackResponse = { output: "Alt field" };
    const e2 = extractTextAndTimestamp(fallbackResponse);
    console.assert(e2.text === "Alt field", "Should use fallback text field");
    log(`extractTextAndTimestamp fallback output: ${JSON.stringify(e2)}`);

    // Null / undefined case
    const e3 = extractTextAndTimestamp(null);
    console.assert(typeof e3.text === "string", "Null response text is string");
    console.assert(e3.text.length === 0, "Null response text is empty string");
    log(`extractTextAndTimestamp null output: ${JSON.stringify(e3)}`);

    log("All basic tests executed. Check console for assertion errors.");
  }

  runTests();
})();
