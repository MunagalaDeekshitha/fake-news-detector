// Content script — runs on every page.
// Currently minimal: the popup pulls selected text via chrome.scripting.executeScript
// directly, so this file is a placeholder you can extend, e.g. to:
//   - auto-detect the main <article> text on news sites
//   - inject an inline badge next to headlines
//   - listen for right-click "Analyze selection" context menu results

console.log("Fake News Detector content script loaded.");
