// Adds a right-click context menu item: "Analyze with Fake News Detector"
// so users can select text on any page and quickly check it.

chrome.runtime.onInstalled.addListener(() => {
  chrome.contextMenus.create({
    id: "analyze-selection",
    title: "Analyze with Fake News Detector",
    contexts: ["selection"],
  });
});

chrome.contextMenus.onClicked.addListener((info, tab) => {
  if (info.menuItemId === "analyze-selection" && info.selectionText) {
    // Store the selection so the popup can pick it up when opened.
    chrome.storage.local.set({ pendingSelection: info.selectionText });
    chrome.action.openPopup?.(); // Not supported in all Chrome versions; safe no-op otherwise.
  }
});
