import { copyText } from "./clipboard.js";

const COPY_FEEDBACK_MS = 2000;
const copyTimers = new WeakMap();

function getCodeBlocks(root = document) {
  return [...root.querySelectorAll("[data-code-block]")];
}

function countSourceLines(block) {
  const sourceNode = block.querySelector("[data-code-source]");
  const sourceText = sourceNode?.textContent ?? "";
  if (!sourceText) return 0;
  return sourceText.split("\n").length;
}

function shouldAutoCollapse(block) {
  const autoCollapseLines = Number.parseInt(block.dataset.autoCollapseLines || "30", 10);
  const autoCollapseHeight = Number.parseInt(block.dataset.autoCollapseHeight || "400", 10);
  const renderedCode = block.querySelector(
    ".code-block-content pre.chroma, .code-block-content pre"
  );
  const renderedHeight = renderedCode?.offsetHeight || 0;
  const sourceLines = countSourceLines(block);

  return sourceLines > autoCollapseLines || renderedHeight > autoCollapseHeight;
}

function updateCollapseButton(block, collapsed) {
  const buttons = block.querySelectorAll("[data-code-action='toggle-collapse']");
  buttons.forEach((button) => {
    const label = collapsed ? button.dataset.labelExpand : button.dataset.labelCollapse;
    const text = button.querySelector(".collapse-text");
    const chevron = button.querySelector(".collapse-chevron");

    if (text) text.textContent = label || "";
    if (chevron) chevron.classList.toggle("rotate-180", !collapsed);

    button.title = label || "";
    button.setAttribute("aria-label", label || "");
    button.setAttribute("aria-expanded", String(!collapsed));
  });
}

function updateCollapseOverlay(block, collapsed) {
  const isOutput = block.matches(".code-block-output");
  const overlay = block.querySelector(".collapse-overlay");
  const bottomBtn = block.querySelector(".code-block-bottom-collapse");
  const chevron = block.querySelector(".output-chevron");

  if (overlay) {
    if (isOutput) {
      overlay.hidden = true;
      overlay.style.display = "none";
    } else {
      overlay.hidden = !collapsed;
      overlay.classList.toggle("pointer-events-none", !collapsed);
      overlay.classList.toggle("opacity-0", !collapsed);
      overlay.classList.toggle("pointer-events-auto", collapsed);
      overlay.classList.toggle("opacity-100", collapsed);
    }
  }

  // Output blocks: bottom button visible when EXPANDED *only if* block was auto-collapsed on init
  // (i.e. long enough output — short outputs don't need the button)
  if (bottomBtn) {
    if (isOutput) {
      var canShowO = !collapsed && block.dataset._isCollapsible === "true";
      bottomBtn.hidden = !canShowO;
      bottomBtn.classList.toggle("hidden", !canShowO);
    } else {
      var canShowN = !collapsed && (block.dataset._isCollapsible === "true" || block.dataset._userToggled === "true");
      bottomBtn.hidden = !canShowN;
      bottomBtn.classList.toggle("hidden", !canShowN);
    }

    var iconSpan = bottomBtn.querySelector(".bottom-collapse-icon");
    if (iconSpan) {
      iconSpan.innerHTML = collapsed
        ? '<svg class="h-5 w-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7"/></svg>'
        : '<svg class="h-5 w-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 15l7-7 7 7"/></svg>';
    }
    var label = collapsed ? bottomBtn.dataset.labelExpand : bottomBtn.dataset.labelCollapse;
    bottomBtn.title = label || "";
    bottomBtn.setAttribute("aria-label", label || "");
  }

  // Rotate output header chevron
  if (chevron) {
    chevron.style.transform = collapsed ? "" : "rotate(180deg)";
  }
}

function setCollapsed(block, collapsed, { animate = true } = {}) {
  const content = block.querySelector(".code-block-content");
  if (!content) return;

  const collapsedHeight = Number.parseInt(block.dataset.collapsedHeight || "120", 10);

  block.dataset.collapsed = String(collapsed);
  content.style.transition = animate ? "max-height 0.35s cubic-bezier(.4,0,.2,1)" : "";
  content.style.maxHeight = collapsed ? `${collapsedHeight}px` : "";
  content.style.overflow = collapsed ? "hidden" : "";

  updateCollapseButton(block, collapsed);
  updateCollapseOverlay(block, collapsed);

  if (animate) {
    window.setTimeout(() => {
      content.style.transition = "";
    }, 350);
  }
}

function initCollapsibleBlocks(root = document) {
  getCodeBlocks(root).forEach((block) => {
    if (block.dataset.collapsible !== "true") return;

    const defaultState = block.dataset.defaultState || "expanded";
    const collapsedAttr = block.dataset.collapsed === "true";
    const autoCollapses = shouldAutoCollapse(block);
    const startCollapsed =
      collapsedAttr || defaultState === "collapsed" || autoCollapses;
    
    // Mark if block actually needs collapse UI (exceeds thresholds), regardless of forced state
    if (autoCollapses) block.dataset._isCollapsible = "true";
    
    setCollapsed(block, startCollapsed, { animate: false });
  });
}

function getCodeSource(block) {
  const sourceNode = block.querySelector("[data-code-source]");
  return sourceNode?.textContent ?? "";
}

function setCopyFeedback(button, copied) {
  const text = button.querySelector(".copy-text");
  const label = copied ? button.dataset.labelCopied : button.dataset.labelCopy;

  if (text) text.textContent = label || "";

  button.classList.toggle("text-green-600", copied);
  button.title = label || "";
  button.setAttribute("aria-label", label || "");
}

async function handleCopy(button) {
  const block = button.closest("[data-code-block]");
  if (!block) return;

  const copied = await copyText(getCodeSource(block));
  if (!copied) return;

  setCopyFeedback(button, true);

  const previousTimer = copyTimers.get(button);
  if (previousTimer) {
    window.clearTimeout(previousTimer);
  }

  const timer = window.setTimeout(() => {
    setCopyFeedback(button, false);
    copyTimers.delete(button);
  }, COPY_FEEDBACK_MS);

  copyTimers.set(button, timer);
}

function handleCodeBlockClick(event) {
  const copyButton = event.target.closest(".copy-code-btn[data-code-action='copy']");
  if (copyButton) {
    event.preventDefault();
    handleCopy(copyButton);
    return;
  }

  const collapseButton = event.target.closest(
    "[data-code-action='toggle-collapse']"
  );
  if (collapseButton) {
    event.preventDefault();
    const block = collapseButton.closest("[data-code-block]");
    if (!block) return;
    block.dataset._userToggled = "true";
    const isCollapsed = block.dataset.collapsed === "true";
    setCollapsed(block, !isCollapsed);
    return;
  }

  const expandButton = event.target.closest("[data-code-action='expand']");
  if (expandButton) {
    event.preventDefault();
    const block = expandButton.closest("[data-code-block]");
    if (!block) return;
    block.dataset._userToggled = "true";
    setCollapsed(block, false);
    return;
  }
}

let initialized = false;

export function initCodeBlocks(root = document) {
  if (initialized) return;
  initialized = true;

  initCollapsibleBlocks(root);

  document.addEventListener("click", handleCodeBlockClick);
}
