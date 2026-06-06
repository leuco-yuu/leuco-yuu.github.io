(function () {
  const dock = document.getElementById("dock");
  const nav = document.getElementById("dock-nav");
  const panel = document.getElementById("dock-toc-panel");
  const toggle = document.getElementById("dock-toc-toggle");
  const current = document.getElementById("dock-toc-current");

  if (!dock || !nav || !panel || !toggle || !current) return;

  let links = [];
  let headings = [];
  let open = false;
  let scrollFrame = 0;
  let resizeFrame = 0;
  let closeTimer = 0;
  const CLOSE_DURATION = 280;

  function init() {
    collectEntries();
    updatePanelBounds();
    updateActiveHeading();

    toggle.addEventListener("click", handleToggle);
    panel.addEventListener("click", handlePanelClick);
    document.addEventListener("click", handleOutsideClick);
    document.addEventListener("keydown", handleKeydown);
    window.addEventListener("scroll", scheduleActiveUpdate, { passive: true });
    window.addEventListener("resize", scheduleResize, { passive: true });
  }

  function collectEntries() {
    links = Array.from(panel.querySelectorAll("#TableOfContents a[href^='#']"));
    headings = links
      .map((link) => {
        const id = decodeHash(link.getAttribute("href"));
        const heading = id ? document.getElementById(id) : null;
        return heading ? { link, heading } : null;
      })
      .filter(Boolean);
  }

  function decodeHash(hash) {
    if (!hash || hash === "#") return "";
    try {
      return decodeURIComponent(hash.slice(1));
    } catch (_) {
      return hash.slice(1);
    }
  }

  function handleToggle(event) {
    event.preventDefault();
    event.stopPropagation();
    setOpen(!open);
  }

  function handlePanelClick(event) {
    const link = event.target.closest("a[href^='#']");
    if (!link) return;
    setOpen(false);
  }

  function handleOutsideClick(event) {
    if (!open || dock.contains(event.target)) return;
    setOpen(false);
  }

  function handleKeydown(event) {
    if (event.key === "Escape" && open) {
      setOpen(false);
      toggle.focus();
    }
  }

  function setOpen(nextOpen) {
    open = nextOpen;
    window.clearTimeout(closeTimer);
    toggle.setAttribute("aria-expanded", String(open));
    toggle.classList.toggle("dock-toc-active", open);

    if (open) {
      updatePanelBounds();
      panel.setAttribute("aria-hidden", "false");
      window.requestAnimationFrame(() => panel.classList.add("is-open"));
      scrollActiveLinkIntoView();
      return;
    }

    panel.classList.remove("is-open");
    closeTimer = window.setTimeout(() => {
      if (!open) panel.setAttribute("aria-hidden", "true");
    }, CLOSE_DURATION);
  }

  function scheduleActiveUpdate() {
    if (scrollFrame) return;
    scrollFrame = window.requestAnimationFrame(() => {
      scrollFrame = 0;
      updateActiveHeading();
    });
  }

  function updateActiveHeading() {
    if (headings.length === 0) {
      collectEntries();
      if (headings.length === 0) return;
    }

    const activationLine = Math.max(96, window.innerHeight * 0.28);
    let active = headings[0];

    headings.forEach((entry) => {
      if (entry.heading.getBoundingClientRect().top <= activationLine) {
        active = entry;
      }
    });

    links.forEach((link) => {
      const isActive = link === active.link;
      link.classList.toggle("active", isActive);
      if (isActive) link.setAttribute("aria-current", "location");
      else link.removeAttribute("aria-current");
    });

    current.textContent = getDisplayNumber(active.heading);
    current.title = active.heading.textContent.trim();

    if (open) scrollActiveLinkIntoView();
  }

  function getDisplayNumber(heading) {
    const number = String(heading.dataset.headingNumber || "").trim();
    if (!number) return "目录";

    const depth = Number.parseInt(heading.dataset.headingDepth || "1", 10);
    if (depth <= 1) return number.replace(/[、\s]+$/u, "");
    if (depth <= 3) return number.replace(/\s+$/u, "");

    const parts = number.split(".");
    const trimmed = parts.length > 3 ? parts.slice(0, 3).join(".") : number;
    return trimmed.replace(/\s+$/u, "");
  }

  function scrollActiveLinkIntoView() {
    const activeLink = panel.querySelector("#TableOfContents a.active");
    activeLink?.scrollIntoView({ block: "nearest" });
  }

  function scheduleResize() {
    if (resizeFrame) return;
    resizeFrame = window.requestAnimationFrame(() => {
      resizeFrame = 0;
      updatePanelBounds();
    });
  }

  function updatePanelBounds() {
    const navRect = nav.getBoundingClientRect();
    const header = document.querySelector(".site-header");
    const headerBottom = header?.getBoundingClientRect().bottom || 0;
    const availableMiddleHeight = Math.max(180, navRect.top - headerBottom - 24);
    const maxHeight = Math.max(160, availableMiddleHeight * (2 / 3));

    panel.style.width = `${navRect.width}px`;
    panel.style.setProperty("--dock-toc-max-height", `${maxHeight}px`);
    panel.style.setProperty("--dock-toc-target-height", `${Math.min(panel.scrollHeight, maxHeight)}px`);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init, { once: true });
  } else {
    init();
  }
})();
