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
    initTocCollapse();
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

  function initTocCollapse() {
    var toc = panel.querySelector("#TableOfContents");
    if (!toc) return;
    var allLis = toc.querySelectorAll("li");
    allLis.forEach(function (li) {
      var childUl = li.querySelector(":scope > ul");
      if (!childUl) return;
      // Create triangle button
      var btn = document.createElement("span");
      btn.className = "toc-toggle";
      btn.setAttribute("role", "button");
      btn.setAttribute("tabindex", "0");
      btn.setAttribute("aria-expanded", "true");
      btn.setAttribute("aria-label", "折叠");
      btn.innerHTML = '<svg width="10" height="10" viewBox="0 0 10 10"><path d="M2 3l3 4 3-4" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/></svg>';
      // Insert before the first child
      li.insertBefore(btn, li.firstChild);
      li.classList.add("toc-parent");
      // Click handler
      btn.addEventListener("click", function (e) {
        e.stopPropagation();
        e.preventDefault();
        toggleTocItem(li);
        // Resize panel after toggle
        window.requestAnimationFrame(function () {
          updatePanelBounds();
        });
      });
      btn.addEventListener("keydown", function (e) {
        if (e.key === "Enter" || e.key === " ") {
          e.preventDefault();
          toggleTocItem(li);
          window.requestAnimationFrame(function () {
            updatePanelBounds();
          });
        }
      });
    });

    // Default collapse behavior differs for archive vs article pages
    var isArchive = toc.querySelector("a[href^='#year-']") !== null;

    var topUl = toc.querySelector(":scope > ul");
    if (topUl) {
      if (isArchive) {
        // Archive: collapse only article level (H3), keep year (H1) + month (H2) visible
        var monthItems = topUl.querySelectorAll("li li.toc-parent");
        monthItems.forEach(function (monthLi) {
          var childUl = monthLi.querySelector(":scope > ul");
          var btn = monthLi.querySelector(":scope > .toc-toggle");
          if (childUl && btn) {
            childUl.classList.add("toc-collapsed");
            btn.classList.add("toc-collapsed");
            btn.setAttribute("aria-expanded", "false");
          }
        });
      } else {
        // Article: collapse all, only show H1
        var rootLis = topUl.querySelectorAll(":scope > li.toc-parent");
        rootLis.forEach(function (rootLi) {
          var childUl = rootLi.querySelector(":scope > ul");
          var btn = rootLi.querySelector(":scope > .toc-toggle");
          if (childUl && btn) {
            childUl.classList.add("toc-collapsed");
            btn.classList.add("toc-collapsed");
            btn.setAttribute("aria-expanded", "false");
          }
        });
        // Also collapse all deeper levels
        var allNested = topUl.querySelectorAll("ul ul");
        allNested.forEach(function (ul) {
          ul.classList.add("toc-collapsed");
          var parentLi = ul.parentElement;
          var btn = parentLi.querySelector(":scope > .toc-toggle");
          if (btn) {
            btn.classList.add("toc-collapsed");
            btn.setAttribute("aria-expanded", "false");
          }
        });
      }
    }
  }

  function toggleTocItem(li) {
    var childUl = li.querySelector(":scope > ul");
    var btn = li.querySelector(":scope > .toc-toggle");
    if (!childUl || !btn) return;
    var collapsed = childUl.classList.toggle("toc-collapsed");
    btn.setAttribute("aria-expanded", String(!collapsed));
    btn.classList.toggle("toc-collapsed", collapsed);
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

    panel.style.width = navRect.width + "px";
    panel.style.setProperty("--dock-toc-max-height", maxHeight + "px");
    var toc = panel.querySelector("#TableOfContents");
    var contentHeight = toc ? toc.offsetHeight : panel.scrollHeight;
    var targetHeight = Math.min(contentHeight, maxHeight);
    panel.style.setProperty("--dock-toc-target-height", targetHeight + "px");
  }

  // Expose sync function for client-side sort/pagination
  window.syncTocOrder = function () {
    var toc = panel.querySelector("#TableOfContents");
    if (!toc) return;
    var tocItems = toc.querySelectorAll("li > a[href^='#card-']");
    if (tocItems.length === 0) return;
    var tocUl = toc.querySelector("ul");
    if (!tocUl) return;

    var cards = document.querySelectorAll("article[id^='card-']");
    var orderMap = {};
    cards.forEach(function (card, i) { orderMap[card.id] = i; });

    var items = Array.prototype.slice.call(tocItems);
    items.sort(function (a, b) {
      var idA = a.getAttribute("href").replace("#", "");
      var idB = b.getAttribute("href").replace("#", "");
      return (orderMap[idA] || 0) - (orderMap[idB] || 0);
    });

    items.forEach(function (item) { tocUl.appendChild(item.parentElement); });
    updatePanelBounds();
  };

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init, { once: true });
  } else {
    init();
  }
})();
