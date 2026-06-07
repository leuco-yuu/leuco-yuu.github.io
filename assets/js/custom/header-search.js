function initHeaderSearch() {
  const normalizePath = (href) => {
    try {
      const path = new URL(href, window.location.origin).pathname;
      return path.endsWith("/") ? path : `${path}/`;
    } catch (_) {
      return "";
    }
  };

  const isCurrentNavPath = (href) => {
    const targetPath = normalizePath(href);
    const currentPath = normalizePath(window.location.href);
    const sectionRoots = ["/posts/", "/projects/", "/categories/", "/series/", "/tags/"];

    if (!targetPath || targetPath === "/") return currentPath === targetPath;
    if (currentPath === targetPath) return true;
    return sectionRoots.includes(targetPath) && currentPath.startsWith(targetPath);
  };

  document.querySelectorAll(".site-header").forEach((header) => {
    if (header.dataset.headerSearchBound === "true") return;
    header.dataset.headerSearchBound = "true";

    const toggles = Array.from(header.querySelectorAll(".header-search-toggle"));
    const fields = Array.from(header.querySelectorAll(".header-search-field"));
    const controls = Array.from(header.querySelectorAll(".header-search-control"));
    const clearButtons = controls.map((control) => {
      const existing = control.querySelector(".header-search-clear");
      if (existing) return existing;

      const button = document.createElement("button");
      button.type = "button";
      button.className = "header-search-clear";
      button.setAttribute("aria-label", "清空搜索");
      button.hidden = true;

      const icon = document.createElement("span");
      icon.setAttribute("aria-hidden", "true");
      icon.textContent = "×";
      button.appendChild(icon);
      control.appendChild(button);
      return button;
    });
    const mobileSearchItems = Array.from(document.querySelectorAll(".mobile-search-only"));
    const isHome = header.dataset.pageHome === "true";
    let geometryFrame = 0;

    const getPanelForToggle = (toggle) => {
      const id = toggle?.getAttribute("data-submenu-id");
      const scope = toggle?.closest(".desktop-search-menu, .desktop-main-nav");
      if (!id || !scope) return null;

      return Array.from(scope.querySelectorAll(".nav-submenu")).find(
        (panel) => panel.getAttribute("data-submenu-id") === id
      );
    };

    const setSubmenuOpen = (toggle, panel, open) => {
      if (!toggle || !panel) return;

      panel.classList.toggle("hidden", !open);
      toggle.setAttribute("aria-expanded", String(open));
      toggle.querySelector(".submenu-chevron")?.classList.toggle("rotate-180", open);
    };

    const updateActiveNavState = () => {
      const panel = document.getElementById("mobile-nav-panel");
      const mobileToggle = document.getElementById("mobile-nav-toggle");
      const candidates = Array.from(
        document.querySelectorAll(
          ".site-header a.nav-link[href], .site-header a.desktop-search-icon-link[href], .site-header a.mobile-quick-link[href], #mobile-nav-panel a.nav-panel-link[href]",
        ),
      );

      document.querySelectorAll("[data-nav-active]").forEach((element) => {
        element.removeAttribute("data-nav-active");
      });
      document.querySelectorAll("[data-nav-parent-active]").forEach((element) => {
        element.removeAttribute("data-nav-parent-active");
      });

      candidates.forEach((link) => {
        if (!isCurrentNavPath(link.href)) return;

        link.setAttribute("data-nav-active", "true");

        const submenu = link.closest(".nav-submenu");
        const submenuToggle = submenu
          ?.closest(".relative, .desktop-search-menu")
          ?.querySelector(".nav-submenu-toggle");
        submenuToggle?.setAttribute("data-nav-parent-active", "true");

        const accordion = link.closest(".nav-accordion-group");
        accordion
          ?.querySelector(".nav-accordion-toggle")
          ?.setAttribute("data-nav-parent-active", "true");
      });

      if (panel && mobileToggle) {
        const activePanelLinks = Array.from(panel.querySelectorAll('[data-nav-active="true"]'));
        const panelOwnsCurrentPage = activePanelLinks.some(
          (link) =>
            !link.classList.contains("mobile-search-only") ||
            header.classList.contains("is-search-open"),
        );
        mobileToggle.toggleAttribute("data-nav-parent-active", panelOwnsCurrentPage);
      }
    };

    const setOpen = (open, { focus = true } = {}) => {
      const desktopMoreToggle = header.querySelector(".desktop-main-nav .nav-submenu-toggle");
      const desktopMorePanel = getPanelForToggle(desktopMoreToggle);
      const searchMoreToggle = header.querySelector(".desktop-search-menu .nav-submenu-toggle");
      const searchMorePanel = getPanelForToggle(searchMoreToggle);
      const desktopMoreWasOpen = Boolean(desktopMorePanel && !desktopMorePanel.classList.contains("hidden"));
      const searchMoreWasOpen = Boolean(searchMorePanel && !searchMorePanel.classList.contains("hidden"));

      header.classList.toggle("is-search-open", open);
      toggles.forEach((toggle) => toggle.setAttribute("aria-pressed", String(open)));
      mobileSearchItems.forEach((item) => {
        item.classList.toggle("hidden", !open);
        item.classList.toggle("flex", open);
      });
      document.dispatchEvent(
        new CustomEvent("mobile-nav-depth:search-toggle", {
          detail: { open },
        }),
      );

      if (open && desktopMoreWasOpen) {
        setSubmenuOpen(desktopMoreToggle, desktopMorePanel, false);
        setSubmenuOpen(searchMoreToggle, searchMorePanel, true);
      }

      if (!open && searchMoreWasOpen) {
        setSubmenuOpen(searchMoreToggle, searchMorePanel, false);
        setSubmenuOpen(desktopMoreToggle, desktopMorePanel, true);
      }

      if (open && focus) {
        window.setTimeout(() => {
          const visibleField = fields.find((field) => field.offsetParent !== null);
          visibleField?.focus({ preventScroll: true });
        }, 180);
      }

      updateActiveNavState();
      scheduleSearchGeometryUpdate();
      window.requestAnimationFrame(syncClearButtons);
    };

    const updateSearchGeometry = () => {
      const visibleControl = controls.find((control) => control.offsetParent !== null);
      if (!visibleControl) return;

      const controlRect = visibleControl.getBoundingClientRect();
      const row = visibleControl.closest(".desktop-header-row, .mobile-header-row");
      const isDesktop = Boolean(row?.classList.contains("desktop-header-row"));
      const leftBoundaryElement = isDesktop
        ? header.querySelector(".desktop-search-nav")
        : row?.firstElementChild;
      const leftRect = leftBoundaryElement?.getBoundingClientRect();
      const leftBoundary = leftRect?.right || header.getBoundingClientRect().left;
      const rootFontSize =
        Number.parseFloat(window.getComputedStyle(document.documentElement).fontSize) || 16;
      const fieldButtonGap = rootFontSize * 0.5;
      const safeGap = isDesktop ? rootFontSize * 1.125 : 10;
      const maxWidth = isDesktop ? 320 : 288;
      const minWidth = 0;
      const fieldRight = controlRect.left - fieldButtonGap;
      const available = Math.floor(fieldRight - leftBoundary - safeGap);
      const width = Math.max(minWidth, Math.min(maxWidth, available));

      header.style.setProperty("--header-search-width", `${width}px`);
    };

    const scheduleSearchGeometryUpdate = () => {
      if (geometryFrame) return;
      geometryFrame = window.requestAnimationFrame(() => {
        geometryFrame = 0;
        updateSearchGeometry();
      });
    };

    const syncClearButtons = () => {
      const hasValue = fields.some((field) => field.value.trim().length > 0);
      clearButtons.forEach((button) => {
        button.hidden = !hasValue;
      });
    };

    const clearSearchFields = () => {
      const activeField = fields.find((field) => field.offsetParent !== null) || fields[0];
      fields.forEach((field) => {
        field.value = "";
      });
      activeField?.dispatchEvent(new Event("input", { bubbles: true }));
      activeField?.focus({ preventScroll: true });
      syncClearButtons();
    };

    controls.forEach((control) => {
      control.addEventListener("click", (event) => {
        event.stopPropagation();
      });
    });

    clearButtons.forEach((button) => {
      button.addEventListener("click", (event) => {
        event.preventDefault();
        event.stopPropagation();
        clearSearchFields();
      });
    });

    fields.forEach((field) => {
      field.addEventListener("input", syncClearButtons);
      field.addEventListener("search", syncClearButtons);
    });

    toggles.forEach((toggle) => {
      toggle.addEventListener("click", (event) => {
        event.preventDefault();
        event.stopPropagation();

        if (isHome) {
          if (window.Search?.show) {
            window.Search.show();
          } else {
            document.dispatchEvent(
              new CustomEvent("search:open", {
                detail: { origin: "header-search", focus: true },
              }),
            );
          }
          return;
        }

        setOpen(!header.classList.contains("is-search-open"));
      });
    });

    document.addEventListener("inline-search:open", (event) => {
      setOpen(true, { focus: event?.detail?.focus !== false });
    });

    document.addEventListener("keydown", (event) => {
      if (event.key === "Escape" && header.classList.contains("is-search-open")) {
        setOpen(false);
        syncClearButtons();
      }
    });

    window.addEventListener("resize", scheduleSearchGeometryUpdate, { passive: true });
    window.addEventListener("pagehide", () => {
      if (header.classList.contains("is-search-open")) {
        setOpen(false, { focus: false });
      }
    });
    window.addEventListener("pageshow", () => {
      if (header.classList.contains("is-search-open")) {
        setOpen(false, { focus: false });
      }
    });

    syncClearButtons();
    updateSearchGeometry();
    updateActiveNavState();
  });
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", initHeaderSearch, { once: true });
} else {
  initHeaderSearch();
}
