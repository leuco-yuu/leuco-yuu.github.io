// Implements a scroll spy system for the ToC, displaying the current section with an indicator and scrolling to it when needed.

// Inspired from https://gomakethings.com/debouncing-your-javascript-events/
function debounced(func: Function) {
    let timeout;
    return () => {
        if (timeout) {
            window.cancelAnimationFrame(timeout);
        }

        timeout = window.requestAnimationFrame(() => func());
    }
}

const headersQuery = ".article-content h1[id], .article-content h2[id], .article-content h3[id], .article-content h4[id], .article-content h5[id], .article-content h6[id]";
const tocQuery = "#TableOfContents";
const navigationQuery = "#TableOfContents li";
const activeClass = "active-class";

function scrollToTocElement(tocElement: HTMLElement, scrollableNavigation: HTMLElement) {
    let textHeight = tocElement.querySelector("a").offsetHeight;
    let scrollTop = tocElement.offsetTop - scrollableNavigation.offsetHeight / 2 + textHeight / 2 - scrollableNavigation.offsetTop;
    if (scrollTop < 0) {
        scrollTop = 0;
    }
    scrollableNavigation.scrollTo({ top: scrollTop, behavior: "smooth" });
}

type IdToElementMap = { [key: string]: HTMLElement };

function buildIdToNavigationElementMap(navigation: NodeListOf<Element>): IdToElementMap {
    const sectionLinkRef: IdToElementMap = {};
    navigation.forEach((navigationElement: HTMLElement) => {
        const link = navigationElement.querySelector("a");
        if (link) {
            const href = link.getAttribute("href");
            if (href && href.startsWith("#")) {
                sectionLinkRef[href.slice(1)] = navigationElement;
            }
        }
    });

    return sectionLinkRef;
}

function computeOffsets(headers: NodeListOf<Element>) {
    let sectionsOffsets = [];
    headers.forEach((header: HTMLElement) => { 
        if (header.id) {
            sectionsOffsets.push({ id: header.id, offset: header.offsetTop }) 
        }
    });
    sectionsOffsets.sort((a, b) => a.offset - b.offset);
    return sectionsOffsets;
}

function setupScrollspy() {
    let headers = document.querySelectorAll(headersQuery);
    if (!headers || headers.length === 0) {
        console.warn("No header matched query", headersQuery);
        return;
    }

    // 查找所有包含目录的元素（移动端和桌面端）
    let tocContainers = document.querySelectorAll('.widget--toc');
    if (!tocContainers || tocContainers.length === 0) {
        console.warn("No toc container found");
        return;
    }

    // 找到当前可见的目录容器
    let scrollableNavigation: HTMLElement | null = null;
    for (let i = 0; i < tocContainers.length; i++) {
        const container = tocContainers[i] as HTMLElement;
        // 检查元素是否可见（不在隐藏的元素内）
        if (container.offsetParent !== null && 
            container.getBoundingClientRect().width > 0 && 
            container.getBoundingClientRect().height > 0) {
            scrollableNavigation = container;
            break;
        }
    }
    
    // 如果没找到可见的，使用第一个
    if (!scrollableNavigation) {
        scrollableNavigation = tocContainers[0] as HTMLElement;
    }

    let navigation = scrollableNavigation.querySelectorAll(navigationQuery);
    if (!navigation || navigation.length === 0) {
        console.warn("No navigation matched query", navigationQuery);
        return;
    }

    let sectionsOffsets = computeOffsets(headers);

    // We need to avoid scrolling when the user is actively interacting with the ToC. Otherwise, if the user clicks on a link in the ToC,
    // we would scroll their view, which is not optimal usability-wise.
    let tocHovered: boolean = false;
    scrollableNavigation.addEventListener("mouseenter", debounced(() => tocHovered = true));
    scrollableNavigation.addEventListener("mouseleave", debounced(() => tocHovered = false));

    let activeSectionLink: Element;

    let idToNavigationElement: IdToElementMap = buildIdToNavigationElementMap(navigation);

    function scrollHandler() {
        // 重新检查当前可见的目录容器（处理响应式切换）
        let visibleToc: HTMLElement | null = null;
        for (let i = 0; i < tocContainers.length; i++) {
            const container = tocContainers[i] as HTMLElement;
            if (container.offsetParent !== null && 
                container.getBoundingClientRect().width > 0 && 
                container.getBoundingClientRect().height > 0) {
                visibleToc = container;
                break;
            }
        }
        
        // 如果可见的目录发生变化，更新相关变量
        if (visibleToc && visibleToc !== scrollableNavigation) {
            scrollableNavigation = visibleToc;
            navigation = scrollableNavigation.querySelectorAll(navigationQuery);
            idToNavigationElement = buildIdToNavigationElementMap(navigation);
            
            // 重置活动状态
            if (activeSectionLink) {
                activeSectionLink.classList.remove(activeClass);
                activeSectionLink = null;
            }
        }

        let scrollPosition = document.documentElement.scrollTop || document.body.scrollTop;

        let newActiveSection: HTMLElement | undefined;

        // Find the section that is currently active.
        // It is possible for no section to be active, so newActiveSection may be undefined.
        sectionsOffsets.forEach((section) => {
            if (scrollPosition >= section.offset - 20) {
                newActiveSection = document.getElementById(section.id);
            }
        });

        // Find the link for the active section. Once again, there are a few edge cases:
        // - No active section = no link => undefined
        // - No active section but the link does not exist in toc (e.g. because it is outside of the applicable ToC levels) => undefined
        let newActiveSectionLink: HTMLElement | undefined
        if (newActiveSection) {
            newActiveSectionLink = idToNavigationElement[newActiveSection.id];
        }

        if (newActiveSection && !newActiveSectionLink) {
            // The active section does not have a link in the ToC, so we can't scroll to it.
            console.debug("No link found for section", newActiveSection);
        } else if (newActiveSectionLink !== activeSectionLink) {
            if (activeSectionLink)
                activeSectionLink.classList.remove(activeClass);
            if (newActiveSectionLink) {
                newActiveSectionLink.classList.add(activeClass);
                if (!tocHovered && scrollableNavigation) {
                    // Scroll so that newActiveSectionLink is in the middle of scrollableNavigation, except when it's from a manual click (hence the tocHovered check)
                    scrollToTocElement(newActiveSectionLink, scrollableNavigation);
                }
            }
            activeSectionLink = newActiveSectionLink;
        }
    }

    window.addEventListener("scroll", debounced(scrollHandler));
    
    // Resizing may cause the offset values to change: recompute them.
    function resizeHandler() {
        sectionsOffsets = computeOffsets(headers);
        
        // 窗口大小变化时，重新检查可见的目录容器
        let visibleToc: HTMLElement | null = null;
        for (let i = 0; i < tocContainers.length; i++) {
            const container = tocContainers[i] as HTMLElement;
            if (container.offsetParent !== null && 
                container.getBoundingClientRect().width > 0 && 
                container.getBoundingClientRect().height > 0) {
                visibleToc = container;
                break;
            }
        }
        
        if (visibleToc && visibleToc !== scrollableNavigation) {
            scrollableNavigation = visibleToc;
            navigation = scrollableNavigation.querySelectorAll(navigationQuery);
            idToNavigationElement = buildIdToNavigationElementMap(navigation);
            
            // 重置活动状态
            if (activeSectionLink) {
                activeSectionLink.classList.remove(activeClass);
                activeSectionLink = null;
            }
        }
        
        scrollHandler();
    }

    // Use ResizeObserver to detect changes in the size of .article-content
    const articleContent = document.querySelector(".article-content");
    if (articleContent) {
        const resizeObserver = new ResizeObserver(debounced(resizeHandler));
        resizeObserver.observe(articleContent);
    }

    window.addEventListener("resize", debounced(resizeHandler));
    
    // 初始调用一次，设置初始状态
    setTimeout(scrollHandler, 100);
}

export { setupScrollspy };
