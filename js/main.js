/**
 * ==========================================================================
 * SOCIAL MEDIA USAGE RESEARCH PROJECT - MAIN JAVASCRIPT
 * Description: Vanilla JavaScript for UI interactions, multi-page navigation, 
 *              and responsive controls. Zero external dependencies.
 * ==========================================================================
 */

// Wait until the HTML document is fully loaded before executing scripts
document.addEventListener("DOMContentLoaded", () => {

    /* ----------------------------------------------------------------------
       1. Mobile Navigation Menu Toggle
       ---------------------------------------------------------------------- */
    const menuToggleBtn = document.querySelector(".menu-toggle");
    const navLinks = document.querySelector(".nav-links");

    if (menuToggleBtn && navLinks) {
        menuToggleBtn.addEventListener("click", () => {
            navLinks.classList.toggle("show");
            
            // Toggle aria-expanded for screen-reader accessibility
            const isExpanded = navLinks.classList.contains("show");
            menuToggleBtn.setAttribute("aria-expanded", isExpanded);
        });

        // Close mobile menu when a navigation link is clicked
        navLinks.querySelectorAll("a").forEach(link => {
            link.addEventListener("click", () => {
                if (navLinks.classList.contains("show")) {
                    navLinks.classList.remove("show");
                    menuToggleBtn.setAttribute("aria-expanded", "false");
                }
            });
        });
    }

    /* ----------------------------------------------------------------------
       2. Active Page Highlighting for Multi-Page Navigation
       Detects current page filename and applies the active class.
       ---------------------------------------------------------------------- */
    const currentPath = window.location.pathname;
    const currentFile = currentPath.substring(currentPath.lastIndexOf("/") + 1) || "index.html";
    const navItems = document.querySelectorAll(".nav-links a");

    navItems.forEach(link => {
        const href = link.getAttribute("href");
        if (href && !href.startsWith("http") && !href.startsWith("#")) {
            const linkFile = href.split("#")[0];
            if (linkFile === currentFile || (currentFile === "" && linkFile === "index.html")) {
                link.classList.add("active");
            }
        }
    });

    /* ----------------------------------------------------------------------
       3. Interactive Topic Pillar Cards (Research Page)
       Clicking a topic card toggles an active state and highlights
       the corresponding potential variable tags.
       ---------------------------------------------------------------------- */
    const topicCards = document.querySelectorAll(".topic-card");

    if (topicCards.length > 0) {
        topicCards.forEach(card => {
            card.addEventListener("click", () => {
                // Remove active state from all cards, then activate clicked card
                topicCards.forEach(c => c.classList.remove("active"));
                card.classList.add("active");

                // Highlight corresponding variable badges
                const topicType = card.getAttribute("data-topic");
                highlightVariablesForTopic(topicType);
            });
        });
    }

    /**
     * Highlights corresponding variable badges when a topic card is selected.
     * @param {string} topic - The data-topic attribute key
     */
    function highlightVariablesForTopic(topic) {
        const variableTags = document.querySelectorAll(".variable-tag");
        
        variableTags.forEach(tag => {
            const tagTopics = tag.getAttribute("data-topics") || "";
            if (topic && tagTopics.includes(topic)) {
                tag.style.borderColor = "var(--primary-color)";
                tag.style.backgroundColor = "var(--primary-light)";
            } else {
                tag.style.borderColor = "var(--border-color)";
                tag.style.backgroundColor = "var(--bg-alt)";
            }
        });
    }

    /* ----------------------------------------------------------------------
       4. Log confirmation for development
       ---------------------------------------------------------------------- */
    console.log("Research website loaded successfully. Multi-page mode active.");
});
